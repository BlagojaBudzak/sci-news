"""
Dynamic search pipeline for Sci News.

Usage:
    python search.py "solid-state batteries"

Flow:
    query -> QueryInterpreter -> OpenAlex (override) + arXiv (free-form)
          -> combine -> dedup -> prefilter (keywords from QueryPlan)
          -> Reviewer -> Writer -> publish to search digest folder.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import arxiv

from config import (
    DATA_DIGEST_DIR,
    DATA_TRACE_DIR,
    SITE_DIGEST_DIR,
    MAX_PAPERS_PER_SOURCE,
)
from src.crew_setup import (
    ReviewerSelection,
    WriterOutput,
    build_review_crew,
    build_write_crew,
    get_local_llm,
    hydrate_digest_output,
    select_papers,
)
from src.digest_writer import write_digest
from src.openalex import fetch_openalex_papers
from src.prefilter import PrefilterConfig, deduplicate_papers, filter_papers
from src.query_interpreter import interpret_query,build_arxiv_query, build_openalex_query
from src.trace import PipelineTrace


# ---------------------------------------------------------------------------
# arXiv free‑form fetch (secondary source)
# ---------------------------------------------------------------------------
def fetch_arxiv_search(query: str, lookback_days: int) -> list[dict]:
    """Fetch recent arXiv papers matching a free-form query string."""
    if not query.strip():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    client = arxiv.Client(page_size=10, delay_seconds=5, num_retries=5)

    search = arxiv.Search(
        query=query,
        max_results=MAX_PAPERS_PER_SOURCE,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    try:
        for result in client.results(search):
            published = result.published
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if published < cutoff:
                continue
            papers.append({
                "id": result.entry_id.split("/")[-1],
                "title": " ".join((result.title or "").split()),
                "abstract": " ".join((result.summary or "").split()),
                "authors": [a.name for a in result.authors],
                "published": published.date().isoformat(),
                "url": result.entry_id,
                "doi": result.doi or result.entry_id,
                "source": "arXiv",
            })
    except Exception:
        # arXiv may fail; return whatever we have
        pass
    return papers


# ---------------------------------------------------------------------------
# Helpers (duplicated from main.py to keep search.py independent)
# ---------------------------------------------------------------------------
def _parse_output(crew_output, model_type):
    """Pull a validated Pydantic object out of a CrewOutput."""
    parsed = getattr(crew_output, "pydantic", None)
    if isinstance(parsed, model_type):
        return parsed

    raw = getattr(crew_output, "raw", None)
    if not isinstance(raw, str):
        return None

    try:
        return model_type.model_validate(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _slug_from_query(query: str) -> str:
    """Create a filesystem-friendly slug from the user query."""
    slug = "".join(
        c.lower() if c.isalnum() or c == "-" else "-"
        for c in query
    ).strip("-")
    return slug[:80] or "search"


def _search_prefilter_config(plan) -> PrefilterConfig:
    """Build a prefilter config from the QueryPlan.

    The dynamic search prefilter is intentionally lenient:
    - positive_match_required=False (we don't drop papers just because they
      lack a keyword; the reviewer will decide)
    - min_relevance_score=0.0 (no hard threshold)
    """
    return PrefilterConfig(
        positive_keywords=plan.positive_keywords,
        negative_keywords=plan.negative_keywords,
        min_abstract_chars=80,
        min_relevance_score=0.0,
        positive_match_required=False,
        relevance_weight=1.0,
        recency_weight=0.5,  # dynamic search cares more about relevance
        recency_half_life_days=10.0,
        max_candidates=15,
    )


# ---------------------------------------------------------------------------
# Main search pipeline
# ---------------------------------------------------------------------------
def run_search(query: str, llm=None) -> None:
    trace = PipelineTrace.start(f"search:{query}")
    category_for_prompt = f"search results for '{query}'"
    slug = _slug_from_query(query)

    def finish() -> None:
        trace.print_summary()
        path = trace.save(DATA_TRACE_DIR)
        print(f"  trace written to {path}")

    print(f"\n=== Dynamic search: {query} ===  (job #{trace.job_id})")

    # 1. Interpret query
    print("[1/6] interpreting query ...")
    with trace.stage("query_interpretation") as s:
        plan = interpret_query(query, llm=llm)
        openalex_query = build_openalex_query(plan)
        arxiv_query = build_arxiv_query(plan)
        s.count_out = 1
        s.count_label = "plan generated"
        s.notes.append(f"search phrases: {plan.search_phrases}")
        s.notes.append(f"OpenAlex query: {openalex_query}")
        s.notes.append(f"arXiv query: {arxiv_query}")
        s.notes.append(f"positive keywords: {plan.positive_keywords}")
        s.notes.append(f"negative keywords: {plan.negative_keywords}")
        s.notes.append(f"lookback days: {plan.lookback_days}")
    print(f"  search phrases: {plan.search_phrases}")
    print(f"  OpenAlex query: {openalex_query}")
    print(f"  arXiv query: {arxiv_query}")
    print(f"  lookback days: {plan.lookback_days}")

    # 2. Fetch papers
    print("[2/6] fetching papers ...")
    with trace.stage("fetch") as s:
        # OpenAlex primary, no cache, override query
        openalex_papers = fetch_openalex_papers(
            "search",  # dummy category name
            {"openalex": {"per_page": 50, "max_results": 30}},
            lookback_days=plan.lookback_days,
            use_cache=False,
            search_query_override=openalex_query,
        )
        # arXiv secondary (free-form search)
        arxiv_papers = fetch_arxiv_search(arxiv_query, plan.lookback_days)

        papers = openalex_papers + arxiv_papers
        s.count_out = len(papers)
        s.count_label = "papers found"
        s.notes.append(f"OpenAlex: {len(openalex_papers)}, arXiv: {len(arxiv_papers)}")

    if not papers:
        print("  no papers found in the lookback window — skipping")
        trace.stages[-1].notes.append("lookback window empty")
        finish()
        return

    print(f"  sources: OpenAlex={len(openalex_papers)}, arXiv={len(arxiv_papers)}")

    # 3. Deduplicate and prefilter
    print("[3/6] deterministic pre-filter ...")
    with trace.stage("prefilter", count_in=len(papers)) as s:
        # Deduplicate first
        unique = deduplicate_papers(papers)
        s.notes.append(f"duplicates removed: {len(papers) - len(unique)}")

        prefilter_config = _search_prefilter_config(plan)
        prefilter = filter_papers(unique, prefilter_config)
        s.count_out = prefilter.output_count
        s.count_label = "candidates retained"
        s.notes.append(f"short/missing abstracts: {prefilter.missing_abstract_count}")
        s.notes.append(f"low relevance: {prefilter.low_relevance_count}")
        for example in prefilter.filtered_examples[:5]:
            s.notes.append(f"filtered: {example['title']!r} — {example['reason']}")

    print(
        f"  pre-filter: {prefilter.input_count} -> {prefilter.output_count} papers "
        f"(duplicates={prefilter.duplicate_count}, "
        f"short abstracts={prefilter.missing_abstract_count}, "
        f"low relevance={prefilter.low_relevance_count})"
    )
    candidates = prefilter.candidates
    if not candidates:
        print("  ! pre-filter removed all candidates — skipping")
        finish()
        return

    # 4. Reviewer
    print("[4/6] reviewer selecting the top 5 ...")
    selected_papers = []
    with trace.stage("reviewer", count_in=len(candidates)) as s:
        review_crew = build_review_crew(candidates, category_for_prompt, llm=llm)
        review_result = review_crew.kickoff()
        selection = _parse_output(review_result, ReviewerSelection)
        if selection is None:
            s.status = "failed"
            s.error = "reviewer output did not parse as ReviewerSelection"
            print("  ! could not parse reviewer output")
            raw = getattr(review_result, "raw", None)
            if raw is not None:
                print(f"  raw output was:\n{raw}")
            finish()
            return

        selected_papers = select_papers(candidates, selection)
        s.count_out = len(selected_papers)
        s.count_label = "selected"
        dropped = len(selection.selected_papers) - len(selected_papers)
        if dropped:
            s.notes.append(f"{dropped} pick(s) referenced an id not in the candidate set")

    if not selected_papers:
        print("  ! none of the reviewer's picks matched a filtered paper")
        finish()
        return

    print(f"  reviewer selected {len(selected_papers)} papers")

    # 5. Writer
    print("[5/6] writer drafting the digest ...")
    digest = None
    with trace.stage("writer", count_in=len(selected_papers)) as s:
        write_crew = build_write_crew(selected_papers, category_for_prompt, llm=llm)
        write_result = write_crew.kickoff()
        writer_output = _parse_output(write_result, WriterOutput)
        if writer_output is None:
            s.status = "failed"
            s.error = "writer output did not parse as WriterOutput"
            print("  ! could not parse writer output")
            raw = getattr(write_result, "raw", None)
            if raw is not None:
                print(f"  raw output was:\n{raw}")
            finish()
            return

        digest = hydrate_digest_output(writer_output, selected_papers)
        s.count_out = len(digest.entries)
        s.count_label = "articles written"
        missing = len(selected_papers) - len(digest.entries)
        if missing:
            s.notes.append(f"{missing} draft(s) referenced an id Python couldn't match back")

    if not digest.entries:
        print("  ! writer produced no entries")
        finish()
        return

    # 6. Publish
    print(f"[6/6] writing digest ({len(digest.entries)} entries) ...")
    with trace.stage("publish", count_in=len(digest.entries)) as s:
        # Publish to a search-specific subfolder
        search_data_dir = DATA_DIGEST_DIR / "search"
        search_site_dir = SITE_DIGEST_DIR / "search"
        search_data_dir.mkdir(parents=True, exist_ok=True)
        search_site_dir.mkdir(parents=True, exist_ok=True)

        write_digest(slug, digest.entries, search_data_dir, search_site_dir)
        s.count_out = len(digest.entries)
        s.count_label = "entries published"

    finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Sci News digest for a custom query.")
    parser.add_argument(
        "query",
        nargs="+",
        help="Free-form scientific query (e.g. 'solid-state batteries').",
    )
    args = parser.parse_args()
    query = " ".join(args.query).strip()
    if not query:
        print("Please provide a non-empty query.")
        sys.exit(1)

    llm = get_local_llm()
    try:
        run_search(query, llm=llm)
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()