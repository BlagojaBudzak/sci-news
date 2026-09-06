"""
Entry point for the whole pipeline.

Usage:
    python main.py                              # run every configured category
    python main.py --categories chemistry       # run just one
    python main.py --categories chemistry physics

Pipeline shape, per category:

    fetch candidates -> deterministic pre-filter -> Reviewer picks 5
    -> select_papers() narrows the data set -> Writer sees only those 5
    -> hydrate_digest_output() attaches metadata -> digest written to disk
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from config import CATEGORIES, DATA_DIGEST_DIR, DATA_TRACE_DIR, SITE_DIGEST_DIR
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
from src.fetcher import fetch_papers_for_category
from src.prefilter import PrefilterConfig, filter_papers
from src.trace import PipelineTrace

ModelT = TypeVar("ModelT", bound=BaseModel)


def _parse_output(crew_output, model: Type[ModelT]) -> Optional[ModelT]:
    """Pull a validated Pydantic object out of a CrewOutput."""
    if isinstance(crew_output.pydantic, model):
        return crew_output.pydantic
    try:
        return model.model_validate(json.loads(crew_output.raw))
    except Exception:  # noqa: BLE001
        return None


def _prefilter_config(category: str) -> PrefilterConfig:
    """Convert category config into the typed deterministic filter config."""
    cfg = CATEGORIES[category].get("prefilter", {})
    return PrefilterConfig(**cfg)


def run_for_category(category: str, llm=None) -> None:
    trace = PipelineTrace.start(category)

    def finish() -> None:
        trace.print_summary()
        path = trace.save(DATA_TRACE_DIR)
        print(f"  trace written to {path}")

    print(f"\n=== {category} ===  (job #{trace.job_id})")

    print("[1/5] fetching papers ...")
    with trace.stage("fetch") as s:
        papers = fetch_papers_for_category(category)
        s.count_out = len(papers)
        s.count_label = "papers found"

    if not papers:
        print(f"  no new {category} papers in the lookback window — skipping")
        trace.stages[-1].notes.append("lookback window empty")
        finish()
        return
    print(f"  fetched {len(papers)} candidate papers")

    print("[2/5] deterministic pre-filter ...")
    with trace.stage("prefilter", count_in=len(papers)) as s:
        result = filter_papers(papers, _prefilter_config(category))
        s.count_out = result.output_count
        s.count_label = "candidates retained"
        s.notes.append(f"duplicates removed: {result.duplicate_count}")
        s.notes.append(f"short/missing abstracts: {result.missing_abstract_count}")
        s.notes.append(f"low relevance: {result.low_relevance_count}")
        for example in result.filtered_examples:
            s.notes.append(
                f"filtered '{example['title']}' — {example['reason']}"
            )

    if not result.candidates:
        print(f"  ! pre-filter rejected all {len(papers)} papers for {category}")
        finish()
        return
    print(
        f"  pre-filter: {result.input_count} -> {result.output_count} papers "
        f"(duplicates={result.duplicate_count}, short abstracts={result.missing_abstract_count}, "
        f"low relevance={result.low_relevance_count})"
    )
    for example in result.filtered_examples:
        print(f"  filtered: '{example['title']}' — {example['reason']}")

    filtered_papers = result.candidates

    print("[3/5] reviewer selecting the top 5 ...")
    selected_papers = []
    with trace.stage("reviewer", count_in=len(filtered_papers)) as s:
        review_crew = build_review_crew(filtered_papers, category, llm=llm)
        review_result = review_crew.kickoff()
        selection = _parse_output(review_result, ReviewerSelection)
        if selection is None:
            s.status = "failed"
            s.error = "reviewer output did not parse as ReviewerSelection"
            print(f"  ! could not parse reviewer output for {category}")
            print(f"  raw output was:\n{review_result.raw}")
            finish()
            return

        selected_papers = select_papers(filtered_papers, selection)
        s.count_out = len(selected_papers)
        s.count_label = "selected"
        dropped = len(selection.selected_papers) - len(selected_papers)
        if dropped:
            s.notes.append(f"{dropped} pick(s) referenced an id not in the candidate set")

    if not selected_papers:
        print(f"  ! none of the reviewer's picks matched a fetched paper for {category}")
        finish()
        return
    print(f"  reviewer selected {len(selected_papers)} papers")

    print("[4/5] writer drafting the digest from those papers only ...")
    digest = None
    with trace.stage("writer", count_in=len(selected_papers)) as s:
        write_crew = build_write_crew(selected_papers, category, llm=llm)
        write_result = write_crew.kickoff()
        writer_output = _parse_output(write_result, WriterOutput)
        if writer_output is None:
            s.status = "failed"
            s.error = "writer output did not parse as WriterOutput"
            print(f"  ! could not parse writer output for {category}")
            print(f"  raw output was:\n{write_result.raw}")
            finish()
            return

        digest = hydrate_digest_output(writer_output, selected_papers)
        s.count_out = len(digest.entries)
        s.count_label = "articles written"
        missing = len(selected_papers) - len(digest.entries)
        if missing:
            s.notes.append(f"{missing} draft(s) referenced an id Python couldn't match back")

    print(f"[5/5] writing digest ({len(digest.entries)} entries) ...")
    with trace.stage("publish", count_in=len(digest.entries)) as s:
        write_digest(category, digest.entries, DATA_DIGEST_DIR, SITE_DIGEST_DIR)
        s.count_out = len(digest.entries)
        s.count_label = "entries published"

    finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Sci News digest.")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=list(CATEGORIES.keys()),
        choices=list(CATEGORIES.keys()),
        help="Which categories to run (default: all configured categories).",
    )
    args = parser.parse_args()

    llm = get_local_llm()
    for category in args.categories:
        run_for_category(category, llm=llm)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
