"""
Reusable research engine for Sci News.

This module contains the core pipeline that can be called programmatically
without any CLI or filesystem coupling. It returns a structured result
object containing all relevant information.

Example usage:
    from src.engine import research
    result = research("hydrogen storage")
    print(result.verification_status)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from pydantic import BaseModel

from config import (
    DATA_DIGEST_DIR,
    DATA_TRACE_DIR,
    SITE_DIGEST_DIR,
    FACT_CHECKER_MAX_REVISION_ROUNDS,
)
from src.arxiv import fetch_arxiv_search
from src.crew_setup import (
    BlogEntry,
    ReviewerSelection,
    WriterOutput,
    build_review_crew,
    build_write_crew,
    get_local_llm,
    hydrate_digest_output,
    select_papers,
)
from src.digest_writer import write_digest
from src.fact_checker import build_revision_feedback, run_fact_check
from src.openalex import fetch_openalex_papers
from src.prefilter import PrefilterConfig, deduplicate_papers, filter_papers
from src.query_interpreter import interpret_query, build_arxiv_query, build_openalex_query
from src.trace import PipelineTrace, StageResult


class ResearchResult(BaseModel):
    """Structured result returned by the research engine."""
    job_id: str
    query: str
    status: str
    digest_entries: Optional[List[BlogEntry]] = None
    verification_status: Optional[str] = None
    trace_path: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


def _parse_output(crew_output, model_type):
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
    slug = "".join(
        c.lower() if c.isalnum() or c == "-" else "-"
        for c in query
    ).strip("-")
    return slug[:80] or "search"


def _search_prefilter_config(plan) -> PrefilterConfig:
    return PrefilterConfig(
        positive_keywords=plan.positive_keywords,
        negative_keywords=plan.negative_keywords,
        min_abstract_chars=80,
        min_relevance_score=0.0,
        positive_match_required=False,
        relevance_weight=1.0,
        recency_weight=0.5,
        recency_half_life_days=10.0,
        max_candidates=15,
    )


def research(
    query: str,
    llm=None,
    verbose: bool = True,
    save_outputs: bool = True,
    job_id: str | None = None,
) -> ResearchResult:
    trace = PipelineTrace.start(f"search:{query}", job_id=job_id)
    category_for_prompt = f"search results for '{query}'"
    slug = _slug_from_query(query)

    # Helper to save trace only once, respecting save_outputs
    trace_saved = False
    def _save_trace() -> Optional[str]:
        nonlocal trace_saved
        if not trace_saved and save_outputs:
            trace_saved = True
            return str(trace.save(DATA_TRACE_DIR))
        return None

    try:
        if verbose:
            print(f"\n=== Dynamic search: {query} ===  (job #{trace.job_id})")

        # 1. Interpret query
        if verbose:
            print("[1/7] interpreting query ...")
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
        if verbose:
            print(f"  search phrases: {plan.search_phrases}")
            print(f"  OpenAlex query: {openalex_query}")
            print(f"  arXiv query: {arxiv_query}")
            print(f"  lookback days: {plan.lookback_days}")

        # 2. Fetch papers
        if verbose:
            print("[2/7] fetching papers ...")
        with trace.stage("fetch") as s:
            openalex_papers = fetch_openalex_papers(
                "search",
                {"openalex": {"per_page": 50, "max_results": 30}},
                lookback_days=plan.lookback_days,
                use_cache=False,
                search_query_override=openalex_query,
            )
            arxiv_papers = []
            try:
                arxiv_papers = fetch_arxiv_search(arxiv_query, plan.lookback_days)
            except Exception as e:
                s.notes.append(f"arXiv retrieval failed: {e}")
                if verbose:
                    print(f"  ! arXiv retrieval error: {e}")

            papers = openalex_papers + arxiv_papers
            s.count_out = len(papers)
            s.count_label = "papers found"
            s.notes.append(f"OpenAlex: {len(openalex_papers)}, arXiv: {len(arxiv_papers)}")

        if not papers:
            if verbose:
                print("  no papers found in the lookback window — skipping")
            trace.stages[-1].notes.append("lookback window empty")
            trace_path = _save_trace()
            return ResearchResult(
                job_id=trace.job_id,
                query=query,
                status="no_papers",
                trace_path=trace_path,
            )

        if verbose:
            print(f"  sources: OpenAlex={len(openalex_papers)}, arXiv={len(arxiv_papers)}")

        # 3. Deduplicate and prefilter
        if verbose:
            print("[3/7] deterministic pre-filter ...")
        with trace.stage("prefilter", count_in=len(papers)) as s:
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

        if verbose:
            print(
                f"  pre-filter: {prefilter.input_count} -> {prefilter.output_count} papers "
                f"(duplicates={prefilter.duplicate_count}, "
                f"short abstracts={prefilter.missing_abstract_count}, "
                f"low relevance={prefilter.low_relevance_count})"
            )
        candidates = prefilter.candidates
        if not candidates:
            if verbose:
                print("  ! pre-filter removed all candidates — skipping")
            trace_path = _save_trace()
            return ResearchResult(
                job_id=trace.job_id,
                query=query,
                status="no_candidates",
                trace_path=trace_path,
            )

        # 4. Reviewer
        if verbose:
            print("[4/7] reviewer selecting the top 5 ...")
        selected_papers = []
        with trace.stage("reviewer", count_in=len(candidates)) as s:
            review_crew = build_review_crew(candidates, category_for_prompt, llm=llm)
            review_result = review_crew.kickoff()
            selection = _parse_output(review_result, ReviewerSelection)
            if selection is None:
                s.status = "failed"
                s.error = "reviewer output did not parse as ReviewerSelection"
                if verbose:
                    print("  ! could not parse reviewer output")
                    raw = getattr(review_result, "raw", None)
                    if raw is not None:
                        print(f"  raw output was:\n{raw}")
                trace_path = _save_trace()
                return ResearchResult(
                    job_id=trace.job_id,
                    query=query,
                    status="failed",
                    error=s.error,
                    trace_path=trace_path,
                )

            selected_papers = select_papers(candidates, selection)
            s.count_out = len(selected_papers)
            s.count_label = "selected"
            dropped = len(selection.selected_papers) - len(selected_papers)
            if dropped:
                s.notes.append(f"{dropped} pick(s) referenced an id not in the candidate set")

        if not selected_papers:
            if verbose:
                print("  ! none of the reviewer's picks matched a filtered paper")
            trace_path = _save_trace()
            return ResearchResult(
                job_id=trace.job_id,
                query=query,
                status="no_entries",
                trace_path=trace_path,
            )

        if verbose:
            print(f"  reviewer selected {len(selected_papers)} papers")

        # 5. Writer + Fact Checker (with possible revision loop)
        if verbose:
            print("[5/7] writer drafting the digest ...")
        max_revisions = FACT_CHECKER_MAX_REVISION_ROUNDS
        writer_output = None
        fact_check_report = None
        final_verification_status = "UNVERIFIED"

        writer_attempt = 0
        fact_check_attempt = 0

        with trace.stage("writer", count_in=len(selected_papers)) as writer_stage:
            feedback = None
            for revision_round in range(max_revisions + 1):
                writer_attempt += 1
                if verbose:
                    print(f"  writer attempt {writer_attempt}/{max_revisions + 1}")
                write_crew = build_write_crew(selected_papers, category_for_prompt, llm=llm, feedback=feedback)
                write_result = write_crew.kickoff()
                writer_output = _parse_output(write_result, WriterOutput)
                if writer_output is None:
                    writer_stage.status = "failed"
                    writer_stage.error = "writer output did not parse as WriterOutput"
                    if verbose:
                        print("  ! could not parse writer output")
                    trace_path = _save_trace()
                    return ResearchResult(
                        job_id=trace.job_id,
                        query=query,
                        status="failed",
                        error=writer_stage.error,
                        trace_path=trace_path,
                    )

                fact_check_attempt += 1
                if verbose:
                    print(f"  fact-check attempt {fact_check_attempt}/{max_revisions + 1}")
                fact_check_report = run_fact_check(writer_output, selected_papers, llm)
                final_verification_status = fact_check_report.overall_status
                if verbose:
                    print(f"    fact-check overall status: {fact_check_report.overall_status}")

                if fact_check_report.overall_status == "PASS" or revision_round == max_revisions:
                    break

                feedback = build_revision_feedback(fact_check_report)
                if not feedback:
                    break

                if verbose:
                    print(f"    revision needed; generating feedback...")
                writer_stage.notes.append(f"revision {revision_round + 1} needed: {fact_check_report.overall_status}")

            writer_stage.count_out = len(writer_output.articles)
            writer_stage.count_label = "articles written"
            writer_stage.notes.append(f"writer attempts: {writer_attempt}")
            if len(writer_output.articles) < len(selected_papers):
                missing = len(selected_papers) - len(writer_output.articles)
                writer_stage.notes.append(f"{missing} draft(s) referenced an id Python couldn't match back")

        # 6. Fact Checker trace stage
        with trace.stage("fact_checker", count_in=len(fact_check_report.assessments)) as fc_stage:
            fc_stage.count_out = len(fact_check_report.assessments)
            fc_stage.count_label = "claims assessed"
            fc_stage.notes.append(f"attempts: {fact_check_attempt}")
            fc_stage.notes.append(f"overall status: {fact_check_report.overall_status}")
            verdict_counts = {}
            for a in fact_check_report.assessments:
                verdict_counts[a.verdict] = verdict_counts.get(a.verdict, 0) + 1
            for v, c in verdict_counts.items():
                fc_stage.notes.append(f"{c} claim(s) with verdict {v}")

        # Publication gate: reject if FAIL
        if final_verification_status == "FAIL":
            if verbose:
                print("  ! Fact-check FAILED — digest will not be published.")
            with trace.stage("publish", count_in=len(writer_output.articles)) as s:
                s.status = "rejected"
                s.error = "Fact-check overall status is FAIL. Digest not published."
                s.count_out = 0
                s.count_label = "entries published"
                s.notes.append(f"verification status: {final_verification_status}")
            trace_path = _save_trace()
            return ResearchResult(
                job_id=trace.job_id,
                query=query,
                status="fact_check_failed",
                error="Fact-check overall status is FAIL. No digest published.",
                trace_path=trace_path,
            )

        # 7. Publish
        if verbose:
            print(f"[7/7] writing digest ({len(writer_output.articles)} entries) ...")
        with trace.stage("publish", count_in=len(writer_output.articles)) as s:
            digest = hydrate_digest_output(
                writer_output,
                selected_papers,
                verification_status=final_verification_status,
            )
            if not digest.entries:
                if verbose:
                    print("  ! writer produced no entries after fact-check")
                trace_path = _save_trace()
                return ResearchResult(
                    job_id=trace.job_id,
                    query=query,
                    status="no_entries",
                    trace_path=trace_path,
                )

            if save_outputs:
                search_data_dir = DATA_DIGEST_DIR / "search"
                search_site_dir = SITE_DIGEST_DIR / "search"
                search_data_dir.mkdir(parents=True, exist_ok=True)
                search_site_dir.mkdir(parents=True, exist_ok=True)
                write_digest(slug, digest.entries, search_data_dir, search_site_dir)
                s.count_out = len(digest.entries)
                s.count_label = "entries published"
                s.notes.append(f"verification status: {final_verification_status}")

        trace_path = _save_trace()
        if verbose:
            print(f"  trace written to {trace_path}")

        return ResearchResult(
            job_id=trace.job_id,
            query=query,
            status="completed",
            digest_entries=digest.entries,
            verification_status=final_verification_status,
            trace_path=trace_path,
            summary={
                "source_counts": {"openalex": len(openalex_papers), "arxiv": len(arxiv_papers)},
                "candidates_after_prefilter": prefilter.output_count,
                "selected_papers": len(selected_papers),
                "fact_check_attempts": fact_check_attempt,
                "writer_attempts": writer_attempt,
            },
        )

    except Exception as e:
        # Record pipeline failure in trace, then save and re-raise
        failure_stage = StageResult(name="error", status="failed", error=str(e))
        trace.stages.append(failure_stage)
        # Save trace unconditionally on crash (for audit/debugging)
        trace_path = str(trace.save(DATA_TRACE_DIR))
        if verbose:
            print(f"  ! pipeline crashed; trace saved to {trace_path}")
        raise