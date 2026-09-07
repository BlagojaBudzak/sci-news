"""
Entry point for the whole pipeline.

Usage:
    python main.py                              # run every configured category
    python main.py --categories chemistry       # run just one
    python main.py --categories chemistry physics

Pipeline shape, per category:

    fetch -> deterministic pre-filter -> Reviewer picks 5 -> select_papers()
    -> Writer sees only those 5 -> Fact Checker verifies claims -> hydrate_digest_output()
    -> digest
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from config import CATEGORIES, DATA_DIGEST_DIR, DATA_TRACE_DIR, SITE_DIGEST_DIR, FACT_CHECKER_MAX_REVISION_ROUNDS
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
from src.fact_checker import FactCheckReport, build_revision_feedback, run_fact_check
from src.fetcher import fetch_combined_papers
from src.prefilter import PrefilterConfig, filter_papers
from src.trace import PipelineTrace

ModelT = TypeVar("ModelT", bound=BaseModel)


def _parse_output(crew_output: Any, model: Type[ModelT]) -> Optional[ModelT]:
    """Pull a validated Pydantic object out of a CrewOutput."""
    parsed = getattr(crew_output, "pydantic", None)
    if isinstance(parsed, model):
        return parsed

    raw = getattr(crew_output, "raw", None)
    if not isinstance(raw, str):
        return None

    try:
        return model.model_validate(json.loads(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _prefilter_config(category: str) -> PrefilterConfig:
    """Build deterministic pre-filter configuration from config.py."""
    cfg = CATEGORIES[category].get("prefilter", {})
    return PrefilterConfig(
        positive_keywords=cfg.get("positive_keywords", {}),
        negative_keywords=cfg.get("negative_keywords", {}),
        min_abstract_chars=cfg.get("min_abstract_chars", 80),
        min_relevance_score=cfg.get("min_relevance_score", 0.0),
        positive_match_required=cfg.get("positive_match_required", True),
        relevance_weight=cfg.get("relevance_weight", 1.0),
        recency_weight=cfg.get("recency_weight", 1.0),
        recency_half_life_days=cfg.get("recency_half_life_days", 14.0),
    )


def run_for_category(category: str, llm=None) -> None:
    trace = PipelineTrace.start(category)

    def finish() -> None:
        trace.print_summary()
        path = trace.save(DATA_TRACE_DIR)
        print(f"  trace written to {path}")

    print(f"\n=== {category} ===  (job #{trace.job_id})")

    print("[1/5] fetching papers ...")
    with trace.stage("fetch") as s:
        papers, source_counts = fetch_combined_papers(category)
        s.count_out = len(papers)
        s.count_label = "papers found"
        s.notes.append(
            f"OpenAlex: {source_counts['openalex']}, "
            f"arXiv: {source_counts['arxiv']}, "
            f"combined: {source_counts['combined']}"
        )

    if not papers:
        print(f"  no new {category} papers in the lookback window — skipping")
        trace.stages[-1].notes.append("lookback window empty")
        finish()
        return

    print(f"  sources: OpenAlex={source_counts['openalex']}, arXiv={source_counts['arxiv']}")

    # [2/5] Deterministic pre‑filter
    print("[2/5] deterministic pre-filter ...")

    prefilter_config = CATEGORIES[category].get("prefilter")

    if prefilter_config is not None:
        with trace.stage("prefilter", count_in=len(papers)) as s:
            prefilter = filter_papers(papers, _prefilter_config(category))
            s.count_out = prefilter.output_count
            s.count_label = "candidates retained"
            s.notes.append(f"duplicates removed: {prefilter.duplicate_count}")
            s.notes.append(f"short/missing abstracts: {prefilter.missing_abstract_count}")
            s.notes.append(f"low relevance: {prefilter.low_relevance_count}")
            for example in prefilter.filtered_examples[:10]:
                s.notes.append(f"filtered: {example['title']!r} — {example['reason']}")

        print(
            f"  pre-filter: {prefilter.input_count} -> {prefilter.output_count} papers "
            f"(duplicates={prefilter.duplicate_count}, "
            f"short abstracts={prefilter.missing_abstract_count}, "
            f"low relevance={prefilter.low_relevance_count})"
        )
        for example in prefilter.filtered_examples[:5]:
            print(f"  filtered: {example['title']!r} — {example['reason']}")

        candidates = prefilter.candidates
        if not candidates:
            print(f"  ! pre-filter removed every {category} candidate — skipping reviewer")
            finish()
            return
    else:
        with trace.stage("prefilter", count_in=len(papers)) as s:
            s.count_out = len(papers)
            s.count_label = "candidates retained (no prefilter config)"
            s.notes.append("No deterministic prefilter configuration; all papers passed through")
        print(f"  pre-filter: no configuration; passing all {len(papers)} papers through")
        candidates = papers

    # [3/5] Reviewer
    print("[3/5] reviewer selecting the top 5 ...")
    selected_papers = []
    with trace.stage("reviewer", count_in=len(candidates)) as s:
        review_crew = build_review_crew(candidates, category, llm=llm)
        review_result = review_crew.kickoff()
        selection = _parse_output(review_result, ReviewerSelection)
        if selection is None:
            s.status = "failed"
            s.error = "reviewer output did not parse as ReviewerSelection"
            print(f"  ! could not parse reviewer output for {category}")
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
        print(f"  ! none of the reviewer's picks matched a filtered paper for {category}")
        finish()
        return

    print(f"  reviewer selected {len(selected_papers)} papers")

    # [4/5] Writer + Fact Checker (with possible revision loop)
    print("[4/5] writer drafting the digest from those papers only ...")
    max_revisions = FACT_CHECKER_MAX_REVISION_ROUNDS
    writer_output = None
    fact_check_report = None
    final_verification_status = "UNVERIFIED"

    # We'll record writer attempts and fact-check attempts in separate trace stages.
    writer_attempt = 0
    fact_check_attempt = 0

    with trace.stage("writer", count_in=len(selected_papers)) as writer_stage:
        feedback = None
        for revision_round in range(max_revisions + 1):
            writer_attempt += 1
            print(f"  writer attempt {writer_attempt}/{max_revisions + 1}")
            write_crew = build_write_crew(selected_papers, category, llm=llm, feedback=feedback)
            write_result = write_crew.kickoff()
            writer_output = _parse_output(write_result, WriterOutput)
            if writer_output is None:
                writer_stage.status = "failed"
                writer_stage.error = "writer output did not parse as WriterOutput"
                print(f"  ! could not parse writer output for {category}")
                raw = getattr(write_result, "raw", None)
                if raw is not None:
                    print(f"  raw output was:\n{raw}")
                finish()
                return

            # Run Fact Checker
            fact_check_attempt += 1
            print(f"  fact-check attempt {fact_check_attempt}/{max_revisions + 1}")
            fact_check_report = run_fact_check(writer_output, selected_papers, llm)
            print(f"    fact-check overall status: {fact_check_report.overall_status}")
            final_verification_status = fact_check_report.overall_status

            if fact_check_report.overall_status == "PASS" or revision_round == max_revisions:
                break

            feedback = build_revision_feedback(fact_check_report)
            if not feedback:
                # Should not happen, but avoid infinite loop.
                break

            print(f"    revision needed; generating feedback...")
            writer_stage.notes.append(f"revision {revision_round + 1} needed: {fact_check_report.overall_status}")

        writer_stage.count_out = len(writer_output.articles)
        writer_stage.count_label = "articles written"
        writer_stage.notes.append(f"writer attempts: {writer_attempt}")
        if len(writer_output.articles) < len(selected_papers):
            missing = len(selected_papers) - len(writer_output.articles)
            writer_stage.notes.append(f"{missing} draft(s) referenced an id Python couldn't match back")

    # Record Fact Checker stage(s)
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

    # [5/5] Publish with verification status
    print(f"[5/5] writing digest ({len(writer_output.articles)} entries) ...")
    with trace.stage("publish", count_in=len(writer_output.articles)) as s:
        digest = hydrate_digest_output(
            writer_output,
            selected_papers,
            verification_status=final_verification_status,
        )
        write_digest(category, digest.entries, DATA_DIGEST_DIR, SITE_DIGEST_DIR)
        s.count_out = len(digest.entries)
        s.count_label = "entries published"
        s.notes.append(f"verification status: {final_verification_status}")

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