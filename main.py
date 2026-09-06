"""
Entry point for the whole pipeline.

Usage:
    python main.py                              # run every configured category
    python main.py --categories chemistry       # run just one
    python main.py --categories chemistry physics

Pipeline shape, per category:

    fetch -> deterministic pre-filter -> Reviewer picks 5 -> select_papers()
    -> Writer sees only those 5 -> hydrate_digest_output() -> digest
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from config import CATEGORIES, DATA_DIGEST_DIR, SITE_DIGEST_DIR
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
    """Build the deterministic pre-filter configuration from config.py."""
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
    print(f"\n=== {category} ===")
    print("[1/5] fetching papers ...")
    papers = fetch_papers_for_category(category)
    if not papers:
        print(f"  no new {category} papers in the lookback window — skipping")
        return
    print(f"  fetched {len(papers)} candidate papers")

    print("[2/5] deterministic pre-filter ...")
    prefilter = filter_papers(papers, _prefilter_config(category))
    print(
        f"  pre-filter: {prefilter.input_count} -> {prefilter.output_count} papers "
        f"(duplicates={prefilter.duplicate_count}, "
        f"short abstracts={prefilter.missing_abstract_count}, "
        f"low relevance={prefilter.low_relevance_count})"
    )
    for example in prefilter.filtered_examples[:5]:
        print(f"  filtered: {example['title']!r} — {example['reason']}")

    if not prefilter.candidates:
        print(f"  ! pre-filter removed every {category} candidate — skipping reviewer")
        return

    print("[3/5] reviewer selecting the top 5 ...")
    review_crew = build_review_crew(prefilter.candidates, category, llm=llm)
    review_result = review_crew.kickoff()
    selection = _parse_output(review_result, ReviewerSelection)
    if selection is None:
        print(f"  ! could not parse reviewer output for {category}")
        print(f"  raw output was:\n{review_result.raw}")
        return

    selected_papers = select_papers(prefilter.candidates, selection)
    if not selected_papers:
        print(f"  ! none of the reviewer's picks matched a fetched paper for {category}")
        return
    print(f"  reviewer selected {len(selected_papers)} papers")

    print("[4/5] writer drafting the digest from those papers only ...")
    write_crew = build_write_crew(selected_papers, category, llm=llm)
    write_result = write_crew.kickoff()
    writer_output = _parse_output(write_result, WriterOutput)
    if writer_output is None:
        print(f"  ! could not parse writer output for {category}")
        print(f"  raw output was:\n{write_result.raw}")
        return

    digest = hydrate_digest_output(writer_output, selected_papers)

    print(f"[5/5] writing digest ({len(digest.entries)} entries) ...")
    write_digest(category, digest.entries, DATA_DIGEST_DIR, SITE_DIGEST_DIR)


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
