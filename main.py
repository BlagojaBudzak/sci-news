"""
Entry point for the whole pipeline.

Usage:
    python main.py                              # run every configured category
    python main.py --categories chemistry       # run just one
    python main.py --categories chemistry physics

Pipeline shape, per category (the two-phase agent handoff lives in
src/crew_setup.py — see that module's docstring for why it's two crews
and not one):

    fetch candidates -> Reviewer picks 5 -> select_papers() narrows the
    data set -> Writer sees only those 5 -> hydrate_digest_output() attaches metadata -> digest written to disk
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

ModelT = TypeVar("ModelT", bound=BaseModel)


def _parse_output(crew_output, model: Type[ModelT]) -> Optional[ModelT]:
    """Pulls a validated Pydantic object out of a CrewOutput.

    Small local models occasionally wrap valid JSON in a sentence or a
    code fence despite `output_pydantic`, so `.pydantic` can come back
    None even when the raw text is recoverable. This is the same fallback
    the original single-crew version used for the Writer; it's now shared
    across both phases since the Reviewer's ids feed straight into
    `select_papers()` and deserve the same safety net.
    """
    # Check if crew_output.pydantic matches the expected model class
    if isinstance(crew_output.pydantic, model):
        return crew_output.pydantic

    # Fallback: re-parse raw JSON string if pydantic type mismatch or None
    try:
        return model.model_validate(json.loads(crew_output.raw))
    except Exception:  # noqa: BLE001
        return None


def run_for_category(category: str, llm=None) -> None:
    print(f"\n=== {category} ===")
    print("[1/4] fetching papers ...")
    papers = fetch_papers_for_category(category)
    if not papers:
        print(f"  no new {category} papers in the lookback window — skipping")
        return
    print(f"  fetched {len(papers)} candidate papers")

    print("[2/4] reviewer selecting the top 5 ...")
    review_crew = build_review_crew(papers, category, llm=llm)
    review_result = review_crew.kickoff()
    selection = _parse_output(review_result, ReviewerSelection)
    if selection is None:
        print(f"  ! could not parse reviewer output for {category}")
        print(f"  raw output was:\n{review_result.raw}")
        return

    selected_papers = select_papers(papers, selection)
    if not selected_papers:
        print(f"  ! none of the reviewer's picks matched a fetched paper for {category}")
        return
    print(f"  reviewer selected {len(selected_papers)} papers")

    print("[3/4] writer drafting the digest from those papers only ...")
    write_crew = build_write_crew(selected_papers, category, llm=llm)
    write_result = write_crew.kickoff()
    writer_output = _parse_output(write_result, WriterOutput)
    if writer_output is None:
        print(f"  ! could not parse writer output for {category}")
        print(f"  raw output was:\n{write_result.raw}")
        return

    # Post-process: Hydrate LLM output with raw Python metadata (links & reviewer reasons)
    digest = hydrate_digest_output(writer_output, selected_papers)

    print(f"[4/4] writing digest ({len(digest.entries)} entries) ...")
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

    # One shared LLM/model instance for the whole run, across all
    # categories — Ollama keeps the same model resident in VRAM the entire
    # time instead of juggling models between categories.
    llm = get_local_llm()

    for category in args.categories:
        run_for_category(category, llm=llm)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)