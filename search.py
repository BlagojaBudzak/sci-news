"""
Command-line interface for Sci News dynamic search.

Usage:
    python search.py "solid-state batteries"

This is a thin wrapper around the research engine (src.engine.research).
"""
from __future__ import annotations

import argparse
import sys

from src.engine import ResearchResult, research
from src.crew_setup import get_local_llm


def run_search(query: str, llm=None) -> ResearchResult:
    """
    Run the research pipeline and print results.
    Kept for backward compatibility; delegates to engine.research.
    """
    return research(query, llm=llm, verbose=True, save_outputs=True)


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