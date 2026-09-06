"""
Data ingestion — no RAG, no vector DB, just two free HTTP APIs.

fetch_papers_for_category() returns a flat list of dicts with a common schema:
    {
        "id": str,          # arXiv id or ChemRxiv id — used to re-link
                             # the writer agent's output back to a URL
        "title": str,
        "abstract": str,
        "authors": list[str],
        "published": "YYYY-MM-DD",
        "url": str,          # link to the abstract / landing page
        "doi": str,
        "source": "arXiv" | "ChemRxiv",
    }
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from src.openalex import fetch_openalex_papers
import requests
import arxiv

from config import CATEGORIES, DATA_RAW_DIR, LOOKBACK_DAYS, MAX_PAPERS_PER_SOURCE

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _cutoff_date() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)


def _clean_text(text: str | None) -> str:
    """Collapse internal whitespace and newlines into single spaces."""
    return " ".join((text or "").split())


def fetch_arxiv(categories: List[str], cutoff: datetime) -> List[Dict]:
    """Fetch recent papers from specified arXiv subject categories."""
    if not categories:
        return []

    client = arxiv.Client(page_size=10, delay_seconds=5, num_retries=5)
    query = " OR ".join(f"cat:{c}" for c in categories)
    search = arxiv.Search(
        query=query,
        max_results=MAX_PAPERS_PER_SOURCE,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        published = result.published
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published < cutoff:
            continue

        papers.append({
            "id": result.entry_id.split("/")[-1],
            "title": _clean_text(result.title),
            "abstract": _clean_text(result.summary),
            "authors": [a.name for a in result.authors],
            "published": published.date().isoformat(),
            "url": result.entry_id,
            "doi": result.doi or result.entry_id,
            "source": "arXiv",
        })
    return papers

def fetch_combined_papers(
    category_config: dict,
    category: str,
    lookback_days: int | None = None,
    search_query_override: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Fetch papers from OpenAlex (primary) and arXiv (secondary),
    combine them, and return (combined_list, source_counts).
    """
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category!r}. Add it to config.py.")

    cfg = CATEGORIES[category]
    cutoff = _cutoff_date()

    # 1. OpenAlex primary
    openalex_papers = []
    if cfg.get("openalex"):
        openalex_papers = fetch_openalex_papers(
            category_name=category,
            category_config=cfg,
            lookback_days=LOOKBACK_DAYS,
            search_query_override=search_query_override,
            use_cache=True,
        )

    # 2. arXiv secondary
    arxiv_papers = fetch_arxiv(category_config, lookback_days)

    # 3. Combine
    combined = openalex_papers + arxiv_papers

    # 4. Cache raw combined fetch
    raw_path = DATA_RAW_DIR / f"{category}_{datetime.now().strftime('%Y%m%d')}.json"
    raw_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    source_counts = {
        "openalex": len(openalex_papers),
        "arxiv": len(arxiv_papers),
        "combined": len(combined),
    }

    return combined, source_counts


def fetch_papers_for_category(category: str):
    """Backward-compatible wrapper returning only the combined list."""
    papers, _ = fetch_combined_papers(category)
    return papers

if __name__ == "__main__":
    for cat in CATEGORIES:
        found = fetch_papers_for_category(cat)
        print(f"{cat}: {len(found)} papers fetched")