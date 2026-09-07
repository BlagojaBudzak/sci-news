"""
arXiv retrieval — free-form query search for recent papers.

This module isolates the arXiv-specific retrieval logic so the research
engine does not need to know implementation details.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Dict
import arxiv

from config import MAX_PAPERS_PER_SOURCE


def fetch_arxiv_search(query: str, lookback_days: int) -> List[Dict]:
    """Fetch recent arXiv papers matching a free-form query string.

    Args:
        query: Free-form query string (already formatted for arXiv).
        lookback_days: Only return papers published within this many days.

    Returns:
        List of paper dicts in the common schema.

    Raises:
        Any exception from the arxiv library (network, parse, etc.).
        Callers should decide how to handle it.
    """
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
    return papers