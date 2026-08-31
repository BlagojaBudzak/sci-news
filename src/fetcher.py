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


def fetch_chemrxiv(
    terms: List[str], cutoff: datetime, max_papers: int = MAX_PAPERS_PER_SOURCE
) -> List[Dict]:
    """Fetch recent preprints from ChemRxiv REST API using custom browser headers."""
    if not terms:
        return []

    date_from = cutoff.strftime("%Y-%m-%dT00:00:00.000Z")
    endpoint = "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"
    papers, seen_ids = [], set()

    session = requests.Session()
    session.headers.update(HEADERS)

    for term in terms:
        params = {
            "term": term,
            "limit": max_papers,
            "sort": "PUBLISHED_DATE_DESC",
            "searchDateFrom": date_from,
        }
        try:
            res = session.get(endpoint, params=params, timeout=10)
            res.raise_for_status()
            data = res.json()

            for hit in data.get("itemHits", []):
                item = hit.get("item", {})
                paper_id = item.get("id") or item.get("doi")
                if not paper_id or paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                authors = [
                    f"{a.get('firstName', '')} {a.get('lastName', '')}".strip()
                    for a in item.get("authors", [])
                ]

                papers.append({
                    "id": paper_id,
                    "title": _clean_text(item.get("title")),
                    "abstract": _clean_text(item.get("abstract")),
                    "authors": authors,
                    "published": str(item.get("publishedDate", ""))[:10],
                    "url": (
                        f"https://doi.org/{item.get('doi')}"
                        if item.get("doi")
                        else f"https://chemrxiv.org/engage/chemrxiv/article-details/{paper_id}"
                    ),
                    "doi": item.get("doi") or "",
                    "source": "ChemRxiv",
                })
        except Exception as e:
            print(
                f"  ! ChemRxiv fetch failed for '{term}': {e}. Falling back to arXiv only."
            )

    return papers


def fetch_papers_for_category(category: str) -> List[Dict]:
    """Fetch + cache raw metadata for a single category."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category!r}. Add it to config.py.")

    cfg = CATEGORIES[category]
    cutoff = _cutoff_date()

    papers = []
    papers += fetch_arxiv(cfg.get("arxiv_categories", []), cutoff)
    papers += fetch_chemrxiv(cfg.get("chemrxiv_terms", []), cutoff)

    # Cache raw fetch to re-run agent loops without re-hitting external APIs
    raw_path = DATA_RAW_DIR / f"{category}_{datetime.now().strftime('%Y%m%d')}.json"
    raw_path.write_text(json.dumps(papers, indent=2), encoding="utf-8")

    return papers


if __name__ == "__main__":
    for cat in CATEGORIES:
        found = fetch_papers_for_category(cat)
        print(f"{cat}: {len(found)} papers fetched")