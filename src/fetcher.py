"""
Data ingestion — no RAG, no vector DB, just two free HTTP APIs.

fetch_papers_for_category() returns a flat list of dicts with a common
schema regardless of source:

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

import arxiv

try:
    import chemrxiv
    CHEMRXIV_AVAILABLE = True
except ImportError:
    # `pip install chemrxiv` is optional — categories with no chemrxiv_terms
    # configured will still work fine without it.
    CHEMRXIV_AVAILABLE = False

from config import CATEGORIES, DATA_RAW_DIR, LOOKBACK_DAYS, MAX_PAPERS_PER_SOURCE


def _cutoff_date() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)


def fetch_arxiv(categories: List[str], cutoff: datetime) -> List[Dict]:
    """Fetch recent papers from one or more arXiv subject classes."""
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
            # Results are sorted newest-first, so once we're past the
            # lookback window everything after is even older — but we keep
            # iterating rather than breaking, since arXiv occasionally
            # returns out-of-order pages.
            continue

        papers.append({
            "id": result.entry_id.split("/")[-1],
            "title": result.title.strip().replace("\n", " "),
            "abstract": result.summary.strip().replace("\n", " "),
            "authors": [a.name for a in result.authors],
            "published": published.date().isoformat(),
            "url": result.entry_id,
            "doi": result.doi or result.entry_id,
            "source": "arXiv",
        })
    return papers


def fetch_chemrxiv(terms: List[str], cutoff: datetime) -> List[Dict]:
    """Fetch recent papers from ChemRxiv (Cambridge Open Engage API)."""
    if not terms or not CHEMRXIV_AVAILABLE:
        return []

    client = chemrxiv.Client()
    date_from = cutoff.strftime("%Y-%m-%dT00:00:00.000Z")

    papers, seen_ids = [], set()
    for term in terms:
        search = chemrxiv.Search(
            term=term,
            limit=MAX_PAPERS_PER_SOURCE,
            sort=chemrxiv.SortCriterion.PUBLISHED_DATE_DESC,
            search_date_from=date_from,
        )
        try:
            for result in client.results(search):
                paper_id = getattr(result, "id", None) or result.doi
                if not paper_id or paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)

                papers.append({
                    "id": paper_id,
                    "title": result.title.strip(),
                    "abstract": (result.abstract or "").strip(),
                    "authors": [str(a) for a in result.authors],
                    "published": str(getattr(result, "published_date", ""))[:10],
                    "url": f"https://doi.org/{result.doi}" if result.doi else "",
                    "doi": result.doi or "",
                    "source": "ChemRxiv",
                })
        except Exception as e:
            print(f"  ! ChemRxiv fetch failed ({e}). Falling back to arXiv only.")
    return papers


def fetch_papers_for_category(category: str) -> List[Dict]:
    """Fetch + cache the raw metadata for one configured category."""
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category!r}. Add it to config.py.")

    cfg = CATEGORIES[category]
    cutoff = _cutoff_date()

    papers = []
    papers += fetch_arxiv(cfg.get("arxiv_categories", []), cutoff)
    papers += fetch_chemrxiv(cfg.get("chemrxiv_terms", []), cutoff)

    # Cache the raw fetch so you can re-run the agent step (e.g. after
    # tweaking a prompt) without re-hitting the APIs.
    raw_path = DATA_RAW_DIR / f"{category}_{datetime.now().strftime('%Y%m%d')}.json"
    raw_path.write_text(json.dumps(papers, indent=2), encoding="utf-8")

    return papers


if __name__ == "__main__":
    # Quick manual check: python -m src.fetcher
    for cat in CATEGORIES:
        found = fetch_papers_for_category(cat)
        print(f"{cat}: {len(found)} papers fetched")
