"""
OpenAlex primary discovery source.

Fetches works from OpenAlex, normalises them into the common internal
paper schema, and provides simple daily caching under data/raw/.
"""

import json
import datetime
from config import DATA_RAW_DIR


import requests

BASE_URL = "https://api.openalex.org/works"
RAW_DIR = DATA_RAW_DIR
USER_AGENT = "SciNews/0.1 (mailto:dev@example.com)"


def _build_params(search_query: str, from_date: str, to_date: str, per_page: int) -> dict:
    """Construct OpenAlex API query parameters."""
    return {
        "search": search_query,
        "filter": (
            f"from_publication_date:{from_date},"
            f"to_publication_date:{to_date},"
            "has_abstract:true"
        ),
        "per-page": per_page,
        "mailto": "dev@example.com",
    }


def reconstruct_abstract(inverted_index) -> str:
    """
    Convert OpenAlex's inverted abstract index into plain text.

    If the input is already a string, return it unchanged.
    If the input is missing / None, return an empty string.
    """
    if not inverted_index:
        return ""
    if isinstance(inverted_index, str):
        return inverted_index

    positions = {}
    for word, indices in inverted_index.items():
        for idx in indices:
            positions[idx] = word

    return " ".join(positions[i] for i in sorted(positions))


def normalize_openalex_work(work: dict) -> dict:
    """
    Convert one OpenAlex work record into the common paper schema.
    """
    openalex_id = work.get("id", "").split("/")[-1]
    doi = work.get("doi") or ""
    title = work.get("title") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])
        if a.get("author")
    ]

    published = work.get("publication_date") or ""
    landing_url = work.get("primary_location", {}).get("landing_page_url") or ""
    if not landing_url and openalex_id:
        landing_url = f"https://openalex.org/{openalex_id}"

    return {
        "id": f"openalex:{openalex_id}" if openalex_id else work.get("id"),
        "source_id": openalex_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "published": published,
        "url": landing_url,
        "doi": doi,
        "source": "openalex",
    }


def fetch_openalex_papers(
    category_name: str,
    category_config: dict,
    lookback_days: int = 7,
    use_cache: bool = True,
) -> list[dict]:
    """
    Fetch papers from OpenAlex for a category.

    Returns a list of normalised paper dicts.
    On any failure, returns an empty list so the pipeline can fall back
    to secondary sources.
    """
    openalex_cfg = category_config.get("openalex", {})
    search_query = openalex_cfg.get("search_query", "").strip()
    if not search_query:
        return []

    per_page = openalex_cfg.get("per_page", 50)
    max_results = openalex_cfg.get("max_results", 30)

    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=lookback_days)

    cache_file = RAW_DIR / f"openalex_{category_name}_{end_date.isoformat()}.json"

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
            return [normalize_openalex_work(w) for w in cached[:max_results]]
        except Exception:
            # Fall through to live API if cache is corrupt
            pass

    params = _build_params(search_query, start_date.isoformat(), end_date.isoformat(), per_page)
    headers = {"User-Agent": USER_AGENT}

    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return [normalize_openalex_work(w) for w in results[:max_results]]

    except Exception:
        # OpenAlex failed – return empty list, secondary fetch will take over
        return []
