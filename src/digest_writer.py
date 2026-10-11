"""
Output generation for a query-specific research digest.

Writes:
  * data/digests/search/<query-slug>_<date>.md — a Markdown archive
  * site/digests/search/<query-slug>.json — the frontend's static fast path
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from src.crew_setup import BlogEntry


def write_digest(
    query_slug: str,
    entries: List[BlogEntry],
    data_digest_dir: Path,
    site_digest_dir: Path,
) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    data_digest_dir.mkdir(parents=True, exist_ok=True)
    site_digest_dir.mkdir(parents=True, exist_ok=True)

    # Markdown archive
    md_lines = [f"# Sci News — {query_slug.replace('-', ' ').title()} — {date_str}\n"]
    for entry in entries:
        md_lines.append(f"## {entry.title}\n")
        md_lines.append(f"{entry.paragraph}\n")
        md_lines.append(f"**Read the paper:** [{entry.link}]({entry.link})\n")
    md_path = data_digest_dir / f"{query_slug}_{date_str}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # JSON consumed by the static frontend fallback
    payload = {
        "query": query_slug,
        "date": date_str,
        "entries": [entry.model_dump() for entry in entries],
    }
    json_path = site_digest_dir / f"{query_slug}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  wrote {md_path}")
    print(f"  wrote {json_path}")
