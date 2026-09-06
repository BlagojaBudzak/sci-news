"""
Output generation — turns a DigestOutput (validated Pydantic model from the
crew) into two artifacts:

  * data/digests/<category>_<date>.md   — a human-readable Markdown archive
  * site/digests/<category>.json        — what the static site fetches
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List

from src.crew_setup import BlogEntry


def write_digest(
    category: str,
    entries: List[BlogEntry],
    data_digest_dir: Path,
    site_digest_dir: Path,
) -> None:
    date_str = datetime.now().strftime("%Y-%m-%d")
    data_digest_dir.mkdir(parents=True, exist_ok=True)
    site_digest_dir.mkdir(parents=True, exist_ok=True)

    # --- Markdown archive -------------------------------------------------
    md_lines = [f"# Sci News — {category.title()} — {date_str}\n"]
    for e in entries:
        md_lines.append(f"## {e.title}\n")
        md_lines.append(f"{e.paragraph}\n")
        md_lines.append(f"**Read the paper:** [{e.link}]({e.link})\n")
    md_path = data_digest_dir / f"{category}_{date_str}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # --- Site JSON ----------------------------------------------------------
    payload = {
        "category": category,
        "date": date_str,
        "entries": [e.model_dump() for e in entries],
    }
    json_path = site_digest_dir / f"{category}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  wrote {md_path}")
    print(f"  wrote {json_path}")
