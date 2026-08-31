"""
Central configuration for the Sci News pipeline.

Editing this file is the main way you'll customize the project — add a
category, point it at different arXiv subject classes, or swap the local
model — without touching the fetch/agent/write logic.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
# arXiv does NOT have a dedicated "Chemistry" category — the closest fit is
# physics.chem-ph (Chemical Physics). For true chemistry (organic synthesis,
# materials, catalysis, etc.) we pull from ChemRxiv instead, and use arXiv
# for the physics-flavored corner of chemistry. Add/replace category codes
# freely: https://arxiv.org/category_taxonomy
CATEGORIES = {
    "chemistry": {
        "label": "Chemistry",
        "arxiv_categories": ["physics.chem-ph"],
        "chemrxiv_terms": ["chemistry"],
    },
    "physics": {
        "label": "Physics",
        "arxiv_categories": ["physics.gen-ph", "cond-mat.mtrl-sci", "quant-ph"],
        "chemrxiv_terms": [],
    },
    "seismology": {
        "label": "Seismology",
        # Seismology lives under Geophysics on arXiv.
        "arxiv_categories": ["physics.geo-ph"],
        "chemrxiv_terms": [],
    },
}

# ---------------------------------------------------------------------------
# Fetch behavior
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 7                  # "papers from the past week"
MAX_PAPERS_PER_SOURCE = 10         # cap per source -> ~20 candidates/category
                                   # reach the Reviewer; the Writer only
                                   # ever sees the 5 it selects (see
                                   # src/crew_setup.py).
# ABSTRACT_TRUNCATE_CHARS = 600    # trims abstracts before they hit the LLM prompt
REVIEWER_ABSTRACT_CHARS = 800      # enough context for ranking ~10 candidates
WRITER_ABSTRACT_CHARS = 1400       # more detail for accurate science summaries


# ---------------------------------------------------------------------------
# Local LLM (Ollama) — tuned for an 8GB VRAM card
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"

# Pull this once with: ollama pull llama3.1:8b-instruct-q4_K_M
# Swap for "ollama/mistral:7b-instruct-q4_K_M" if you want a slightly smaller
# footprint. See README.md "VRAM budget" section before changing this.
OLLAMA_MODEL = "ollama/llama3.1:8b-instruct-q4_K_M"

# Context window handed to Ollama via num_ctx. Larger = more papers per
# prompt but more VRAM used for the KV cache. 4096 is not a hard floor —
# it's simply enough for the Reviewer's ~20-candidate prompt (the biggest
# one in the pipeline, now that the Writer only ever sees its 5 already-
# selected papers) on top of the ~4.7GB of Q4_K_M weights on an 8GB card.
# Raise it if you raise MAX_PAPERS_PER_SOURCE. See README.md for the math.
OLLAMA_NUM_CTX = 4096
OLLAMA_TEMPERATURE = 0.2          # lower = more reliable JSON from output_pydantic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_RAW_DIR = ROOT / "data" / "raw"          # cached raw API responses (debugging)
DATA_DIGEST_DIR = ROOT / "data" / "digests"   # markdown archive, one file per run
SITE_DIGEST_DIR = ROOT / "site" / "digests"   # JSON consumed by the static site

for _d in (DATA_RAW_DIR, DATA_DIGEST_DIR, SITE_DIGEST_DIR):
    _d.mkdir(parents=True, exist_ok=True)