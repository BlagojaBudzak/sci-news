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
CATEGORIES = {
    "chemistry": {
        "label": "Chemistry",
        "arxiv_categories": ["physics.chem-ph"],
        "chemrxiv_terms": [],   # ChemRxiv is no longer used
        # OpenAlex primary discovery configuration
        "openalex": {
            "search_query": (
                "chemistry OR chemical reaction OR chemical synthesis "
                "OR catalysis OR catalyst OR electrochemistry OR electrochemical "
                "OR battery OR molecular OR molecule OR materials OR polymer "
                "OR organic chemistry OR inorganic chemistry OR spectroscopy"
            ),
            "per_page": 50,
            "max_results": 30,
        },
        # Deterministic pre-filter. We intentionally keep these rules in
        # configuration so adding a field does not require editing the
        # filtering algorithm.
        "prefilter": {
            "positive_keywords": {
                "chemistry": 2.0,
                "chemical reaction": 2.0,
                "chemical synthesis": 2.0,
                "catalysis": 2.5,
                "catalyst": 2.0,
                "electrochemistry": 2.5,
                "electrochemical": 2.0,
                "battery": 1.5,
                "molecular": 1.5,
                "molecule": 1.5,
                "materials": 1.0,
                "material": 1.0,
                "polymer": 2.0,
                "organic": 1.5,
                "inorganic": 1.5,
                "spectroscopy": 2.0,
                "reaction": 1.0,
                "synthesis": 1.5,
                "nanomaterial": 2.0,
            },
            "negative_keywords": {
                "fluid dynamics": 2.0,
                "computational fluid dynamics": 2.5,
                "turbulence": 2.0,
                "general relativity": 3.0,
                "cosmology": 3.0,
                "astrophysics": 3.0,
                "plasma physics": 2.5,
                "quantum information": 2.0,
                "particle physics": 3.0,
            },
            "min_abstract_chars": 120,
            "min_relevance_score": 1.0,
            "positive_match_required": True,
            "relevance_weight": 1.0,
            "recency_weight": 1.0,
            "recency_half_life_days": 14.0,
        },
    },
    "physics": {
        "label": "Physics",
        "arxiv_categories": ["physics.gen-ph", "cond-mat.mtrl-sci", "quant-ph"],
        "chemrxiv_terms": [],   # no ChemRxiv
        # No openalex config yet -> OpenAlex fetch will return []
    },
    "seismology": {
        "label": "Seismology",
        "arxiv_categories": ["physics.geo-ph"],
        "chemrxiv_terms": [],   # no ChemRxiv
        # No openalex config yet
    },
}

# ---------------------------------------------------------------------------
# Fetch behavior
# ---------------------------------------------------------------------------
LOOKBACK_DAYS = 7
MAX_PAPERS_PER_SOURCE = 10
REVIEWER_ABSTRACT_CHARS = 800
WRITER_ABSTRACT_CHARS = 1400

# ---------------------------------------------------------------------------
# Local LLM (Ollama)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "ollama/llama3.1:8b-instruct-q4_K_M"
OLLAMA_NUM_CTX = 4096
OLLAMA_TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_DIGEST_DIR = ROOT / "data" / "digests"
SITE_DIGEST_DIR = ROOT / "site" / "digests"
DATA_TRACE_DIR = ROOT / "data" / "traces"

for _d in (DATA_RAW_DIR, DATA_DIGEST_DIR, SITE_DIGEST_DIR, DATA_TRACE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FACT_CHECKER_MAX_REVISION_ROUNDS = 1
FACT_CHECKER_ABSTRACT_CHARS = 1400  # same as writer