"""
Central configuration for the Sci News pipeline.
"""
import os

os.environ.setdefault("CREWAI_TESTING", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from pathlib import Path

ROOT = Path(__file__).resolve().parent

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
OLLAMA_MODEL = "ollama/qwen3:4b-instruct-2507-q4_K_M"
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