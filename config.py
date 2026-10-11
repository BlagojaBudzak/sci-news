"""
Central configuration for the query-driven Sci News research engine.
"""
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Retrieval and prompt sizes
MAX_PAPERS_PER_SOURCE = 10
REVIEWER_ABSTRACT_CHARS = 800
WRITER_ABSTRACT_CHARS = 1400

# Local LLM (Ollama)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "ollama/llama3.1:8b-instruct-q4_K_M"
OLLAMA_NUM_CTX = 4096
OLLAMA_TEMPERATURE = 0.2

# Runtime paths
DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_DIGEST_DIR = ROOT / "data" / "digests"
SITE_DIGEST_DIR = ROOT / "site" / "digests"
DATA_TRACE_DIR = ROOT / "data" / "traces"

for _directory in (DATA_RAW_DIR, DATA_DIGEST_DIR, SITE_DIGEST_DIR, DATA_TRACE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

FACT_CHECKER_MAX_REVISION_ROUNDS = 1
FACT_CHECKER_ABSTRACT_CHARS = 1400
