# Sci News

**An AI-powered science news and research blog.** Enter a topic or question and Sci News searches recent scientific literature, selects up to five relevant papers, and turns them into accessible blog summaries with links back to the original research.

The site is designed around free-form searches—not fixed categories.

## How a search works

```text
User topic or question
        ↓
LLM creates a search plan
        ↓
OpenAlex + arXiv search
        ↓
Python deduplication and relevance filtering
        ↓
Science Editor selects up to 5 real papers
        ↓
Science Writer drafts one blog summary per paper
        ↓
Fact-checker checks claims against the available abstracts
        ↓
Python restores trusted metadata and source links
        ↓
Five-or-fewer article cards on the website
```

The LLM does not choose the final URLs or invent the metadata shown for a paper. Python matches generated drafts back to fetched records, and the pipeline records a trace for debugging. Abstract-based fact-checking is a useful safeguard, but it is not a substitute for reading the original paper.

## Project structure

- `app.py` — FastAPI API with background research jobs and SQLite job persistence.
- `src/engine.py` — the query-driven research pipeline.
- `src/query_interpreter.py` — turns a natural-language topic into search phrases and filtering rules.
- `src/openalex.py`, `src/arxiv.py` — literature retrieval.
- `src/prefilter.py` — deterministic deduplication and candidate filtering.
- `src/crew_setup.py` — structured reviewer and writer outputs.
- `src/fact_checker.py` — claim validation against fetched abstracts.
- `src/trace.py` — per-job pipeline traces.
- `site/` — static HTML/CSS/JavaScript frontend.
- `site/digests/search/` — pre-generated example digests used as a fast path.
- `data/jobs.db`, `data/raw/`, `data/traces/` — local runtime state and debugging artifacts; these should not be committed.

## Run locally

Requirements: Python, [Ollama](https://ollama.com/), and the model configured in `config.py` (currently `llama3.1:8b-instruct-q4_K_M`).

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b-instruct-q4_K_M
```

Start the API in one terminal:

```bash
uvicorn app:app --reload --port 8000
```

Serve the frontend from the repository root in a second terminal:

```bash
python -m http.server 8080 --directory site
```

Open [http://localhost:8080](http://localhost:8080), enter a topic, and submit a search. The frontend first checks for a pre-generated example digest; if one is not available, it submits a job to the API and polls for the result.

You can also run a search from the terminal:

```bash
python main.py "sodium-ion batteries"
# or
python search.py "hydrogen storage"
```

The API exposes:
- `GET /health` — liveness check.
- `POST /research` with JSON such as `{"query":"hydrogen storage"}` — start a job.
- `GET /research/{job_id}` — check job status and retrieve the result.

Run automated tests with:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
python -m pytest test_pipeline_logic.py -v
```

## Deployment status and limitations

GitHub Pages serves only the static files in `site/`; it does **not** run the Python/FastAPI backend. The repository's current Pages workflow publishes the frontend, while the API currently defaults to `http://localhost:8000` for local development.

That means arbitrary live searches work in a local setup as described above. To enable them for public visitors, deploy the FastAPI backend separately, set `site/config.js`'s `apiBaseUrl` to that backend's HTTPS URL, and configure the backend's `CORS_ORIGINS` environment variable to allow the GitHub Pages origin. Do not put secret API keys in frontend JavaScript: anything shipped to the browser is public.

The API currently uses a single-process, thread-based job runner and local SQLite storage. It is intentionally a small learning-project architecture, not a distributed production job system. If public traffic becomes a goal, review request limits, abuse protection, process restarts, and hosting costs before opening the API to unrestricted use.

## Technology

Python · FastAPI · CrewAI · Ollama · Pydantic · OpenAlex · arXiv · SQLite · HTML · CSS · Vanilla JavaScript

## Project goal

Sci News explores how to build a grounded LLM application around real scientific sources: retrieval, filtering, structured generation, validation, fact-checking, and a simple public-facing reading experience.
