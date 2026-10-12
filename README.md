# Sci News

**An AI-powered science news blog built around real research.**

Enter a scientific topic or question—such as `perovskite solar cells`, `solid-state batteries`, or `CRISPR gene editing`—and Sci News searches scholarly literature, selects up to five relevant papers, and creates accessible summaries with links to the source publications.

Sci News is a learning project focused on building a grounded LLM application: retrieval, ranking, structured generation, validation, and a usable website.

## How it works

```text
Topic or question
       |
       v
LLM query planner
(search phrases, relevance terms, time window)
       |
       +-------------------+
       |                   |
       v                   v
    OpenAlex             arXiv
       |                   |
       +---------+---------+
                 |
                 v
     Python deduplication and filtering
                 |
                 v
      Prefer original research when possible
                 |
                 v
        Science Editor selects up to 5
                 |
                 v
        Science Writer drafts summaries
                 |
                 v
      Fact-check claims against abstracts
                 |
                 v
   Python restores trusted paper metadata/URLs
                 |
                 v
     JSON digest + website article cards
```

Python owns paper IDs, source metadata, deduplication, ranking, and final source links. The LLM does not get to invent the URLs attached to its generated summaries. Each run can also write a pipeline trace to help diagnose retrieval, parsing, and generation problems.

**The fact-checker is an additional safeguard, not proof of full-paper accuracy.** It evaluates claims against the abstracts retrieved for a search; it does not independently read every full paper.

## Research window and retrieval limits

Python now parses common time expressions deterministically: **60 days by default**, `last week` = 7 days, `past month` = 30 days, `past year` = 365 days, `this year` = year-to-date, and `since YYYY` = days since January 1 of that year. All windows are capped at 365 days, so older `since YYYY` requests search only the most recent year. The LLM proposes topical search phrases and keywords but cannot override Python's date-window decision.

A longer window does not mean the model reads every paper published during that period. Retrieval is intentionally bounded: the current engine requests up to 30 OpenAlex records and 10 arXiv records, then keeps at most 15 candidates for review. This makes searches manageable on a local machine, but it also means Sci News is a focused digest generator—not an exhaustive systematic-literature-review tool.

The pipeline prefers original research where enough candidates are available. Review articles may still be selected when there are not enough suitable original-research candidates.

## Technology

- **Backend:** Python, FastAPI, SQLite
- **LLM orchestration:** CrewAI, Pydantic
- **Local inference:** Ollama
- **Research sources:** OpenAlex and arXiv
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Hosting:** GitHub Pages for the static frontend

## Run locally

### Requirements

- Python 3.12 is the tested development environment.
- [Ollama](https://ollama.com/) installed and running.
- The model configured in `config.py` (currently `llama3.1:8b-instruct-q4_K_M`).

### 1. Create an environment and install dependencies

Windows PowerShell:

```powershell
python -m venv scivenv
.\scivenv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
ollama pull llama3.1:8b-instruct-q4_K_M
```

Linux/macOS:

```bash
python -m venv scivenv
source scivenv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
ollama pull llama3.1:8b-instruct-q4_K_M
```

If Ollama is already running, there is no need to start a second `ollama serve` process.

### 2. Run a search from the terminal

```powershell
python search.py "perovskite solar cells"
```

The compatibility entry point also works:

```powershell
python main.py "hydrogen storage"
```

The search pipeline writes a dated Markdown digest under `data/digests/search/`, a JSON digest under `site/digests/search/`, and a trace under `data/traces/`. Local runtime files and generated Markdown/traces are ignored by Git; selected JSON examples in `site/digests/search/` are kept in the repository for the static-site fast path.

### 3. Run the API and website

Start FastAPI in one terminal:

```powershell
uvicorn app:app --reload --port 8000
```

Start the static frontend in a second terminal:

```powershell
python -m http.server 8080 --directory site
```

Open [http://localhost:8080](http://localhost:8080). Enter a topic and submit it. The page first checks a pinned example digest or a fresh saved snapshot (14-day freshness window); stale non-example snapshots fall through to live research. It then submits a research job to FastAPI and polls for the result.

The API currently exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/research` | Start a research job with JSON such as `{"query":"perovskite solar cells"}` |
| `GET` | `/research/{job_id}` | Check job status and retrieve the result |

FastAPI's interactive endpoint documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) while the backend is running.

### Optional environment settings

`config.py` loads a project-root `.env` file. Useful settings include:

- `OLLAMA_BASE_URL` — Ollama endpoint; defaults to `http://localhost:11434`.
- `OLLAMA_MODEL` — model used for query planning and CrewAI tasks.
- `ENVIRONMENT` — use `production` to activate the production CORS default.
- `CORS_ORIGINS` — comma-separated allowed origins; overrides the CORS default.
- `SCI_NEWS_API_KEY` — optional API request gate. A key placed in public frontend JavaScript is not secret, so this does not by itself protect a publicly deployed API.

## Tests

Run the full automated test suite from the repository root:

```powershell
python -m pytest tests/ -v
```

The tests mock external retrieval and LLM calls where appropriate, so they validate pipeline and API logic without requiring a live Ollama inference for every test. Keep a separate live-search smoke test—such as `python search.py "perovskite solar cells"`—to verify that the external data sources and local model work together.

## Deployment: important limitation

GitHub Pages serves only the static files under `site/`. **It does not run Python or FastAPI.** The checked-in `site/config.js` defaults to `http://localhost:8000`, which is correct for local development.

For public visitors to run new searches, the API must be deployed separately. Then:

1. Set `site/config.js`'s `apiBaseUrl` to the deployed backend's HTTPS URL.
2. Set `ENVIRONMENT=production` or configure `CORS_ORIGINS` to allow the exact GitHub Pages origin.
3. Add sensible rate limits, queue bounds, abuse protection, and retention for generated jobs before exposing the API publicly.
4. Do not put private API keys or credentials in frontend JavaScript.

The API is currently a small, single-process learning-project backend with a thread-based job runner and local SQLite storage. It is not a distributed or high-traffic service. Timeouts mark queued/running jobs as failed, and a late worker cannot overwrite that terminal state; Python cannot forcibly cancel a running thread, so timed-out work may continue consuming resources until it returns. Default in-memory admission limits are 10 requests/minute and 30 requests/hour per client IP, the queue is capped at 10, and terminal job history is pruned to a 24-hour retention period / 500 stored-job cap. These controls are basic single-process protections, not a substitute for a production API gateway.

## Known limitations and next improvements

- The search window is capped at one year. Phrases such as `between 2020 and 2022` are not yet represented as exact start/end dates.
- Search retrieval remains bounded (up to 30 OpenAlex and 10 arXiv records, then at most 15 candidates), so results are not exhaustive.
- The reviewer and fact-checker use the same local model; abstract-level checking cannot establish full-paper accuracy.
- A timed-out Python thread is marked failed but cannot be forcibly stopped. For public deployment, add a proper worker queue/process supervisor, stronger abuse controls, and monitoring.
- GitHub Actions runs the automated suite on pushes and pull requests; a separate live smoke test still requires Ollama and working research-source access.

## Repository structure

```text
app.py                    FastAPI backend and job manager
search.py                 Free-form search CLI
main.py                   CLI compatibility entry point
config.py                 Local model and runtime configuration
src/
  query_interpreter.py    LLM topical plan + deterministic time window
  openalex.py             OpenAlex retrieval and normalization
  arxiv.py                arXiv retrieval and normalization
  prefilter.py            Deduplication, relevance scoring, filtering
  crew_setup.py           Structured reviewer and writer agents
  fact_checker.py         Abstract-based claim checking
  engine.py               End-to-end research pipeline
  digest_writer.py        Markdown and JSON output
  trace.py                Pipeline observability
tests/                    Automated tests
site/                     Static frontend deployed to GitHub Pages
```

