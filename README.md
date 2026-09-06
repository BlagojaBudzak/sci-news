# 🔬 Sci News

**A weekly AI-powered science news digest built with CrewAI, Ollama, Python, and vanilla JavaScript.**

Sci News automatically collects recent scientific preprints, uses a local LLM to select the most interesting papers, writes accessible summaries, and publishes the results to a lightweight static website.

The project currently covers:

- 🧪 Chemistry
- ⚛️ Physics
- 🌎 Seismology

---

## ✨ What does it do?

Every week, Sci News runs a small agentic pipeline:

```text
              arXiv / ChemRxiv
                     │
                     ▼
              Paper Collection
                     │
                     ▼
            ┌─────────────────┐
            │  Senior Science │
            │     Editor      │
            └────────┬────────┘
                     │
             Selects 5 papers
                     │
                     ▼
            Deterministic Python
               validation
                     │
                     ▼
            ┌─────────────────┐
            │   Science       │
            │     Writer      │
            └────────┬────────┘
                     │
             Writes summaries
                     │
                     ▼
            Python validation
                     │
                     ▼
                  JSON
                     │
                     ▼
              Static Website
```

The important design principle is that **the LLM is not treated as the source of truth**.

The agents make decisions and generate text, while ordinary Python code handles data validation, metadata, filtering, and publishing.

---

## 🧠 Why two agents?

Sci News uses two specialized agents instead of asking one LLM to do everything.

### Senior Science Editor

Reviews the candidate papers and selects the five most interesting based on:

- scientific impact
- novelty
- broad interest
- quality of the available abstract

The reviewer returns structured output containing the paper IDs and selection reasons.

### Science Writer

Receives **only the five papers selected by the reviewer** and produces a short, accessible science-news summary for each.

This creates a simple multi-agent workflow:

```text
Reviewer
   ↓
5 paper IDs
   ↓
Python selects the actual papers
   ↓
Writer receives only those papers
```

---

## 🛡️ Keeping the LLM grounded

One of the main goals of the project is to avoid letting the model become the source of truth.

The pipeline uses Pydantic models and regular Python logic to validate the agents' output.

For example, the reviewer may return:

```json
{
  "selected_papers": [
    {
      "id": "2608.28025v1",
      "reason": "..."
    }
  ]
}
```

Python then matches that ID against the real fetched papers.

If an agent invents an ID, it is rejected.

The same principle is used after the Writer finishes:

```text
LLM output
    ↓
paper ID
    ↓
Python lookup
    ↓
real URL + reviewer reason + metadata
```

This keeps things such as paper URLs and source metadata outside the LLM's control.

---

## 🧰 Tech Stack

### AI / Agents

- [CrewAI](https://github.com/crewAIInc/crewAI)
- [Ollama](https://ollama.com/)
- Llama 3.1 8B
- Pydantic

### Data

- Python
- arXiv API
- ChemRxiv API
- JSON
- Markdown

### Frontend

- HTML
- CSS
- Vanilla JavaScript

### Deployment

- Git
- GitHub
- GitHub Pages

---

## 📁 Project Structure

```text
sci-news-aggregator/
│
├── main.py
├── config.py
├── run_pipeline.bat
├── README.md
│
├── src/
│   ├── crew_setup.py
│   ├── fetcher.py
│   ├── digest_writer.py
│   └── trace.py
│
├── tests/
│   ├── test_pipeline_logic.py
│   └── test_pipeline_smoke.py
│
├── data/
│   ├── raw/
│   │   └── ...
│   │
│   ├── digests/
│   │   └── ...
│   │
│   └── traces/
│       └── ...
│
└── site/
    ├── index.html
    ├── script.js
    ├── style.css
    │
    └── digests/
        ├── chemistry.json
        ├── physics.json
        └── seismology.json
```

### `data/raw/`

Cached paper metadata fetched from the external sources.

### `data/digests/`

Human-readable Markdown versions of generated digests.

### `data/traces/`

One JSON file per pipeline run, recording what happened at each stage — see [Observability](#-observability) below. Not committed to the public site; these are debugging artifacts for the developer.

### `site/digests/`

JSON files consumed directly by the website.

---

## 🚀 Running locally

Clone the repository:

```bash
git clone https://github.com/BlagojaBudzak/sci-news-aggregator.git
cd sci-news-aggregator
```

Create and activate the Python environment:

```powershell
python -m venv scivenv
scivenv\Scripts\activate
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

Make sure Ollama is installed and running, then pull the configured model:

```powershell
ollama pull llama3.1:8b-instruct-q4_K_M
```

Run a single category:

```powershell
python main.py --categories chemistry
```

Or run all configured categories:

```powershell
python main.py --categories chemistry physics seismology
```

---

## ✅ Running the tests

The tests cover the pure-Python logic (id matching, metadata hydration, the trace) with the LLM/network calls faked out — they run in under a second and don't touch arXiv, ChemRxiv, or Ollama.

```powershell
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

---

## 🌐 Running the website locally

The generated frontend is located in `site/`.

Start a simple local web server:

```powershell
cd site
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

The JavaScript frontend automatically loads the generated JSON files:

```text
site/digests/chemistry.json
site/digests/physics.json
site/digests/seismology.json
```

No frontend build system is required.

---

## ⚙️ Automated pipeline

The project also includes `run_pipeline.bat` for scheduled execution on Windows.

The intended workflow is:

```text
Windows Task Scheduler
        ↓
run_pipeline.bat
        ↓
Ollama
        ↓
main.py
        ↓
Generate all category digests
        ↓
Update site/digests/
        ↓
Git commit
        ↓
Git push
        ↓
GitHub Pages
```

This allows the site to become a continuously refreshed weekly science publication with minimal manual intervention.

---

## 🔍 Data flow

For each category, the pipeline performs four major steps:

### 1. Fetch

Recent papers are collected from arXiv and, when available, ChemRxiv.

```text
External APIs
     ↓
Candidate papers
```

### 2. Review

The Senior Science Editor selects the most interesting papers.

```text
~10–20 candidates
      ↓
Reviewer Agent
      ↓
5 selected IDs + reasons
```

### 3. Write

Python resolves those IDs back to the actual papers before constructing the Writer's prompt.

```text
5 selected IDs
      ↓
Python lookup
      ↓
5 real papers
      ↓
Writer Agent
      ↓
5 summaries
```

### 4. Hydrate & publish

Python combines the generated summaries with trusted metadata:

```text
LLM title + paragraph
          +
real paper URL
          +
reviewer reason
          ↓
       JSON
          ↓
      Website
```

---

## 🔍 Observability

Every run of `main.py` produces a trace: a per-stage record of what happened, printed to the console and saved as JSON under `data/traces/`.

```text
JOB #20260906-125536  (chemistry)

FETCH
  20 papers found  (3.4s)

REVIEWER
  5 selected  (14.2s)

WRITER
  5 articles written  (18.9s)

PUBLISH
  5 entries published  (0.0s)
```

If a stage fails — the reviewer's output doesn't parse, the writer references an id that doesn't exist — the trace records which stage failed, why, and how long it ran before failing, instead of the run just disappearing into console scrollback. This is what makes it possible to say "the reviewer failed on job 20260906-125536" instead of "something went wrong somewhere."

See `src/trace.py` for the implementation, and `tests/test_pipeline_smoke.py` for an example run with the LLM calls faked out.

---

## 🎯 Project goals

This project is primarily a learning project for exploring:

- LLM applications
- agentic workflows
- CrewAI
- local LLM inference
- structured LLM outputs
- Pydantic validation
- API-based data ingestion
- deterministic code surrounding probabilistic models
- static website generation
- automated publishing

The goal is not simply to make an LLM summarize papers.

The goal is to understand **how to build a reliable system around an LLM**.

---

## 🧪 Current status

| Component | Status |
|---|---|
| arXiv ingestion | ✅ |
| ChemRxiv ingestion | ✅ |
| Paper caching | ✅ |
| Reviewer agent | ✅ |
| Reviewer → Python handoff | ✅ |
| Paper selection validation | ✅ |
| Writer agent | ✅ |
| Structured writer output | ✅ |
| Metadata hydration | ✅ |
| Markdown digest generation | ✅ |
| JSON generation | ✅ |
| Author / date / source metadata in JSON | ✅ |
| Per-run observability trace | ✅ |
| Local website | ✅ |
| Category switching | ✅ |
| GitHub Pages deployment | ✅ |
| Deterministic pre-filtering (before the Reviewer sees candidates) | 🚧 |
| Fact-checking stage | 🚧 |
| Automated weekly publishing | 🚧 |
| UI redesign | 🚧 |

---

## 🔮 Future ideas

Possible future improvements include:

- Better paper ranking and deduplication
- More scientific categories
- More robust handling of API failures
- Abstract quality checks
- Automatic word-count validation
- Search and filtering
- Paper source badges
- Publication dates and authors
- DOI / arXiv metadata
- Visualizations and scientific figures
- Improved mobile UI
- Automated weekly deployment
- Email or RSS digests

---

## 📌 Philosophy

Sci News deliberately keeps the architecture simple.

Instead of hiding everything behind a framework, the project separates responsibilities:

```text
Python
→ data + orchestration + validation

CrewAI
→ agent behavior

Ollama
→ local LLM inference

JSON
→ frontend data layer

JavaScript
→ rendering

HTML/CSS
→ presentation
```

The result is a small system where every step can be inspected, tested, and understood.

---

## 👤 Author

Built by **Blagoja Budzakoski** while exploring agentic AI, scientific computing, and the intersection of chemistry and machine learning.

---

## 📄 License

This project is released under the MIT License.
