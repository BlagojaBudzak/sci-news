# Sci News

A fully local, zero-API-cost scientific news aggregator. A Python script
fetches new preprints from arXiv and ChemRxiv, a two-agent CrewAI pipeline
running on a local Ollama model picks the best 5 and writes them up, and a
static site displays the result with a field dropdown.

## 1. One-time setup

```powershell
# from PyCharm's terminal, inside the project folder
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Install [Ollama for Windows](https://ollama.ai), then pull a 4-bit
quantized model that fits comfortably in 8GB of VRAM:

```powershell
ollama pull llama3.1:8b-instruct-q4_K_M
```

Optionally set these two environment variables (System Properties →
Environment Variables, or in the shell before running Ollama) so Ollama
never tries to load a second model alongside the first and quietly blow
past your VRAM budget:

```
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_NUM_PARALLEL=1
```

## 2. Run it once, by hand

```powershell
python main.py --categories chemistry
```

Watch `nvidia-smi` (or `ollama ps`) in another terminal the first time —
you should see one model resident in VRAM for the whole run, not spiking
between agent calls.

## 3. VRAM budget — the actual math

| Component | Approx. size |
|---|---|
| Llama 3.1 8B, Q4_K_M weights | ~4.7 GB |
| KV cache @ `num_ctx=8192` | ~1–1.5 GB |
| Windows/desktop GPU overhead | ~0.5–1 GB |
| **Total** | **~6.5–7.2 GB of 8 GB** |

That leaves a thin margin, which is why `config.py` does three things on
purpose:

1. **One shared `LLM` instance for both agents.** If the Reviewer and
   Writer pointed at two different models, Ollama would unload one to load
   the other on every call — slow, and it briefly holds both in memory
   during the swap.
2. **`num_ctx=8192`**, not the model's max. Context window directly sets
   KV-cache VRAM; doubling it roughly doubles that line in the table above.
3. **Abstracts truncated to `ABSTRACT_TRUNCATE_CHARS` (600 chars)** and
   `MAX_PAPERS_PER_SOURCE` capped at 20 before they ever reach the prompt,
   so a busy week doesn't silently overflow the context window.

If you still see VRAM pressure: switch `OLLAMA_MODEL` in `config.py` to
`"ollama/mistral:7b-instruct-q4_K_M"` (smaller footprint), or drop
`OLLAMA_NUM_CTX` to `4096` and lower `MAX_PAPERS_PER_SOURCE` to ~12.

## 4. A note on local-model reliability

`output_pydantic` asks the model to return valid, schema-matching JSON.
GPT-4-class models are very reliable at this; an 8B local model
occasionally isn't — it might wrap the JSON in a sentence, or in a code
fence. `main.py` has a fallback that tries to parse `result.raw` if
`result.pydantic` comes back empty. If a category keeps failing to parse,
lower `OLLAMA_TEMPERATURE` in `config.py` (0.1–0.2) before touching
anything else — determinism helps structured output a lot more than
prompt tweaking does.

## 5. Automate it on Windows

1. Edit `PROJECT_DIR` at the top of `run_pipeline.bat` to your actual path.
2. Open **Task Scheduler** → *Create Task*.
3. **Trigger**: Weekly (or daily), whatever cadence you want the digest
   refreshed.
4. **Action**: Start a program → Program/script: `run_pipeline.bat`,
   Start in: your project folder.
5. Under *Conditions*, uncheck "Start the task only if the computer is on
   AC power" if this is a laptop.
6. Run the task once manually from Task Scheduler to confirm it works
   end-to-end (including the `git push`) before trusting the schedule.

## 6. Deploy to GitHub Pages

```powershell
git init
git remote add origin https://github.com/<you>/sci-news-aggregator.git
git add .
git commit -m "Initial commit"
git push -u origin main
```

Then in the repo on GitHub: **Settings → Pages → Source**, pick the `main`
branch and the `/site` folder, save. Your dashboard is now live at
`https://<you>.github.io/sci-news-aggregator/`, and every scheduled run
that pushes new JSON into `site/digests/` updates it automatically —
GitHub Pages needs no rebuild step since it's plain static HTML/JS.

**Vercel alternative:** `vercel --cwd site` (or connect the repo in the
Vercel dashboard and set the root directory to `site/`) works identically,
with the advantage of instant cache invalidation on push.

## 7. Adding a new field

Add an entry to `CATEGORIES` in `config.py` with the arXiv category codes
(see the [arXiv taxonomy](https://arxiv.org/category_taxonomy)) and/or
ChemRxiv search terms, then add a matching `<option>` to
`site/index.html`'s `<select>`. Nothing else needs to change.
