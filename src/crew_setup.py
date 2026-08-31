"""
Agentic pipeline — two agents, one shared local model, two phases.

    candidate papers (~20)
            |
            v
      Reviewer Agent  <-- build_review_crew() / kicked off in main.py
            |
      picks 5 ids + reasons  (ReviewerSelection, via output_pydantic)
            |
            v
   select_papers()  <-- plain Python, narrows the candidate list down
            |             to just the 5 the Reviewer picked
            v
       Writer Agent  <-- build_write_crew() / kicked off in main.py
            |             sees ONLY those 5 papers' title/abstract/URL
            v
   5 blog entries  (DigestOutput, via output_pydantic)

Each phase is its own single-task `Crew`. This is deliberate, not
incidental: the Writer's prompt has to be *built from the Reviewer's
output*, and a Task's description is fixed Python text at construction
time — there's no way to hand a Task data it doesn't have yet. Splitting
into two crews lets ordinary Python sit between the two agent calls to do
that narrowing, which is exactly what `select_papers()` does. `main.py`
runs both phases, one `.kickoff()` each, back to back.

Design choices worth noting:

1. Both agents share a single `LLM` instance pointed at the same Ollama
   model, passed in from `main.py` and reused across both phases and every
   category. Ollama can only keep so much in an 8GB card at once — if the
   Reviewer and Writer used *different* models, Ollama would unload/reload
   between every call, which is slow and can transiently spike VRAM during
   the swap. One resident model avoids that entirely.

2. Both tasks use `output_pydantic` instead of asking the LLM to return
   free-text Markdown. Small local models are noticeably less reliable at
   following loose formatting instructions than GPT-4-class models, so we
   ask CrewAI to validate/parse structured JSON for us rather than
   regex-scraping prose later. The Reviewer's structured `id` list is also
   what makes `select_papers()` possible in the first place.

3. Paper JSON is embedded directly into each Task's `description` with an
   f-string, not passed through `crew.kickoff(inputs=...)`. CrewAI's
   inputs-templating does a naive `{key}` substitution, which breaks on the
   literal curly braces inside JSON — so we render the prompt ourselves.

4. The Writer task has no `context=[...]`. That parameter forwards a prior
   task's output *within the same crew*, but the Writer now lives in a
   separate `Crew` from the Reviewer, and its prompt is already built from
   the Reviewer's decision (via `select_papers()`) before the Writer's
   `Task` even exists — there's nothing left for `context` to add.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from crewai import Agent, Crew, LLM, Process, Task
from pydantic import BaseModel

from config import (
    ABSTRACT_TRUNCATE_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
)


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------
class SelectedPaper(BaseModel):
    id: str
    reason: str


class ReviewerSelection(BaseModel):
    selected_papers: List[SelectedPaper]


class BlogEntry(BaseModel):
    title: str
    paragraph: str
    link: str


class DigestOutput(BaseModel):
    entries: List[BlogEntry]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
def get_local_llm() -> LLM:
    return LLM(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=OLLAMA_TEMPERATURE,
    )


def _papers_for_prompt(papers: List[Dict]) -> str:
    trimmed = [
        {
            "id": p["id"],
            "title": p["title"],
            "abstract": p["abstract"][:ABSTRACT_TRUNCATE_CHARS],
        }
        for p in papers
    ]
    return json.dumps(trimmed, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
def _build_reviewer(llm: LLM, category: str) -> Agent:
    return Agent(
        role="Senior Science Editor",
        goal=(
            f"Select the 5 most impactful, novel, or broadly interesting new "
            f"{category} papers from this week's preprint feed."
        ),
        backstory=(
            "You've spent 15 years as an editor at a respected science "
            "magazine. You have a nose for genuinely important results and "
            "no patience for incremental, over-hyped, or narrow findings."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def _build_writer(llm: LLM) -> Agent:
    return Agent(
        role="Science Writer",
        goal=(
            "Turn dense abstracts into short, accurate, and genuinely "
            "engaging blog paragraphs for curious non-specialist readers."
        ),
        backstory=(
            "You write the popular-science column that scientists "
            "themselves enjoy reading, because it never oversimplifies to "
            "the point of being wrong."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
def _build_review_task(papers: List[Dict], category: str, reviewer: Agent) -> Task:
    papers_json = _papers_for_prompt(papers)
    return Task(
        description=(
            f"Here are {len(papers)} recent {category} preprints as a JSON "
            f"array of objects with 'id', 'title', and 'abstract':\n\n"
            f"{papers_json}\n\n"
            "Pick exactly 5 papers that are most scientifically impactful, "
            "novel, or broadly interesting to an intelligent general "
            "audience. Give a one-sentence reason for each pick. Only use "
            "ids that appear in the JSON above — never invent one."
        ),
        expected_output=(
            "A JSON object with a 'selected_papers' list of exactly 5 "
            "items, each with 'id' and 'reason'."
        ),
        agent=reviewer,
        output_pydantic=ReviewerSelection,
    )


def _build_write_task(selected_papers: List[Dict], category: str, writer: Agent) -> Task:
    """Builds the Writer's task from an *already-narrowed* paper list.

    Only `selected_papers` (the output of `select_papers()`) reaches the
    prompt — never the original candidate set — which is the fix for the
    "Writer sees everything" problem this pipeline used to have.
    """
    papers_json = _papers_for_prompt(selected_papers)
    id_to_url = {p["id"]: (p.get("url") or p.get("doi") or "") for p in selected_papers}
    n = len(selected_papers)

    return Task(
        description=(
            f"The Reviewer has already chosen the following {n} {category} "
            "papers as this week's most impactful, novel, or interesting "
            "picks. Write one engaging, accurate paragraph (roughly "
            "120-180 words) for each, suitable for a general-interest "
            "science blog. Explain what was found, why it matters, and "
            "avoid unexplained jargon. Do not invent results that aren't "
            "supported by the abstract, and do not discuss any paper other "
            "than the ones listed below — this is the complete set.\n\n"
            f"Metadata for each selected paper (id -> title/abstract):\n\n"
            f"{papers_json}\n\n"
            "Use this id-to-URL mapping for the 'link' field of each "
            f"entry — do not invent a URL:\n\n{json.dumps(id_to_url)}"
        ),
        expected_output=(
            f"A JSON object with an 'entries' list of exactly {n} items, "
            "each with 'title', 'paragraph', and 'link' (the paper's "
            "URL/DOI — do not put the link inside the paragraph text "
            "itself)."
        ),
        agent=writer,
        output_pydantic=DigestOutput,
    )


# ---------------------------------------------------------------------------
# Reviewer -> Writer handoff (plain Python, no LLM call)
# ---------------------------------------------------------------------------
def select_papers(papers: List[Dict], selection: ReviewerSelection) -> List[Dict]:
    """Narrows the original candidate list down to the Reviewer's picks.

    This is the Python step between the two agent calls. It also guards
    against the Reviewer hallucinating an id that was never in the
    candidate set — such ids are logged and skipped rather than crashing
    the pipeline or silently reaching the Writer as a KeyError-shaped bug.
    Order follows the Reviewer's selection order.
    """
    by_id = {p["id"]: p for p in papers}
    selected = []
    for pick in selection.selected_papers:
        paper = by_id.get(pick.id)
        if paper is None:
            print(f"  ! reviewer picked an id not in the candidate set, skipping: {pick.id!r}")
            continue
        selected.append(paper)
    return selected


# ---------------------------------------------------------------------------
# Crews — one per phase
# ---------------------------------------------------------------------------
def build_review_crew(papers: List[Dict], category: str, llm: Optional[LLM] = None) -> Crew:
    """Phase 1: the Reviewer looks at every candidate and picks 5."""
    if not papers:
        raise ValueError("build_review_crew() called with no papers to review")

    llm = llm or get_local_llm()
    reviewer = _build_reviewer(llm, category)
    review_task = _build_review_task(papers, category, reviewer)

    return Crew(
        agents=[reviewer],
        tasks=[review_task],
        process=Process.sequential,
        verbose=True,
    )


def build_write_crew(selected_papers: List[Dict], category: str, llm: Optional[LLM] = None) -> Crew:
    """Phase 2: the Writer drafts blog entries for the 5 selected papers
    only. Call `select_papers()` to produce `selected_papers` first.
    """
    if not selected_papers:
        raise ValueError("build_write_crew() called with no selected papers")

    llm = llm or get_local_llm()
    writer = _build_writer(llm)
    write_task = _build_write_task(selected_papers, category, writer)

    return Crew(
        agents=[writer],
        tasks=[write_task],
        process=Process.sequential,
        verbose=True,
    )