"""
End-to-end pipeline evaluation for representative queries.

Runs the research engine with all external network and LLM calls faked,
using a small set of typical queries. Verifies that the pipeline handles
PASS, PARTIAL, and FAIL fact-check outcomes correctly, and that traces and
digests are written as expected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src import engine
from src.crew_setup import (
    Claim,
    LLMDraftEntry,
    ReviewerSelection,
    SelectedPaper,
    WriterOutput,
)
from src.fact_checker import ClaimAssessment, FactCheckReport

# ---------------------------------------------------------------------------
# Representative queries
# ---------------------------------------------------------------------------
REPRESENTATIVE_QUERIES = [
    "conductive polymers",
    "solid-state batteries",
    "CRISPR gene editing",
    "quantum machine learning",
    "earthquake-resistant materials",
    "hydrogen storage",
]

# ---------------------------------------------------------------------------
# Fixed mock data (same for all queries – the pipeline logic is what's tested)
# ---------------------------------------------------------------------------
FIXTURE_PAPERS = [
    {
        "id": "paper-1",
        "title": "A Representative Paper",
        "abstract": ("This paper presents a detailed study of the topic. "
                     "We combine experimental and computational methods to "
                     "reveal new insights into the underlying mechanisms."),
        "authors": ["A. Author", "B. Author"],
        "published": "2026-08-30",
        "url": "http://example.com/paper-1",
        "doi": "10.1000/rep1",
        "source": "test",
    },
    {
        "id": "paper-2",
        "title": "Another Representative Paper",
        "abstract": ("A second study explores a related aspect of the topic "
                     "using advanced techniques. The findings demonstrate "
                     "significant improvements."),
        "authors": ["C. Author"],
        "published": "2026-08-31",
        "url": "http://example.com/paper-2",
        "doi": "10.1000/rep2",
        "source": "test",
    },
]

class _FakeCrewOutput:
    def __init__(self, pydantic_obj):
        self.pydantic = pydantic_obj
        self.raw = ""


class _FakeCrew:
    def __init__(self, output):
        self._output = output

    def kickoff(self):
        return self._output


# ---------------------------------------------------------------------------
# Mock pipeline helper
# ---------------------------------------------------------------------------
def _patch_pipeline(monkeypatch, tmp_path, fact_check_status: str):
    """Replace all external/LLM components with deterministic versions."""
    # Query interpretation
    monkeypatch.setattr(
        engine,
        "interpret_query",
        lambda query, llm=None: type("Plan", (), {
            "search_phrases": ["representative topic"],
            "positive_keywords": {},
            "negative_keywords": {},
            "lookback_days": 7,
        })(),
    )
    monkeypatch.setattr(engine, "build_openalex_query", lambda plan: "representative topic")
    monkeypatch.setattr(engine, "build_arxiv_query", lambda plan: "representative topic")

    # Fetching
    monkeypatch.setattr(engine, "fetch_openalex_papers", lambda *a, **k: FIXTURE_PAPERS)
    monkeypatch.setattr(engine, "fetch_arxiv_search", lambda *a, **k: [])

    # Reviewer
    reviewer_selection = ReviewerSelection(selected_papers=[
        SelectedPaper(id="paper-1", reason="Novel result."),
        SelectedPaper(id="paper-2", reason="Interesting."),
    ])
    monkeypatch.setattr(
        engine,
        "build_review_crew",
        lambda papers, category, llm=None: _FakeCrew(_FakeCrewOutput(reviewer_selection)),
    )

    # Writer
    writer_output = WriterOutput(articles=[
        LLMDraftEntry(
            id="paper-1",
            title="Headline 1",
            paragraph="Body text 1.",
            claims=[
                Claim(
                    claim="The paper presents a detailed study.",
                    paper_ids=["paper-1"],
                )
            ],
        ),
        LLMDraftEntry(
            id="paper-2",
            title="Headline 2",
            paragraph="Body text 2.",
            claims=[
                Claim(
                    claim="The paper demonstrates significant improvements.",
                    paper_ids=["paper-2"],
                )
            ],
        ),
    ])
    monkeypatch.setattr(
        engine,
        "build_write_crew",
        lambda papers, category, llm=None, feedback=None: _FakeCrew(_FakeCrewOutput(writer_output)),
    )

    # Fact Checker: uses the provided status
    def fake_fact_check(wo, sp, llm=None):
        assessments = []
        if fact_check_status == "PASS":
            verdict = "SUPPORTED"
            justification = "Abstract supports this claim."
        elif fact_check_status == "PARTIAL":
            verdict = "NOT_CHECKABLE"
            justification = "Abstract lacks sufficient detail."
        else:  # FAIL
            verdict = "UNSUPPORTED"
            justification = "Abstract does not support this claim."

        for article in wo.articles:
            for claim in article.claims:
                assessments.append(ClaimAssessment(
                    claim=claim.claim,
                    paper_ids=claim.paper_ids,
                    verdict=verdict,
                    justification=justification,
                ))
        return FactCheckReport(
            assessments=assessments,
            overall_status=fact_check_status,
            summary=f"All claims {fact_check_status}.",
        )

    monkeypatch.setattr(engine, "run_fact_check", fake_fact_check)

    # Redirect output directories to tmp_path
    data_digest_dir = tmp_path / "data" / "digests"
    site_digest_dir = tmp_path / "site" / "digests"
    trace_dir = tmp_path / "data" / "traces"
    monkeypatch.setattr(engine, "DATA_DIGEST_DIR", data_digest_dir)
    monkeypatch.setattr(engine, "SITE_DIGEST_DIR", site_digest_dir)
    monkeypatch.setattr(engine, "DATA_TRACE_DIR", trace_dir)

    return site_digest_dir, trace_dir


# ---------------------------------------------------------------------------
# Helper to compute the slug exactly as the engine does
# ---------------------------------------------------------------------------
def _slug_from_query(query: str) -> str:
    return "".join(
        c.lower() if c.isalnum() or c == "-" else "-"
        for c in query
    ).strip("-") or "search"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query", REPRESENTATIVE_QUERIES)
def test_pipeline_pass(query, monkeypatch, tmp_path):
    site_digest_dir, trace_dir = _patch_pipeline(monkeypatch, tmp_path, "PASS")

    result = engine.research(query, verbose=False, save_outputs=True)

    assert result.status == "completed"
    assert result.verification_status == "PASS"
    assert result.digest_entries is not None
    assert len(result.digest_entries) == 2

    # Verify digest was written using the same slug as the engine
    slug = _slug_from_query(query)
    payload_path = site_digest_dir / "search" / f"{slug}.json"
    payload = json.loads(payload_path.read_text())
    assert payload["entries"][0]["verification_status"] == "PASS"
    assert all(e["source"] == "test" for e in payload["entries"])

    # Trace exists and all stages ok
    trace_file = Path(result.trace_path)
    assert trace_file.exists()
    trace = json.loads(trace_file.read_text())
    stage_names = [s["name"] for s in trace["stages"]]
    assert "fact_checker" in stage_names
    assert "publish" in stage_names
    assert all(s["status"] == "ok" for s in trace["stages"])


@pytest.mark.parametrize("query", REPRESENTATIVE_QUERIES)
def test_pipeline_partial(query, monkeypatch, tmp_path):
    site_digest_dir, trace_dir = _patch_pipeline(monkeypatch, tmp_path, "PARTIAL")

    result = engine.research(query, verbose=False, save_outputs=True)

    assert result.status == "completed"
    assert result.verification_status == "PARTIAL"
    assert result.digest_entries is not None
    assert len(result.digest_entries) == 2

    # Digest still published but with PARTIAL status
    slug = _slug_from_query(query)
    payload_path = site_digest_dir / "search" / f"{slug}.json"
    payload = json.loads(payload_path.read_text())
    assert payload["entries"][0]["verification_status"] == "PARTIAL"


@pytest.mark.parametrize("query", REPRESENTATIVE_QUERIES)
def test_pipeline_fail(query, monkeypatch, tmp_path):
    site_digest_dir, trace_dir = _patch_pipeline(monkeypatch, tmp_path, "FAIL")

    result = engine.research(query, verbose=False, save_outputs=True)

    assert result.status == "fact_check_failed"
    assert result.digest_entries is None
    assert "Fact-check overall status is FAIL" in result.error

    # No digest file should be created
    slug = _slug_from_query(query)
    digest_path = site_digest_dir / "search" / f"{slug}.json"
    assert not digest_path.exists()

    # Trace must include a rejected publish stage
    trace_file = Path(result.trace_path)
    assert trace_file.exists()
    trace = json.loads(trace_file.read_text())
    publish_stages = [s for s in trace["stages"] if s["name"] == "publish"]
    assert len(publish_stages) == 1
    assert publish_stages[0]["status"] == "rejected"
    assert "Fact-check overall status is FAIL" in publish_stages[0]["error"]