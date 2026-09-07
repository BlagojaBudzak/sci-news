"""
End-to-end smoke test for the research engine with network/LLM faked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import engine
from src.crew_setup import (
    Claim,
    LLMDraftEntry,
    ReviewerSelection,
    SelectedPaper,
    WriterOutput,
)
from src.fact_checker import ClaimAssessment, FactCheckReport

FIXTURE_PAPERS = [
    {
        "id": "2608.00001v1",
        "title": "A Real Paper",
        "abstract": ("This paper presents a detailed study of catalytic mechanisms in "
                     "organometallic chemistry. We combine experimental and computational "
                     "methods to reveal new insights into reaction pathways and selectivity."),
        "authors": ["A. Author", "B. Author"],
        "published": "2026-08-30",
        "url": "http://arxiv.org/abs/2608.00001v1",
        "doi": "10.1000/real",
        "source": "arXiv",
    },
    {
        "id": "2608.00002v1",
        "title": "Another Real Paper",
        "abstract": ("Electrochemical synthesis of novel battery materials is explored "
                     "using advanced in situ spectroscopy. The findings demonstrate "
                     "significant improvements in energy density and cycling stability."),
        "authors": ["C. Author"],
        "published": "2026-08-31",
        "url": "http://arxiv.org/abs/2608.00002v1",
        "doi": "",
        "source": "arXiv",
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


def _patch_pipeline(monkeypatch, tmp_path):
    # Mock query interpreter
    monkeypatch.setattr(
        engine,
        "interpret_query",
        lambda query, llm=None: type("Plan", (), {
            "search_phrases": ["catalysis"],
            "positive_keywords": {"catalysis": 1.0},
            "negative_keywords": {},
            "lookback_days": 7,
            "openalex_query": "catalysis",
            "arxiv_query": "cat:physics.chem-ph",
        })(),
    )
    monkeypatch.setattr(engine, "build_openalex_query", lambda plan: "catalysis")
    monkeypatch.setattr(engine, "build_arxiv_query", lambda plan: "cat:physics.chem-ph")

    # Mock fetching functions
    monkeypatch.setattr(engine, "fetch_openalex_papers", lambda *args, **kwargs: FIXTURE_PAPERS)
    monkeypatch.setattr(engine, "fetch_arxiv_search", lambda query, lookback_days: [])

    # Mock Reviewer
    reviewer_selection = ReviewerSelection(selected_papers=[
        SelectedPaper(id="2608.00001v1", reason="Novel result."),
    ])
    monkeypatch.setattr(
        engine, "build_review_crew",
        lambda papers, category, llm=None: _FakeCrew(_FakeCrewOutput(reviewer_selection)),
    )

    # Mock Writer
    writer_output = WriterOutput(articles=[
        LLMDraftEntry(
            id="2608.00001v1",
            title="Headline",
            paragraph="Body text.",
            claims=[
                Claim(
                    claim="The paper presents a detailed study of catalytic mechanisms.",
                    paper_ids=["2608.00001v1"],
                )
            ],
        ),
    ])
    monkeypatch.setattr(
        engine, "build_write_crew",
        lambda papers, category, llm=None, feedback=None: _FakeCrew(_FakeCrewOutput(writer_output)),
    )

    # Mock Fact Checker
    monkeypatch.setattr(
        engine,
        "run_fact_check",
        lambda wo, sp, llm=None: FactCheckReport(
            assessments=[
                ClaimAssessment(
                    claim="The paper presents a detailed study of catalytic mechanisms.",
                    paper_ids=["2608.00001v1"],
                    verdict="SUPPORTED",
                    justification="Abstract supports this claim.",
                )
            ],
            overall_status="PASS",
            summary="All claims supported.",
        ),
    )

    # Redirect output directories
    data_digest_dir = tmp_path / "data" / "digests"
    site_digest_dir = tmp_path / "site" / "digests"
    trace_dir = tmp_path / "data" / "traces"
    monkeypatch.setattr(engine, "DATA_DIGEST_DIR", data_digest_dir)
    monkeypatch.setattr(engine, "SITE_DIGEST_DIR", site_digest_dir)
    monkeypatch.setattr(engine, "DATA_TRACE_DIR", trace_dir)
    return trace_dir, site_digest_dir


def test_research_end_to_end_with_faked_llm(monkeypatch, tmp_path):
    trace_dir, site_digest_dir = _patch_pipeline(monkeypatch, tmp_path)

    result = engine.research("catalysis", verbose=False, save_outputs=True)

    # Verify result
    assert result.status == "completed"
    assert result.verification_status == "PASS"
    assert result.digest_entries is not None
    assert len(result.digest_entries) == 1

    # The site JSON got written with metadata and verification_status
    payload = json.loads((site_digest_dir / "search" / "catalysis.json").read_text())
    assert payload["entries"][0]["authors"] == ["A. Author", "B. Author"]
    assert payload["entries"][0]["source"] == "arXiv"
    assert payload["entries"][0]["verification_status"] == "PASS"

    # Trace file exists using result.trace_path
    assert result.trace_path is not None
    trace_file = Path(result.trace_path)
    assert trace_file.exists()
    trace = json.loads(trace_file.read_text())
    stage_names = [s["name"] for s in trace["stages"]]
    assert "fact_checker" in stage_names
    assert all(s["status"] == "ok" for s in trace["stages"])


if __name__ == "__main__":
    import tempfile
    import traceback

    class _MonkeyPatch:
        def __init__(self):
            self._restores = []

        def setattr(self, obj, name, value):
            self._restores.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, old in reversed(self._restores):
                setattr(obj, name, old)

    mp = _MonkeyPatch()
    try:
        with tempfile.TemporaryDirectory() as d:
            test_research_end_to_end_with_faked_llm(mp, Path(d))
        print("PASS  test_research_end_to_end_with_faked_llm")
    except Exception:
        print("FAIL  test_research_end_to_end_with_faked_llm")
        traceback.print_exc()
    finally:
        mp.undo()