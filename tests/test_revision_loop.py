"""
Test that the revision loop works: first fact-check PARTIAL triggers writer revision,
second fact-check PASS allows publishing.
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
                     "organometallic chemistry, combining experimental and computational "
                     "methods to reveal new insights into reaction pathways and selectivity."),
        "authors": ["A. Author"],
        "published": "2026-08-30",
        "url": "http://arxiv.org/abs/2608.00001v1",
        "doi": "10.1000/real",
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


def _patch_for_revision_test(monkeypatch, tmp_path):
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

    # Mock fetching
    monkeypatch.setattr(engine, "fetch_openalex_papers", lambda *args, **kwargs: FIXTURE_PAPERS)
    monkeypatch.setattr(engine, "fetch_arxiv_search", lambda query, lookback_days: [])

    # Mock Reviewer
    monkeypatch.setattr(
        engine, "build_review_crew",
        lambda papers, category, llm=None: _FakeCrew(_FakeCrewOutput(
            ReviewerSelection(selected_papers=[SelectedPaper(id="2608.00001v1", reason="Novel")])
        )),
    )

    # Mock Writer: capture feedback argument
    writer_calls = []
    def fake_build_write_crew(papers, category, llm=None, feedback=None):
        writer_calls.append(feedback)
        return _FakeCrew(_FakeCrewOutput(
            WriterOutput(articles=[
                LLMDraftEntry(
                    id="2608.00001v1",
                    title="Headline",
                    paragraph="Body.",
                    claims=[Claim(claim="A claim", paper_ids=["2608.00001v1"])],
                )
            ])
        ))
    monkeypatch.setattr(engine, "build_write_crew", fake_build_write_crew)

    # Mock Fact Checker: first PARTIAL, then PASS
    fact_check_calls = []
    def fake_run_fact_check(writer_output, selected_papers, llm=None):
        fact_check_calls.append(1)
        if len(fact_check_calls) == 1:
            return FactCheckReport(
                assessments=[
                    ClaimAssessment(
                        claim="A claim",
                        paper_ids=["2608.00001v1"],
                        verdict="PARTIALLY_SUPPORTED",
                        justification="Missing detail about method",
                    )
                ],
                overall_status="PARTIAL",
                summary="One claim partially supported",
            )
        else:
            return FactCheckReport(
                assessments=[
                    ClaimAssessment(
                        claim="A claim",
                        paper_ids=["2608.00001v1"],
                        verdict="SUPPORTED",
                        justification="Now supported",
                    )
                ],
                overall_status="PASS",
                summary="All good",
            )
    monkeypatch.setattr(engine, "run_fact_check", fake_run_fact_check)

    # Redirect directories
    data_digest_dir = tmp_path / "data" / "digests"
    site_digest_dir = tmp_path / "site" / "digests"
    trace_dir = tmp_path / "data" / "traces"
    monkeypatch.setattr(engine, "DATA_DIGEST_DIR", data_digest_dir)
    monkeypatch.setattr(engine, "SITE_DIGEST_DIR", site_digest_dir)
    monkeypatch.setattr(engine, "DATA_TRACE_DIR", trace_dir)
    return writer_calls, fact_check_calls, trace_dir, site_digest_dir


def test_revision_loop_first_partial_then_pass(monkeypatch, tmp_path):
    writer_calls, fact_check_calls, trace_dir, site_digest_dir = _patch_for_revision_test(monkeypatch, tmp_path)

    result = engine.research("catalysis", verbose=False, save_outputs=True)

    # Assert writer was called twice
    assert len(writer_calls) == 2
    # First call: feedback None
    assert writer_calls[0] is None
    # Second call: feedback should contain the original claim and verdict reason
    feedback_second = writer_calls[1]
    assert feedback_second is not None
    assert "A claim" in feedback_second
    assert "PARTIALLY_SUPPORTED" in feedback_second
    assert "Missing detail about method" in feedback_second

    # Fact checker called twice
    assert len(fact_check_calls) == 2

    # Final result PASS
    assert result.verification_status == "PASS"
    assert result.status == "completed"

    # Trace contains revision note
    assert result.trace_path is not None
    trace_file = Path(result.trace_path)
    assert trace_file.exists()
    trace = json.loads(trace_file.read_text())
    writer_stage = next(s for s in trace["stages"] if s["name"] == "writer")
    assert "revision 1 needed: PARTIAL" in writer_stage["notes"]


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
            test_revision_loop_first_partial_then_pass(mp, Path(d))
        print("PASS  test_revision_loop_first_partial_then_pass")
    except Exception:
        print("FAIL  test_revision_loop_first_partial_then_pass")
        traceback.print_exc()
    finally:
        mp.undo()