"""
Tests for the research engine's publication gate and trace persistence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import src.engine  # import the module to allow direct attribute patching
from src.engine import research
from src.crew_setup import ReviewerSelection, SelectedPaper, WriterOutput, LLMDraftEntry, Claim
from src.fact_checker import FactCheckReport, ClaimAssessment
from src.trace import PipelineTrace

# ---------------------------------------------------------------------------
# Helpers to build fake pipeline objects
# ---------------------------------------------------------------------------
def make_plan():
    class Plan:
        search_phrases = ["test"]
        positive_keywords = []
        negative_keywords = []
        lookback_days = 7
    return Plan()

def make_papers():
    return [
        {
            "id": "paper1",
            "title": "Test Paper",
            "abstract": "This is a test abstract with enough length to pass.",
            "authors": ["Author"],
            "published": "2026-01-01",
            "source": "openalex",
            "doi": "10.0000/test",
            "url": "http://example.com/paper1",
        }
    ]

def make_reviewer_selection():
    return ReviewerSelection(selected_papers=[SelectedPaper(id="paper1", reason="Interesting")])

def make_writer_output():
    claim = Claim(claim="A test claim", paper_ids=["paper1"])
    entry = LLMDraftEntry(id="paper1", title="Title", paragraph="Paragraph", claims=[claim])
    return WriterOutput(articles=[entry])

def make_fact_check_report(status: str):
    if status == "FAIL":
        assessment = ClaimAssessment(
            claim="A test claim",
            paper_ids=["paper1"],
            verdict="UNSUPPORTED",
            justification="Not in abstract",
        )
    elif status == "PASS":
        assessment = ClaimAssessment(
            claim="A test claim",
            paper_ids=["paper1"],
            verdict="SUPPORTED",
            justification="Supported",
        )
    else:  # PARTIAL
        assessment = ClaimAssessment(
            claim="A test claim",
            paper_ids=["paper1"],
            verdict="NOT_CHECKABLE",
            justification="Missing detail",
        )
    return FactCheckReport(assessments=[assessment], overall_status=status, summary="Test")

# ---------------------------------------------------------------------------
# Fixture for monkeypatching the engine components
# ---------------------------------------------------------------------------
@pytest.fixture
def patch_pipeline(monkeypatch):
    """Replace all external/LLM components with deterministic fakes."""
    # Query interpretation
    monkeypatch.setattr("src.engine.interpret_query", lambda query, llm=None: make_plan())
    monkeypatch.setattr("src.engine.build_openalex_query", lambda plan: "fake_openalex")
    monkeypatch.setattr("src.engine.build_arxiv_query", lambda plan: "fake_arxiv")

    # Fetching
    monkeypatch.setattr("src.engine.fetch_openalex_papers", lambda *a, **k: make_papers())
    monkeypatch.setattr("src.engine.fetch_arxiv_search", lambda *a, **k: [])

    # Prefilter
    class FakePrefilterResult:
        output_count = 1
        candidates = make_papers()
        missing_abstract_count = 0
        low_relevance_count = 0
        filtered_examples = []
        input_count = 1
        duplicate_count = 0

    monkeypatch.setattr("src.engine.deduplicate_papers", lambda papers: papers)
    monkeypatch.setattr("src.engine.filter_papers", lambda papers, cfg: FakePrefilterResult())

    # Reviewer
    class FakeCrew:
        def kickoff(self):
            class Result:
                pydantic = make_reviewer_selection()
                raw = None
            return Result()
    monkeypatch.setattr("src.engine.build_review_crew", lambda *a, **k: FakeCrew())

    # Writer and Fact Checker
    fact_check_status = {"value": "PASS"}
    def fake_run_fact_check(*a, **k):
        return make_fact_check_report(fact_check_status["value"])
    monkeypatch.setattr("src.engine.run_fact_check", fake_run_fact_check)

    class FakeWriterCrew:
        def kickoff(self):
            class Result:
                pydantic = make_writer_output()
                raw = None
            return Result()
    monkeypatch.setattr("src.engine.build_write_crew", lambda *a, **k: FakeWriterCrew())

    return fact_check_status

# ---------------------------------------------------------------------------
# Test 1: Publication gate rejects on FAIL
# ---------------------------------------------------------------------------
def test_fact_check_fail_gate(patch_pipeline, monkeypatch, tmp_path):
    patch_pipeline["value"] = "FAIL"
    # Prevent actual file writes
    monkeypatch.setattr("src.engine.write_digest", lambda *a, **k: (_ for _ in ()).throw(AssertionError("write_digest should not be called")))

    result = research("test query", verbose=False, save_outputs=False)

    assert result.status == "fact_check_failed"
    assert "Fact-check overall status is FAIL" in result.error
    # Digest entries not set
    assert result.digest_entries is None
    # No trace file saved because save_outputs=False
    assert result.trace_path is None

# ---------------------------------------------------------------------------
# Test 2: Unexpected exception saves trace and re-raises
# ---------------------------------------------------------------------------
def test_unexpected_exception_saves_trace(patch_pipeline, monkeypatch, tmp_path):
    # Override fetch to raise an exception
    def boom(*a, **k):
        raise RuntimeError("Simulated fetch failure")
    monkeypatch.setattr("src.engine.fetch_openalex_papers", boom)

    # Redirect DATA_TRACE_DIR to tmp_path by patching the module attribute
    monkeypatch.setattr(src.engine, "DATA_TRACE_DIR", tmp_path)

    # Patch PipelineTrace.save to capture the call and avoid actual file write
    calls = []
    original_save = PipelineTrace.save
    def fake_save(self, directory):
        calls.append(directory)
        return tmp_path / "dummy_trace.json"  # no actual write
    monkeypatch.setattr(PipelineTrace, "save", fake_save)

    with pytest.raises(RuntimeError, match="Simulated fetch failure"):
        research("test query", verbose=False, save_outputs=True)

    # Assert that save was called exactly once (in the except block)
    assert len(calls) == 1
    # The directory passed should be tmp_path because we patched DATA_TRACE_DIR
    assert calls[0] == tmp_path