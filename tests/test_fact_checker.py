"""
Tests for the Fact Checker module.

These tests cover ID validation, verdict aggregation, LLM parsing, and
overall status computation. They do not require a real LLM; the LLM calls
are mocked or bypassed when invalid IDs are detected.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.crew_setup import Claim, LLMDraftEntry, WriterOutput
from src.fact_checker import (
    ClaimAssessment,
    FactCheckReport,
    _compute_overall_status,
    _parse_fact_check_output,
    _validate_claim_ids,
    build_revision_feedback,
    run_fact_check,
)

# ---------------------------------------------------------------------------
# Helper objects
# ---------------------------------------------------------------------------
def make_claim(text: str, ids: list[str]) -> Claim:
    return Claim(claim=text, paper_ids=ids)

def make_writer_output(articles: list[tuple[str, list[Claim]]]) -> WriterOutput:
    entries = [
        LLMDraftEntry(id=id, title=f"Title {id}", paragraph="Para", claims=claims)
        for id, claims in articles
    ]
    return WriterOutput(articles=entries)

@pytest.fixture
def selected_papers():
    return [
        {"id": "paper1", "title": "Paper One", "abstract": "Abstract for paper1.", "authors": [], "published": "", "source": "", "doi": "", "url": "http://example.com/1"},
        {"id": "paper2", "title": "Paper Two", "abstract": "Abstract for paper2.", "authors": [], "published": "", "source": "", "doi": "", "url": "http://example.com/2"},
    ]

# ---------------------------------------------------------------------------
# ID validation
# ---------------------------------------------------------------------------
def test_validate_claim_ids_all_valid():
    claims = [make_claim("Claim 1", ["paper1"]), make_claim("Claim 2", ["paper2"])]
    invalid, valid = _validate_claim_ids(claims, {"paper1", "paper2"})
    assert len(invalid) == 0
    assert len(valid) == 2

def test_validate_claim_ids_some_invalid():
    claims = [make_claim("Claim 1", ["paper1", "fake"]), make_claim("Claim 2", ["paper2"])]
    invalid, valid = _validate_claim_ids(claims, {"paper1", "paper2"})
    assert len(invalid) == 1
    assert invalid[0].verdict == "INVALID_CITATION"
    assert "fake" in invalid[0].justification
    assert len(valid) == 1

def test_validate_claim_ids_all_invalid():
    claims = [make_claim("Claim 1", ["fake1"]), make_claim("Claim 2", ["fake2"])]
    invalid, valid = _validate_claim_ids(claims, {"paper1"})
    assert len(invalid) == 2
    assert len(valid) == 0

def test_validate_claim_ids_no_ids():
    claims = [make_claim("Claim without IDs", [])]
    invalid, valid = _validate_claim_ids(claims, {"paper1"})
    # No IDs means no invalid IDs, but claim is considered valid (will be NOT_CHECKABLE later)
    assert len(invalid) == 0
    assert len(valid) == 1

# ---------------------------------------------------------------------------
# Overall status computation
# ---------------------------------------------------------------------------
def test_compute_overall_status_pass():
    assessments = [ClaimAssessment(claim="c", paper_ids=["p1"], verdict="SUPPORTED", justification="ok")]
    assert _compute_overall_status(assessments) == "PASS"

def test_compute_overall_status_partial():
    assessments = [
        ClaimAssessment(claim="c1", paper_ids=["p1"], verdict="SUPPORTED", justification="ok"),
        ClaimAssessment(claim="c2", paper_ids=["p2"], verdict="NOT_CHECKABLE", justification="missing details"),
    ]
    assert _compute_overall_status(assessments) == "PARTIAL"

def test_compute_overall_status_fail_unsupported():
    assessments = [
        ClaimAssessment(claim="c1", paper_ids=["p1"], verdict="SUPPORTED", justification="ok"),
        ClaimAssessment(claim="c2", paper_ids=["p2"], verdict="UNSUPPORTED", justification="not supported"),
    ]
    assert _compute_overall_status(assessments) == "FAIL"

def test_compute_overall_status_fail_invalid():
    assessments = [ClaimAssessment(claim="c", paper_ids=["fake"], verdict="INVALID_CITATION", justification="bad id")]
    assert _compute_overall_status(assessments) == "FAIL"

def test_compute_overall_status_empty():
    assert _compute_overall_status([]) == "PASS"

# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------
def test_parse_fact_check_output_valid():
    raw = json.dumps({
        "assessments": [
            {"claim": "Claim 1", "paper_ids": ["p1"], "verdict": "SUPPORTED", "justification": "yes"},
            {"claim": "Claim 2", "paper_ids": ["p2"], "verdict": "NOT_CHECKABLE", "justification": "not enough"},
        ],
        "summary": "ok"
    })
    valid_claims = [make_claim("Claim 1", ["p1"]), make_claim("Claim 2", ["p2"])]
    report = _parse_fact_check_output(raw, valid_claims)
    assert len(report.assessments) == 2
    assert report.overall_status == "PARTIAL"

def test_parse_fact_check_output_missing_claim():
    raw = json.dumps({
        "assessments": [
            {"claim": "Claim 1", "paper_ids": ["p1"], "verdict": "SUPPORTED", "justification": "yes"}
        ],
        "summary": "ok"
    })
    valid_claims = [make_claim("Claim 1", ["p1"]), make_claim("Claim 2", ["p2"])]
    report = _parse_fact_check_output(raw, valid_claims)
    assert len(report.assessments) == 2
    # Missing claim marked NOT_CHECKABLE
    missing = [a for a in report.assessments if a.claim == "Claim 2"]
    assert len(missing) == 1 and missing[0].verdict == "NOT_CHECKABLE"
    assert report.overall_status == "PARTIAL"

def test_parse_fact_check_output_invalid_verdict():
    raw = json.dumps({
        "assessments": [
            {"claim": "Claim 1", "paper_ids": ["p1"], "verdict": "MAYBE", "justification": "uncertain"}
        ],
        "summary": "bad"
    })
    valid_claims = [make_claim("Claim 1", ["p1"])]
    report = _parse_fact_check_output(raw, valid_claims)
    assert report.assessments[0].verdict == "NOT_CHECKABLE"

def test_parse_fact_check_output_malformed_json():
    raw = "not json"
    valid_claims = [make_claim("Claim 1", ["p1"])]
    report = _parse_fact_check_output(raw, valid_claims)
    assert report.assessments[0].verdict == "NOT_CHECKABLE"
    assert report.overall_status == "PARTIAL"

# ---------------------------------------------------------------------------
# run_fact_check (with mocked LLM not needed if all invalid or no claims)
# ---------------------------------------------------------------------------
def test_run_fact_check_all_invalid_ids(selected_papers, monkeypatch):
    writer_output = make_writer_output([
        ("paper1", [make_claim("Claim with bad id", ["fake"])])
    ])
    # No LLM should be called; we can check by ensuring that get_local_llm is not invoked.
    monkeypatch.setattr("src.fact_checker.get_local_llm", lambda: None)
    report = run_fact_check(writer_output, selected_papers)
    assert report.overall_status == "FAIL"
    assert len(report.assessments) == 1
    assert report.assessments[0].verdict == "INVALID_CITATION"

def test_run_fact_check_no_claims(selected_papers, monkeypatch):
    writer_output = make_writer_output([("paper1", [])])
    monkeypatch.setattr("src.fact_checker.get_local_llm", lambda: None)
    report = run_fact_check(writer_output, selected_papers)
    assert report.overall_status == "PASS"
    assert len(report.assessments) == 0

# ---------------------------------------------------------------------------
# Revision feedback
# ---------------------------------------------------------------------------
def test_build_revision_feedback():
    report = FactCheckReport(
        assessments=[
            ClaimAssessment(claim="Bad claim", paper_ids=["p1"], verdict="UNSUPPORTED", justification="Not in abstract"),
            ClaimAssessment(claim="Good claim", paper_ids=["p2"], verdict="SUPPORTED", justification="Ok"),
        ],
        overall_status="FAIL",
        summary="One claim unsupported",
    )
    feedback = build_revision_feedback(report)
    assert "Bad claim" in feedback
    assert "UNSUPPORTED" in feedback
    assert "Not in abstract" in feedback
    assert "Good claim" not in feedback  # Only failed claims are included