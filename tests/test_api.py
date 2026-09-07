"""
Tests for the FastAPI backend using TestClient.
All engine.research calls are mocked; no real LLM/network is used.
"""
from __future__ import annotations

import importlib
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Make imports work when running from repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as api_app
from src.engine import ResearchResult

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    """Return a TestClient for the app, clearing jobs before each test."""
    api_app.JOBS.clear()
    api_app._semaphore = threading.Semaphore(api_app.MAX_CONCURRENT_RESEARCH)
    return TestClient(api_app.app)


def _wait_for_status(client: TestClient, job_id: str, expected_status: str, timeout: float = 5.0) -> dict:
    """Poll the job endpoint until the status matches or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/research/{job_id}")
        if resp.status_code == 200:
            data = resp.json()
            if data["status"] == expected_status:
                return data
        time.sleep(0.05)
    raise AssertionError(f"Job did not reach status {expected_status!r} in time")


def _fake_research_success(*args, **kwargs):
    return ResearchResult(
        job_id=kwargs.get("job_id", "test"),
        query=kwargs.get("query", "test"),
        status="completed",
        digest_entries=[],
        verification_status="PASS",
        trace_path=None,
        summary={},
    )


def _fake_research_failure(*args, **kwargs):
    raise RuntimeError("Simulated research failure")


def _fake_research_blocking(*args, **kwargs):
    # Sleep long enough to ensure the first job holds the semaphore.
    time.sleep(3.0)
    return _fake_research_success(*args, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_research_rejects_empty_query(client):
    resp = client.post("/research", json={"query": ""})
    assert resp.status_code in (400, 422)


def test_create_research_rejects_too_long_query(client):
    long_query = "a" * 501
    resp = client.post("/research", json={"query": long_query})
    assert resp.status_code in (400, 422)


def test_create_research_creates_job(client, monkeypatch):
    monkeypatch.setattr(api_app.engine, "research", _fake_research_blocking)
    resp = client.post("/research", json={"query": "hydrogen storage"})
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"

    # Poll until running or completed
    job_data = _wait_for_status(client, data["job_id"], "running", timeout=3.0)
    assert job_data["status"] == "running"


def test_get_research_unknown_job_returns_404(client):
    resp = client.get("/research/unknown-id")
    assert resp.status_code == 404


def test_background_research_success(client, monkeypatch):
    monkeypatch.setattr(api_app.engine, "research", _fake_research_success)
    resp = client.post("/research", json={"query": "catalysis"})
    job_id = resp.json()["job_id"]

    job_data = _wait_for_status(client, job_id, "completed")
    assert job_data["status"] == "completed"
    assert job_data["verification_status"] == "PASS"
    assert job_data["result"]["verification_status"] == "PASS"


def test_background_research_failure(client, monkeypatch):
    monkeypatch.setattr(api_app.engine, "research", _fake_research_failure)
    resp = client.post("/research", json={"query": "bad query"})
    job_id = resp.json()["job_id"]

    job_data = _wait_for_status(client, job_id, "failed")
    assert job_data["status"] == "failed"
    assert "Simulated research failure" in job_data["error"]


def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("SCI_NEWS_API_KEY", "secret")
    importlib.reload(api_app)
    monkeypatch.setattr(api_app.engine, "research", _fake_research_success)
    api_app.JOBS.clear()
    api_app._semaphore = threading.Semaphore(api_app.MAX_CONCURRENT_RESEARCH)
    client = TestClient(api_app.app)

    resp = client.post("/research", json={"query": "test"})
    assert resp.status_code == 401

    resp = client.get("/research/some-id")
    assert resp.status_code == 401

    resp = client.post(
        "/research",
        json={"query": "test"},
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 202

    monkeypatch.delenv("SCI_NEWS_API_KEY", raising=False)
    importlib.reload(api_app)


def test_api_key_disabled_when_not_configured(client, monkeypatch):
    monkeypatch.delenv("SCI_NEWS_API_KEY", raising=False)
    importlib.reload(api_app)
    monkeypatch.setattr(api_app.engine, "research", _fake_research_success)
    client = TestClient(api_app.app)

    resp = client.post("/research", json={"query": "test"})
    assert resp.status_code == 202


def test_concurrency_limit_prevents_unlimited_jobs(client, monkeypatch):
    # Set max concurrency to 1 and reset semaphore
    monkeypatch.setattr(api_app, "MAX_CONCURRENT_RESEARCH", 1)
    api_app._semaphore = threading.Semaphore(1)

    monkeypatch.setattr(api_app.engine, "research", _fake_research_blocking)

    # Start first job
    resp1 = client.post("/research", json={"query": "first"})
    job1_id = resp1.json()["job_id"]

    # Wait until first job is running
    job1 = _wait_for_status(client, job1_id, "running", timeout=3.0)

    # Start second job while first is running
    resp2 = client.post("/research", json={"query": "second"})
    job2_id = resp2.json()["job_id"]
    job2 = client.get(f"/research/{job2_id}").json()

    # Second job should still be queued because semaphore is taken
    assert job1["status"] == "running"
    assert job2["status"] == "queued"

    # Wait for first to complete and then second should eventually run
    _wait_for_status(client, job1_id, "completed", timeout=6.0)
    _wait_for_status(client, job2_id, "completed", timeout=6.0)