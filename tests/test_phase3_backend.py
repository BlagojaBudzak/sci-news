"""
Tests for Phase 3 backend hardening: SQLite persistence, rate limiting,
duplicate suppression, and CORS configuration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app, JOBS, DB_PATH, _init_db, _save_job_to_db, _load_job_from_db

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def client(monkeypatch, tmp_path):
    """Return a TestClient with isolated DB and rate limit dictionaries."""
    # Use temporary DB file
    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "test_jobs.db")
    _init_db()  # Recreate the table in the temp DB

    # Clear in-memory structures
    JOBS.clear()
    monkeypatch.setattr(app_module, "_request_log", {})
    monkeypatch.setattr(app_module, "_hour_log", {})
    monkeypatch.setattr(app_module, "_recent_queries", {})

    # Set high limits so tests can submit jobs
    monkeypatch.setattr(app_module, "RATE_LIMIT_PER_MINUTE", 1000)
    monkeypatch.setattr(app_module, "RATE_LIMIT_PER_HOUR", 10000)
    monkeypatch.setattr(app_module, "MAX_QUEUED_JOBS", 1000)
    monkeypatch.setattr(app_module, "DUPLICATE_SUPPRESSION_WINDOW_SECONDS", 300)

    # Mock the research engine to avoid actual LLM calls
    def fake_research(query, **kwargs):
        class FakeResult:
            def model_dump(self):
                return {"query": query, "status": "completed"}
            verification_status = "PASS"
        return FakeResult()
    monkeypatch.setattr(app_module.engine, "research", fake_research)
    monkeypatch.setattr(app_module, "get_local_llm", lambda: None)

    with TestClient(app) as c:
        yield c

# ---------------------------------------------------------------------------
# SQLite persistence tests
# ---------------------------------------------------------------------------
def test_sqlite_job_creation_and_retrieval(client, monkeypatch, tmp_path):
    # Submit a job
    response = client.post("/research", json={"query": "test query"})
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    # Job should be persisted in SQLite immediately.
    loaded = _load_job_from_db(job_id)
    assert loaded is not None
    assert loaded.query == "test query"

    # GET endpoint should retrieve the job (from memory or DB)
    get_response = client.get(f"/research/{job_id}")
    assert get_response.status_code == 200
    assert get_response.json()["job_id"] == job_id

def test_sqlite_job_status_update(monkeypatch, tmp_path):
    # Use direct functions, no HTTP layer needed
    from app import JobInternal
    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "test_jobs.db")
    _init_db()
    job = JobInternal(job_id="123", query="q")
    _save_job_to_db(job)
    # Update status
    job.status = "completed"
    job.result = {"some": "data"}
    _save_job_to_db(job)
    loaded = _load_job_from_db("123")
    assert loaded.status == "completed"
    assert loaded.result == {"some": "data"}

def test_sqlite_job_persists_across_restart(monkeypatch, tmp_path):
    from app import JobInternal
    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "test_jobs.db")
    _init_db()
    job = JobInternal(job_id="abc", query="persistent")
    _save_job_to_db(job)

    # Simulate restart: clear JOBS, re-init DB (already exists)
    JOBS.clear()
    loaded = _load_job_from_db("abc")
    assert loaded is not None
    assert loaded.query == "persistent"

# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------
def test_rate_limit_per_minute(client, monkeypatch):
    # Lower per-minute limit to 2
    monkeypatch.setattr(app_module, "RATE_LIMIT_PER_MINUTE", 2)
    # Clear logs
    monkeypatch.setattr(app_module, "_request_log", {})
    monkeypatch.setattr(app_module, "_hour_log", {})

    # First two requests should succeed
    r1 = client.post("/research", json={"query": "q1"})
    assert r1.status_code == 202
    r2 = client.post("/research", json={"query": "q2"})
    assert r2.status_code == 202

    # Third request should be rejected (429)
    r3 = client.post("/research", json={"query": "q3"})
    assert r3.status_code == 429

def test_duplicate_query_suppression(client, monkeypatch):
    # Set duplicate window to 60 seconds
    monkeypatch.setattr(app_module, "DUPLICATE_SUPPRESSION_WINDOW_SECONDS", 60)
    monkeypatch.setattr(app_module, "_recent_queries", {})

    # First query accepted
    r1 = client.post("/research", json={"query": "duplicate"})
    assert r1.status_code == 202

    # Same query immediately after should be rejected
    r2 = client.post("/research", json={"query": "duplicate"})
    assert r2.status_code == 429

# ---------------------------------------------------------------------------
# CORS tests
# ---------------------------------------------------------------------------
def test_cors_default_development(client):
    # Default is development (from app import)
    assert app_module.CORS_ORIGINS == ["*"]

def test_cors_production_origin(monkeypatch):
    # Simulate production environment
    monkeypatch.setenv("ENVIRONMENT", "production")
    import importlib
    importlib.reload(app_module)
    # Now CORS_ORIGINS should be the GitHub Pages origin
    assert app_module.CORS_ORIGINS == ["https://BLAGOJABUDZAK.github.io"]
    # Clean up: revert reload by re-importing original (or set env back and reload)
    monkeypatch.setenv("ENVIRONMENT", "development")
    importlib.reload(app_module)
    assert app_module.CORS_ORIGINS == ["*"]