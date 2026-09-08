"""
FastAPI backend for the Sci News research engine.

This module exposes the research engine as a REST API with a persistent
SQLite job store, basic rate limiting, and optional API key protection.

Job persistence:
    Jobs are stored in `data/jobs.db` (SQLite). The in-memory `JOBS`
    dictionary is kept synchronised so existing code and tests continue to
    work. On a restart, jobs that were completed or failed are still
    available from the database.

Rate limiting:
    Simple per-IP request limits (minute and hour), a maximum number of
    queued jobs, and duplicate query suppression within a short window.
    These are in-memory and suitable for a single-machine deployment.

API key:
    If SCI_NEWS_API_KEY is set, requests must include a matching
    X-API-Key header. This is only a basic gate; it is NOT a secret when
    used from a public frontend.

CORS:
    In development (ENVIRONMENT unset or "development"), CORS_ORIGINS
    defaults to "*" (all origins). In production (ENVIRONMENT=production),
    it defaults to the GitHub Pages origin. CORS_ORIGINS can always be
    overridden with the environment variable.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import engine
from src.crew_setup import get_local_llm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH = 500
MAX_CONCURRENT_RESEARCH = int(os.getenv("MAX_CONCURRENT_RESEARCH", "1"))
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))

# Concurrency control (thread-safe, used across worker threads)
_semaphore = threading.Semaphore(MAX_CONCURRENT_RESEARCH)

# Rate limiting defaults (high enough for local development and tests)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "1000"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "10000"))
MAX_QUEUED_JOBS = int(os.getenv("MAX_QUEUED_JOBS", "1000"))
DUPLICATE_SUPPRESSION_WINDOW_SECONDS = int(
    os.getenv("DUPLICATE_SUPPRESSION_WINDOW_SECONDS", "300")
)

# CORS: production origin unless overridden
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
if ENVIRONMENT == "production":
    DEFAULT_CORS_ORIGINS = ["https://BLAGOJABUDZAK.github.io"]
else:
    DEFAULT_CORS_ORIGINS = ["*"]

_cors_origins_env = os.getenv("CORS_ORIGINS", "")
if _cors_origins_env:
    CORS_ORIGINS = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]
else:
    CORS_ORIGINS = DEFAULT_CORS_ORIGINS

API_KEY = os.getenv("SCI_NEWS_API_KEY", "")

# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------
DB_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DB_DIR / "jobs.db"
DB_DIR.mkdir(parents=True, exist_ok=True)

# Global lock for SQLite operations (sqlite3 connections are not thread-safe
# across threads by default; we use a global lock for simplicity).
_db_lock = threading.Lock()

def _get_db_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db() -> None:
    """Create the jobs table if it does not already exist."""
    with _db_lock, contextlib.closing(_get_db_connection()) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                verification_status TEXT,
                result TEXT,
                error TEXT
            )
        """)
        conn.commit()

_init_db()

# ---------------------------------------------------------------------------
# Helper functions for job persistence
# ---------------------------------------------------------------------------
def _serialize_datetime(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None

def _deserialize_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s)

def _job_to_row(job: "JobInternal") -> Dict[str, Any]:
    """Convert a JobInternal object to a dict suitable for SQLite."""
    return {
        "job_id": job.job_id,
        "query": job.query,
        "status": job.status,
        "created_at": _serialize_datetime(job.created_at),
        "started_at": _serialize_datetime(job.started_at),
        "completed_at": _serialize_datetime(job.completed_at),
        "verification_status": job.verification_status,
        "result": json.dumps(job.result) if job.result is not None else None,
        "error": job.error,
    }

def _row_to_job(row: sqlite3.Row) -> "JobInternal":
    """Convert a SQLite row to a JobInternal object."""
    job = JobInternal(job_id=row["job_id"], query=row["query"])
    job.status = row["status"]
    job.created_at = _deserialize_datetime(row["created_at"]) or job.created_at
    job.started_at = _deserialize_datetime(row["started_at"])
    job.completed_at = _deserialize_datetime(row["completed_at"])
    job.verification_status = row["verification_status"]
    if row["result"]:
        job.result = json.loads(row["result"])
    job.error = row["error"]
    return job

def _save_job_to_db(job: "JobInternal") -> None:
    """Insert or replace a job record in SQLite."""
    data = _job_to_row(job)
    with _db_lock, contextlib.closing(_get_db_connection()) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO jobs
            (job_id, query, status, created_at, started_at, completed_at,
             verification_status, result, error)
            VALUES
            (:job_id, :query, :status, :created_at, :started_at, :completed_at,
             :verification_status, :result, :error)
        """, data)
        conn.commit()

def _load_job_from_db(job_id: str) -> Optional["JobInternal"]:
    """Fetch a job from SQLite, or None if not found."""
    with _db_lock, contextlib.closing(_get_db_connection()) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        return _row_to_job(row)

# ---------------------------------------------------------------------------
# In-memory job store (kept synchronised)
# ---------------------------------------------------------------------------
class JobInternal:
    def __init__(self, job_id: str, query: str):
        self.job_id = job_id
        self.query = query
        self.status = "queued"
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now(timezone.utc)
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.verification_status: Optional[str] = None

JOBS: Dict[str, JobInternal] = {}

# ---------------------------------------------------------------------------
# Rate limiting (in-memory)
# ---------------------------------------------------------------------------
_request_log: Dict[str, list[float]] = {}   # IP -> list of timestamps (minute)
_hour_log: Dict[str, list[float]] = {}      # IP -> list of timestamps (hour)
_recent_queries: Dict[str, Dict[str, float]] = {}  # IP -> {query_hash: last_timestamp}
_lock = threading.Lock()

def _cleanup_old_entries(log: Dict[str, list[float]], window: float) -> None:
    now = time.time()
    for ip in list(log.keys()):
        log[ip] = [t for t in log[ip] if now - t < window]
        if not log[ip]:
            del log[ip]

def _check_rate_limit(ip: str) -> None:
    """Raise HTTPException if IP exceeded rate limits or queue is full."""
    now = time.time()
    with _lock:
        # Clean up old entries
        _cleanup_old_entries(_request_log, 60)
        _cleanup_old_entries(_hour_log, 3600)

        minute_count = len(_request_log.get(ip, []))
        hour_count = len(_hour_log.get(ip, []))
        queued_count = sum(1 for j in JOBS.values() if j.status == "queued")

        if minute_count >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Rate limit exceeded (per minute)")
        if hour_count >= RATE_LIMIT_PER_HOUR:
            raise HTTPException(status_code=429, detail="Rate limit exceeded (per hour)")
        if queued_count >= MAX_QUEUED_JOBS:
            raise HTTPException(status_code=429, detail="Too many queued jobs")

        # Record this request
        _request_log.setdefault(ip, []).append(now)
        _hour_log.setdefault(ip, []).append(now)

def _check_duplicate_query(ip: str, query: str) -> None:
    """Raise HTTPException if the same query was submitted recently by this IP."""
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
    now = time.time()
    with _lock:
        ip_queries = _recent_queries.setdefault(ip, {})
        last_time = ip_queries.get(query_hash)
        if last_time is not None and (now - last_time) < DUPLICATE_SUPPRESSION_WINDOW_SECONDS:
            raise HTTPException(status_code=429, detail="Duplicate query; please wait before retrying")
        ip_queries[query_hash] = now
        # Clean up old query records for this IP
        ip_queries = {h: t for h, t in ip_queries.items() if now - t < DUPLICATE_SUPPRESSION_WINDOW_SECONDS}
        _recent_queries[ip] = ip_queries

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)

class JobCreateResponse(BaseModel):
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    query: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    verification_status: Optional[str] = None

class HealthResponse(BaseModel):
    status: str

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Sci News API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key"],
)

# ---------------------------------------------------------------------------
# API key dependency
# ---------------------------------------------------------------------------
def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    If SCI_NEWS_API_KEY is set, require a matching X-API-Key header.
    Otherwise, authentication is disabled.
    """
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
            )

# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------
def _run_research_sync(job_id: str, query: str) -> None:
    """
    Synchronous wrapper that runs the research engine under the semaphore.
    This function is executed in its own daemon thread.
    """
    job = JOBS.get(job_id)
    if job is None:
        return   # Job may have been purged; do nothing.
    with _semaphore:
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        _save_job_to_db(job)
        try:
            llm = get_local_llm()
            result = engine.research(
                query=query,
                llm=llm,
                verbose=False,
                save_outputs=True,
                job_id=job_id,
            )
            job.result = result.model_dump()
            job.verification_status = result.verification_status
            job.status = "completed"
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            logger.exception("Research job failed")
        finally:
            job.completed_at = datetime.now(timezone.utc)
            _save_job_to_db(job)

def _mark_job_timeout(job_id: str) -> None:
    """
    Mark a job as failed if it is still running after the timeout.
    This does not cancel the underlying thread, but makes the job
    visible as failed instead of staying in 'running' indefinitely.
    """
    job = JOBS.get(job_id)
    if job and job.status == "running":
        job.status = "failed"
        job.error = f"Research job exceeded {JOB_TIMEOUT_SECONDS}s timeout"
        job.completed_at = datetime.now(timezone.utc)
        _save_job_to_db(job)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/research", response_model=JobCreateResponse, status_code=202)
async def create_research(
    request: ResearchRequest,
    req: Request,
    _: None = Depends(verify_api_key),
):
    """Start a new research job in the background."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    client_ip = req.client.host if req.client else "unknown"

    # Rate limiting and duplicate suppression
    _check_rate_limit(client_ip)
    _check_duplicate_query(client_ip, query)

    job_id = uuid.uuid4().hex
    job = JobInternal(job_id=job_id, query=query)
    JOBS[job_id] = job
    _save_job_to_db(job)

    # Start research in a daemon thread
    thread = threading.Thread(target=_run_research_sync, args=(job_id, query), daemon=True)
    thread.start()

    # Set a timeout watchdog
    timer = threading.Timer(JOB_TIMEOUT_SECONDS, _mark_job_timeout, args=(job_id,))
    timer.daemon = True
    timer.start()

    return JobCreateResponse(job_id=job_id, status="queued")

@app.get("/research/{job_id}", response_model=JobStatusResponse)
async def get_research_status(
    job_id: str,
    _: None = Depends(verify_api_key),
):
    """Return the current status and result of a research job."""
    job = JOBS.get(job_id)
    if job is None:
        # Try loading from SQLite (for jobs created before a restart)
        job = _load_job_from_db(job_id)
        if job is not None:
            # Cache it in memory for future requests
            JOBS[job_id] = job
        else:
            raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.job_id,
        query=job.query,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=job.result,
        error=job.error,
        verification_status=job.verification_status,
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check."""
    return HealthResponse(status="ok")