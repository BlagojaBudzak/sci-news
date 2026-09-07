"""
FastAPI backend prototype for the Sci News research engine.

This module exposes the existing engine as a REST API with a simple
in-memory job manager. It intentionally does not include database
persistence, distributed queues, or full rate limiting.

API key protection (optional):
    SCI_NEWS_API_KEY environment variable. If set, requests to
    POST /research and GET /research/{job_id} must include
    an X-API-Key header matching the value.

    NOTE: This is NOT a secret when used from a public browser frontend.
    It is only a basic gate to prevent casual unauthorised access.
    For real anti-bot protection, a proper authentication and rate
    limiting layer must be added later.
"""
from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import engine
from src.crew_setup import get_local_llm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH = 500
MAX_CONCURRENT_RESEARCH = int(os.getenv("MAX_CONCURRENT_RESEARCH", "1"))
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))

_cors_origins_env = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [
    origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()
]
# If the environment variable is missing, allow all origins (development).
# In production, set CORS_ORIGINS explicitly to the GitHub Pages URL.

API_KEY = os.getenv("SCI_NEWS_API_KEY", "")

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
# In-memory job store
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

# Concurrency control (thread-safe, used across worker threads)
_semaphore = threading.Semaphore(MAX_CONCURRENT_RESEARCH)

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Sci News API", version="0.1.0")

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
    job = JOBS[job_id]
    with _semaphore:
        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
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

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/research", response_model=JobCreateResponse, status_code=202)
async def create_research(
    request: ResearchRequest,
    _: None = Depends(verify_api_key),
):
    """Start a new research job in the background."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    job_id = uuid.uuid4().hex
    job = JobInternal(job_id=job_id, query=query)
    JOBS[job_id] = job

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
    if not job:
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