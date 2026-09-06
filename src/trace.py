"""
Per-run observability trace.

The problem this solves: with multiple LLM stages in a row, a silent bad
result (an empty digest, a reviewer that picked a nonexistent id, a writer
call that failed to parse) previously left no record beyond whatever
scrolled past in the console. This module gives every pipeline run a
job id, times each stage, records how many items went in/out and any
failure, and writes it all to data/traces/<category>_<job_id>.json —
so a bad run can be attributed to a specific stage after the fact
instead of disappearing into an opaque "the AI did something" box.

Usage (see main.py):

    trace = PipelineTrace.start(category)

    with trace.stage("fetch") as s:
        papers = fetch_papers_for_category(category)
        s.count_out = len(papers)
        s.count_label = "papers found"

    trace.print_summary()
    trace.save(DATA_TRACE_DIR)

Stages that don't exist yet in the pipeline (deterministic filtering,
fact-checking) simply aren't recorded yet — this module doesn't
fabricate stages that aren't real. Add a `with trace.stage(...)` block
around each new stage as it's built.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class StageResult:
    name: str
    status: str = "pending"  # pending | ok | failed
    count_in: Optional[int] = None
    count_out: Optional[int] = None
    count_label: str = "items"
    duration_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "count_in": self.count_in,
            "count_out": self.count_out,
            "count_label": self.count_label,
            "duration_seconds": round(self.duration_seconds, 2),
            "notes": self.notes,
            "error": self.error,
        }


@dataclass
class PipelineTrace:
    category: str
    job_id: str
    started_at: str
    stages: List[StageResult] = field(default_factory=list)

    @classmethod
    def start(cls, category: str) -> "PipelineTrace":
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(
            category=category,
            job_id=job_id,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )

    @contextmanager
    def stage(self, name: str, count_in: Optional[int] = None) -> Iterator[StageResult]:
        """Times a stage and records ok/failed automatically.

        On an unhandled exception inside the block, the stage is marked
        failed with the exception message and the exception re-raises
        (this module observes, it doesn't swallow errors). If the code
        inside the block sets `result.status = "failed"` itself before
        a plain `return`, that's respected too — see main.py, which
        prefers "print + return" over raising for expected failure
        modes like unparseable LLM output.
        """
        result = StageResult(name=name, count_in=count_in)
        self.stages.append(result)
        t0 = time.perf_counter()
        try:
            yield result
            if result.status == "pending":
                result.status = "ok"
        except Exception as exc:  # noqa: BLE001 - record, then propagate
            result.status = "failed"
            result.error = str(exc)
            raise
        finally:
            result.duration_seconds = time.perf_counter() - t0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "category": self.category,
            "started_at": self.started_at,
            "stages": [s.to_dict() for s in self.stages],
        }

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.category}_{self.job_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def print_summary(self) -> None:
        print(f"\nJOB #{self.job_id}  ({self.category})")
        for s in self.stages:
            print(f"\n{s.name.upper()}")
            if s.status == "failed":
                print(f"  FAILED after {s.duration_seconds:.1f}s: {s.error}")
            elif s.count_out is not None:
                print(f"  {s.count_out} {s.count_label}  ({s.duration_seconds:.1f}s)")
            else:
                print(f"  done  ({s.duration_seconds:.1f}s)")
            for note in s.notes:
                print(f"  - {note}")
