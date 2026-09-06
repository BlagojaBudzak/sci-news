"""
End-to-end smoke test for main.run_for_category() with the network/LLM
boundary faked out: fetch_papers_for_category, build_review_crew, and
build_write_crew are monkeypatched, but everything else — the trace
wiring, select_papers(), hydrate_digest_output(), write_digest(), the
console summary — runs for real. This is the fastest way to catch a
wiring mistake (wrong import, wrong argument order, a stage that never
gets marked "ok") without waiting on arXiv, ChemRxiv, or a local Ollama
call.

Run with:
    python -m pytest tests/test_pipeline_smoke.py -v
or, without pytest installed:
    python tests/test_pipeline_smoke.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main
from src.crew_setup import (
    LLMDraftEntry,
    ReviewerSelection,
    SelectedPaper,
    WriterOutput,
)

FIXTURE_PAPERS = [
    {
        "id": "2608.00001v1",
        "title": "A Real Paper",
        "abstract": "An abstract about real chemistry.",
        "authors": ["A. Author", "B. Author"],
        "published": "2026-08-30",
        "url": "http://arxiv.org/abs/2608.00001v1",
        "doi": "10.1000/real",
        "source": "arXiv",
    },
    {
        "id": "2608.00002v1",
        "title": "Another Real Paper",
        "abstract": "Another abstract.",
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
    monkeypatch.setattr(main, "fetch_papers_for_category", lambda category: FIXTURE_PAPERS)

    reviewer_selection = ReviewerSelection(selected_papers=[
        SelectedPaper(id="2608.00001v1", reason="Novel result."),
    ])
    monkeypatch.setattr(
        main, "build_review_crew",
        lambda papers, category, llm=None: _FakeCrew(_FakeCrewOutput(reviewer_selection)),
    )

    writer_output = WriterOutput(articles=[
        LLMDraftEntry(id="2608.00001v1", title="Headline", paragraph="Body text."),
    ])
    monkeypatch.setattr(
        main, "build_write_crew",
        lambda papers, category, llm=None: _FakeCrew(_FakeCrewOutput(writer_output)),
    )

    data_digest_dir = tmp_path / "data" / "digests"
    site_digest_dir = tmp_path / "site" / "digests"
    trace_dir = tmp_path / "data" / "traces"
    monkeypatch.setattr(main, "DATA_DIGEST_DIR", data_digest_dir)
    monkeypatch.setattr(main, "SITE_DIGEST_DIR", site_digest_dir)
    monkeypatch.setattr(main, "DATA_TRACE_DIR", trace_dir)
    return trace_dir, site_digest_dir


def test_run_for_category_end_to_end_with_faked_llm(monkeypatch, tmp_path):
    trace_dir, site_digest_dir = _patch_pipeline(monkeypatch, tmp_path)

    main.run_for_category("chemistry")

    # The site JSON got written with the new metadata fields intact.
    payload = json.loads((site_digest_dir / "chemistry.json").read_text())
    assert payload["entries"][0]["authors"] == ["A. Author", "B. Author"]
    assert payload["entries"][0]["source"] == "arXiv"

    # Exactly one trace file was written, with all four stages recorded ok.
    trace_files = list(trace_dir.glob("chemistry_*.json"))
    assert len(trace_files) == 1
    trace = json.loads(trace_files[0].read_text())
    stage_names = [s["name"] for s in trace["stages"]]
    assert stage_names == ["fetch", "reviewer", "writer", "publish"]
    assert all(s["status"] == "ok" for s in trace["stages"])


if __name__ == "__main__":
    import tempfile
    import traceback

    class _MonkeyPatch:
        """Tiny stand-in for pytest's monkeypatch fixture, for running
        this file directly without pytest installed."""

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
            test_run_for_category_end_to_end_with_faked_llm(mp, Path(d))
        print("PASS  test_run_for_category_end_to_end_with_faked_llm")
    except Exception:
        print("FAIL  test_run_for_category_end_to_end_with_faked_llm")
        traceback.print_exc()
    finally:
        mp.undo()
