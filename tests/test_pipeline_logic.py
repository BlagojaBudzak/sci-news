"""
Regression tests for the pure-Python parts of the pipeline — the parts
that don't require a live LLM call, arXiv/ChemRxiv access, or Ollama.
These exist so a schema or matching-logic change can be verified in
seconds instead of burning a full pipeline run to find out it broke.

Run with:
    python -m pytest tests/test_pipeline_logic.py -v
or, without pytest installed:
    python tests/test_pipeline_logic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crew_setup import (
    BlogEntry,
    LLMDraftEntry,
    ReviewerSelection,
    SelectedPaper,
    WriterOutput,
    hydrate_digest_output,
    select_papers,
)
from src.trace import PipelineTrace

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


def test_select_papers_matches_real_ids_and_attaches_reason():
    selection = ReviewerSelection(selected_papers=[
        SelectedPaper(id="2608.00001v1", reason="Novel result."),
    ])
    selected = select_papers(FIXTURE_PAPERS, selection)
    assert len(selected) == 1
    assert selected[0]["id"] == "2608.00001v1"
    assert selected[0]["reason"] == "Novel result."


def test_select_papers_drops_hallucinated_id():
    """Core anti-hallucination guarantee: an id the reviewer invented
    must never survive into the published digest."""
    selection = ReviewerSelection(selected_papers=[
        SelectedPaper(id="2608.00001v1", reason="Real pick."),
        SelectedPaper(id="9999.99999v1", reason="Invented id."),
    ])
    selected = select_papers(FIXTURE_PAPERS, selection)
    assert len(selected) == 1
    assert all(p["id"] != "9999.99999v1" for p in selected)


def test_hydrate_digest_output_carries_metadata_through():
    """Regression test for the schema fix: authors/published/source/doi
    must survive from the fetched paper dict into the final BlogEntry,
    not just id/reason/title/paragraph/link."""
    selection = ReviewerSelection(selected_papers=[
        SelectedPaper(id="2608.00001v1", reason="Novel result."),
    ])
    selected = select_papers(FIXTURE_PAPERS, selection)

    writer_output = WriterOutput(articles=[
        LLMDraftEntry(id="2608.00001v1", title="Headline", paragraph="Body text."),
    ])

    digest = hydrate_digest_output(writer_output, selected)
    assert len(digest.entries) == 1

    entry: BlogEntry = digest.entries[0]
    assert entry.title == "Headline"
    assert entry.link == "http://arxiv.org/abs/2608.00001v1"
    assert entry.authors == ["A. Author", "B. Author"]
    assert entry.published == "2026-08-30"
    assert entry.source == "arXiv"
    assert entry.doi == "10.1000/real"


def test_hydrate_digest_output_drops_unmatched_draft_id():
    selection = ReviewerSelection(selected_papers=[
        SelectedPaper(id="2608.00001v1", reason="Novel result."),
    ])
    selected = select_papers(FIXTURE_PAPERS, selection)

    writer_output = WriterOutput(articles=[
        LLMDraftEntry(id="2608.00001v1", title="Headline", paragraph="Body text."),
        LLMDraftEntry(id="0000.00000v1", title="Ghost", paragraph="Should be dropped."),
    ])

    digest = hydrate_digest_output(writer_output, selected)
    assert len(digest.entries) == 1
    assert digest.entries[0].id == "2608.00001v1"


def test_pipeline_trace_records_success():
    trace = PipelineTrace.start("chemistry")
    with trace.stage("fetch") as s:
        s.count_out = 12
        s.count_label = "papers found"

    assert trace.stages[0].status == "ok"
    assert trace.stages[0].count_out == 12
    d = trace.to_dict()
    assert d["category"] == "chemistry"
    assert d["stages"][0]["name"] == "fetch"


def test_pipeline_trace_records_exception_as_failed():
    trace = PipelineTrace.start("physics")
    try:
        with trace.stage("reviewer"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert trace.stages[0].status == "failed"
    assert "boom" in trace.stages[0].error


def test_pipeline_trace_save_writes_json(tmp_path):
    trace = PipelineTrace.start("seismology")
    with trace.stage("fetch") as s:
        s.count_out = 3
    path = trace.save(tmp_path)
    assert path.exists()
    assert path.read_text().strip().startswith("{")


if __name__ == "__main__":
    # Allow running as a plain script if pytest isn't installed.
    import inspect
    import tempfile
    import traceback

    tests = [
        (name, fn) for name, fn in list(globals().items())
        if name.startswith("test_") and inspect.isfunction(fn)
    ]
    passed = 0
    for name, fn in tests:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
