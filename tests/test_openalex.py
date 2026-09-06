import pytest
from src.openalex import normalize_openalex_work, reconstruct_abstract, fetch_openalex_papers

SAMPLE_WORK = {
    "id": "https://openalex.org/W123",
    "doi": "https://doi.org/10.1000/xyz",
    "title": "A Test Paper on Electrochemistry",
    "abstract_inverted_index": {
        "This": [0],
        "is": [1],
        "a": [2],
        "test": [3],
    },
    "authorships": [
        {"author": {"display_name": "Alice Example"}},
        {"author": {"display_name": "Bob Sample"}},
    ],
    "publication_date": "2026-09-01",
    "primary_location": {"landing_page_url": "https://example.com/paper"},
}

def test_reconstruct_abstract_from_inverted_index():
    inverted = {"This": [0], "is": [1], "a": [2], "test": [3]}
    assert reconstruct_abstract(inverted) == "This is a test"

def test_reconstruct_abstract_string_passthrough():
    assert reconstruct_abstract("Already plain text") == "Already plain text"

def test_normalize_openalex_work():
    paper = normalize_openalex_work(SAMPLE_WORK)
    assert paper["source"] == "openalex"
    assert paper["title"] == "A Test Paper on Electrochemistry"
    assert paper["abstract"] == "This is a test"
    assert paper["authors"] == ["Alice Example", "Bob Sample"]
    assert paper["doi"] == "https://doi.org/10.1000/xyz"
    assert paper["url"] == "https://example.com/paper"
    assert paper["published"] == "2026-09-01"

def test_fetch_openalex_no_config_returns_empty():
    config = {}  # no openalex block
    assert fetch_openalex_papers("physics", config) == []