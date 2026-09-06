from src.prefilter import deduplicate_papers

def test_deduplicate_by_doi():
    papers = [
        {"id": "1", "doi": "10.1234/abc", "url": "http://a"},
        {"id": "2", "doi": "10.1234/abc", "url": "http://b"},
    ]
    result = deduplicate_papers(papers)
    assert len(result) == 1
    assert result[0]["id"] == "1"

def test_deduplicate_by_normalized_url():
    papers = [
        {"id": "1", "doi": "", "url": "https://example.com/path"},
        {"id": "2", "doi": "", "url": "https://example.com/path"},
    ]
    result = deduplicate_papers(papers)
    assert len(result) == 1

def test_deduplicate_by_source_id():
    papers = [
        {"id": "openalex:W1", "source_id": "W1", "doi": "", "url": ""},
        {"id": "arxiv:1234", "source_id": "W1", "doi": "", "url": ""},
    ]
    result = deduplicate_papers(papers)
    assert len(result) == 1