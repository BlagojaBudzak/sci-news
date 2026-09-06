from unittest.mock import patch
from src.openalex import fetch_openalex_papers

@patch("src.openalex.requests.get")
def test_openalex_search_query_override(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"results": []}

    dummy_config = {"openalex": {"search_query": "default_query"}}

    fetch_openalex_papers(
        "test_category",
        dummy_config,
        use_cache=False,
        search_query_override="override_query",
    )

    params = mock_get.call_args.kwargs["params"]
    assert params["search"] == "override_query"


@patch("src.openalex.requests.get")
def test_openalex_fallback_to_config(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"results": []}

    dummy_config = {"openalex": {"search_query": "default_query"}}

    fetch_openalex_papers(
        "test_category",
        dummy_config,
        use_cache=False,
        search_query_override=None,
    )

    params = mock_get.call_args.kwargs["params"]
    assert params["search"] == "default_query"