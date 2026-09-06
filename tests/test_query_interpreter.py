import json

import pytest

from src.query_interpreter import (
    QueryPlan,
    sanitize_plan,
    build_openalex_query,
    build_arxiv_query,
)


def test_sanitize_plan_caps_search_phrases():
    plan = QueryPlan(
        search_phrases=["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"],
        positive_keywords={"a": 1.0},
        negative_keywords={"b": 1.0},
        lookback_days=100,
    )
    sanitized = sanitize_plan(plan)
    assert len(sanitized.search_phrases) <= 10
    assert sanitized.lookback_days == 60


def test_sanitize_plan_deduplicates_phrases():
    plan = QueryPlan(
        search_phrases=["Hydrogen storage", "hydrogen storage", "  fuel cell  "],
        positive_keywords={},
        negative_keywords={},
        lookback_days=7,
    )
    sanitized = sanitize_plan(plan)
    assert sanitized.search_phrases == ["Hydrogen storage", "fuel cell"]


def test_build_openalex_query():
    plan = QueryPlan(
        search_phrases=["hydrogen storage", "fuel cell"],
        positive_keywords={},
        negative_keywords={},
        lookback_days=7,
    )
    query = build_openalex_query(plan)
    assert query == '"hydrogen storage" OR "fuel cell"'


def test_build_arxiv_query():
    plan = QueryPlan(
        search_phrases=["hydrogen storage", "fuel cell"],
        positive_keywords={},
        negative_keywords={},
        lookback_days=7,
    )
    query = build_arxiv_query(plan)
    assert query == 'all:"hydrogen storage" OR all:"fuel cell"'


def test_build_queries_empty_plan():
    plan = QueryPlan(
        search_phrases=[],
        positive_keywords={},
        negative_keywords={},
        lookback_days=7,
    )
    assert build_openalex_query(plan) == ""
    assert build_arxiv_query(plan) == ""


def test_sanitize_plan_removes_generic_negative_keywords():
    plan = QueryPlan(
        search_phrases=["hydrogen storage"],
        positive_keywords={"hydrogen storage": 3.0},
        negative_keywords={
            "review": 2.0,
            "modeling": 1.5,
            "hydrogen storage tank": 2.0,
        },
        lookback_days=7,
    )
    sanitized = sanitize_plan(plan)
    assert "review" not in sanitized.negative_keywords
    assert "modeling" not in sanitized.negative_keywords
    assert "hydrogen storage tank" in sanitized.negative_keywords



# Keep old keyword sanitization tests if present; adapt as needed.