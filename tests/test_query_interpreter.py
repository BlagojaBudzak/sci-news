import json

from datetime import date

import pytest

from src.query_interpreter import (
    QueryPlan,
    sanitize_plan,
    build_openalex_query,
    build_arxiv_query,
    deterministic_lookback_days,
    interpret_query,
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
    assert sanitized.lookback_days == 100


def test_sanitize_plan_clamps_lookback_to_max():
    plan = QueryPlan(
        search_phrases=["x"],
        positive_keywords={},
        negative_keywords={},
        lookback_days=9999,
    )
    sanitized = sanitize_plan(plan)
    assert sanitized.lookback_days == 365


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


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("perovskite solar cells", 60),
        ("recent advances in catalysis", 60),
        ("papers from the last week", 7),
        ("past 2 weeks on batteries", 14),
        ("research from the past month", 30),
        ("last 3 months of research", 90),
        ("past year of CRISPR research", 365),
        ("research today", 1),
    ],
)
def test_deterministic_lookback_common_phrases(query, expected):
    assert deterministic_lookback_days(query, today=date(2026, 10, 12)) == expected


def test_deterministic_lookback_calendar_periods_and_cap():
    today = date(2026, 10, 12)
    assert deterministic_lookback_days("this year in chemistry", today) == 284
    assert deterministic_lookback_days("since 2026", today) == 284
    assert deterministic_lookback_days("since 2020", today) == 365
    assert deterministic_lookback_days("since 2035", today) == 60


def test_interpret_query_ignores_model_lookback_days(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "search_phrases": ["perovskite solar cells"],
                            "positive_keywords": {"perovskite": 2.0},
                            "negative_keywords": {},
                            "lookback_days": 7,
                        })
                    }
                }]
            }

    monkeypatch.setattr("src.query_interpreter.requests.post", lambda *args, **kwargs: FakeResponse())
    assert interpret_query("perovskite solar cells").lookback_days == 60
    assert interpret_query("perovskite solar cells from the past week").lookback_days == 7
