"""
Query interpreter for dynamic user searches.

This module takes a free-form scientific query and asks the local LLM to
produce a structured QueryPlan. The plan contains:

    - search phrases (exact phrases to send to OpenAlex / arXiv)
    - positive/negative prefilter keywords with weights
    - a lookback window in days

The raw LLM plan is then sanitized by deterministic Python rules before it
is returned to the caller. Python also builds the actual source-specific
search strings from the plan.

No changes are made to the existing category-based pipeline.
"""

from __future__ import annotations

import json
import re
from typing import Any

import requests
from pydantic import BaseModel, Field

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE


# ---------------------------------------------------------------------------
# Pydantic output model
# ---------------------------------------------------------------------------
class QueryPlan(BaseModel):
    """Structured search plan proposed by the LLM."""

    search_phrases: list[str] = Field(
        default_factory=list,
        description="Exact phrases that best describe the user's research topic. ",
    )
    positive_keywords: dict[str, float] = Field(
        default_factory=dict,
        description="Map of positive prefilter keywords to relevance weights (0.5-3.0).",
    )
    negative_keywords: dict[str, float] = Field(
        default_factory=dict,
        description="Map of negative prefilter keywords to relevance weights (0.5-3.0).",
    )
    lookback_days: int = Field(
        default=7,
        description="How far back to search, in days (1-60).",
    )


# ---------------------------------------------------------------------------
# Sanitization constants
# ---------------------------------------------------------------------------
MAX_SEARCH_PHRASES = 10
MAX_PHRASE_LENGTH = 200
MAX_POSITIVE_KEYWORDS = 15
MAX_NEGATIVE_KEYWORDS = 10
MIN_WEIGHT = 0.5
MAX_WEIGHT = 3.0
MIN_LOOKBACK_DAYS = 1
MAX_LOOKBACK_DAYS = 60
GENERIC_NEGATIVE_KEYWORDS = {
    "review",
    "tutorial",
    "survey",
    "methodology",
    "analysis",
    "modeling",
    "modelling",
    "theoretical",
    "simulation",
    "study",
    "paper",
    "article",
    "research",
    "approach",
    "method",
    "framework",
    "system",
    "application",
    "case study",
    "case studies",
    "overview",
    "perspective",
    "opinion",
}

def _clean_phrase(phrase: str) -> str:
    """Normalize a search phrase and keep only useful characters."""
    # Collapse whitespace and remove leading/trailing punctuation except quotes
    phrase = " ".join(str(phrase).split()).strip().strip('"').strip()
    return phrase[:MAX_PHRASE_LENGTH]


def sanitize_plan(plan: QueryPlan) -> QueryPlan:
    """Apply deterministic constraints to the LLM plan."""
    lookback = min(max(plan.lookback_days, MIN_LOOKBACK_DAYS), MAX_LOOKBACK_DAYS)

    phrases = []
    seen = set()
    for raw in plan.search_phrases:
        phrase = _clean_phrase(raw)
        if not phrase:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
        if len(phrases) >= MAX_SEARCH_PHRASES:
            break

    pos = plan.positive_keywords or {}
    if pos:
        pos_items = sorted(pos.items(), key=lambda kv: kv[1], reverse=True)[:MAX_POSITIVE_KEYWORDS]
        pos = {k: min(max(float(v), MIN_WEIGHT), MAX_WEIGHT) for k, v in pos_items}

    neg = plan.negative_keywords or {}
    if neg:
        neg_items = sorted(neg.items(), key=lambda kv: kv[1], reverse=True)[:MAX_NEGATIVE_KEYWORDS]
        neg = {k: min(max(float(v), MIN_WEIGHT), MAX_WEIGHT) for k, v in neg_items}
        # Remove overly generic negative keywords that could remove valid papers.
        neg = {k: v for k, v in neg.items() if k.strip().lower() not in GENERIC_NEGATIVE_KEYWORDS}

    return QueryPlan(
        search_phrases=phrases,
        positive_keywords=pos,
        negative_keywords=neg,
        lookback_days=lookback,
    )


# ---------------------------------------------------------------------------
# Source-specific query builders
# ---------------------------------------------------------------------------
def build_openalex_query(plan: QueryPlan) -> str:
    """Build an OpenAlex `search` string from the sanitized plan."""
    phrases = [p for p in plan.search_phrases if p]
    if not phrases:
        return ""
    # Quote each phrase and OR them together. Parentheses make the string safe.
    quoted = [f'"{p}"' for p in phrases]
    return " OR ".join(quoted)


def build_arxiv_query(plan: QueryPlan) -> str:
    """Build an arXiv `all:` field query from the sanitized plan."""
    phrases = [p for p in plan.search_phrases if p]
    if not phrases:
        return ""
    # arXiv field query: all:"phrase". OR them for broad but relevant recall.
    quoted = [f'all:"{p}"' for p in phrases]
    return " OR ".join(quoted)


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------
def _extract_json_object(text: str) -> str:
    """Extract the first JSON object from a string, even if surrounded by prose."""
    # Find the first '{' and last '}' in the text.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM output: {text[:500]!r}")
    return text[start : end + 1]


def _parse_llm_output(output: Any) -> dict:
    """Extract a JSON dict from common LLM output formats."""
    if isinstance(output, str):
        text = output
    elif hasattr(output, "content"):
        text = output.content
    else:
        text = str(output)

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Remove markdown code fences if present
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text, flags=re.IGNORECASE)

    # Extract the first JSON object from the remaining text
    try:
        json_text = _extract_json_object(text)
        return json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse JSON from LLM output: {text[:500]!r}") from exc


# ---------------------------------------------------------------------------
# Query interpretation
# ---------------------------------------------------------------------------
def interpret_query(query: str, llm=None) -> QueryPlan:
    """Convert a free-form user query into a validated, sanitized QueryPlan."""
    prompt = f"""
You are a scientific search assistant. A user wants to see a digest of
recent research on the following topic:

"{query}"

Please create a search plan for the OpenAlex scholarly database and a
deterministic prefilter. The plan must include:

1. search_phrases: a JSON array of 3-10 exact phrases that best capture the
   research topic. These phrases will be used verbatim as search queries for
   OpenAlex and arXiv. Use simple noun phrases or multi-word terms, not
   sentences.

2. positive_keywords: a JSON object mapping 3-10 important topical phrases
   to relevance weights (0.5 to 3.0). Higher weight means more important.

3. negative_keywords: a JSON object mapping 0-5 phrases that should be
   down-weighted because they are often false positives for this topic.
   Use weights from 0.5 to 3.0.

4. lookback_days: an integer from 1 to 60. Default 7 if unsure.

Output only a JSON object with those four keys. Do not include any other
text, explanations, or markdown.
"""

    # Use the OpenAI-compatible chat endpoint
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
    model_name = OLLAMA_MODEL
    if model_name.startswith("ollama/"):
        model_name = model_name[len("ollama/"):]

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": OLLAMA_TEMPERATURE,
        "stream": False,
        # Force the model's sampler to emit only valid JSON. This uses
        # Ollama's grammar-constrained decoding and effectively eliminates
        # the "missing comma" class of failures we saw with qwen3:4b.
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        raw_output = data["choices"][0]["message"]["content"]
        if not raw_output:
            raise ValueError("Empty response from LLM")
    except Exception as exc:
        raise RuntimeError(f"Failed to call local LLM at {url}: {exc}") from exc

    parsed = _parse_llm_output(raw_output)
    plan = QueryPlan.model_validate(parsed)
    return sanitize_plan(plan)


if __name__ == "__main__":
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "solid-state batteries"
    plan = interpret_query(q)
    print("Search phrases:", plan.search_phrases)
    print("OpenAlex query:", build_openalex_query(plan))
    print("arXiv query:", build_arxiv_query(plan))
    print("Positive keywords:", plan.positive_keywords)
    print("Negative keywords:", plan.negative_keywords)
    print("Lookback days:", plan.lookback_days)