"""Deterministic candidate-paper filtering and ranking."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PrefilterConfig:
    positive_keywords: Mapping[str, float] = field(default_factory=dict)
    negative_keywords: Mapping[str, float] = field(default_factory=dict)
    min_abstract_chars: int = 80
    min_relevance_score: float = 0.0
    positive_match_required: bool = True
    relevance_weight: float = 1.0
    recency_weight: float = 1.0
    recency_half_life_days: float = 14.0


@dataclass(frozen=True)
class FilteredPaper:
    paper: dict[str, Any]
    score: float
    relevance_score: float
    recency_score: float
    positive_matches: tuple[str, ...]
    negative_matches: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PrefilterResult:
    candidates: list[dict[str, Any]]
    details: list[FilteredPaper]
    input_count: int
    output_count: int
    duplicate_count: int
    missing_abstract_count: int
    low_relevance_count: int
    filtered_examples: list[dict[str, Any]]


def _normalize(text: Any) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _normalize_identifier(value: Any) -> str:
    text = _normalize(value).rstrip("/")
    if not text:
        return ""
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    text = text.replace("https://arxiv.org/abs/", "")
    text = text.replace("https://arxiv.org/pdf/", "")
    return text


def stable_identifiers(paper: Mapping[str, Any]) -> tuple[str, ...]:
    values = []
    for key in ("doi", "url", "id"):
        normalized = _normalize_identifier(paper.get(key))
        if normalized:
            values.append(normalized)
    return tuple(dict.fromkeys(values))


def _phrase_matches(text: str, phrases: Iterable[str]) -> list[str]:
    normalized = _normalize(text)
    matches = []
    for phrase in phrases:
        p = _normalize(phrase)
        if p and re.search(rf"(?<!\w){re.escape(p)}(?!\w)", normalized):
            matches.append(phrase)
    return matches


def _recency_score(published: Any, today: date, half_life_days: float) -> float:
    if not published:
        return 0.0
    try:
        published_date = date.fromisoformat(str(published)[:10])
    except (TypeError, ValueError):
        return 0.0
    age_days = max((today - published_date).days, 0)
    return 0.5 ** (age_days / max(float(half_life_days), 0.0001))


def _score_paper(paper: Mapping[str, Any], config: PrefilterConfig, today: date) -> FilteredPaper | None:
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "").strip()
    if len(abstract) < config.min_abstract_chars:
        return None

    positive_matches = _phrase_matches(f"{title} {abstract}", config.positive_keywords)
    negative_matches = _phrase_matches(f"{title} {abstract}", config.negative_keywords)
    relevance_score = (
        sum(float(config.positive_keywords[item]) for item in positive_matches)
        - sum(float(config.negative_keywords[item]) for item in negative_matches)
    )
    recency_score = _recency_score(paper.get("published"), today, config.recency_half_life_days)
    total = config.relevance_weight * relevance_score + config.recency_weight * recency_score

    reasons = []
    if positive_matches:
        reasons.append(f"positive matches: {', '.join(positive_matches)}")
    if negative_matches:
        reasons.append(f"negative matches: {', '.join(negative_matches)}")
    reasons.append(f"recency score: {recency_score:.3f}")

    return FilteredPaper(
        paper=dict(paper),
        score=total,
        relevance_score=relevance_score,
        recency_score=recency_score,
        positive_matches=tuple(positive_matches),
        negative_matches=tuple(negative_matches),
        reasons=tuple(reasons),
    )


def _deduplicate(papers: Sequence[Mapping[str, Any]]):
    seen: set[str] = set()
    unique = []
    examples = []
    for paper in papers:
        identifiers = stable_identifiers(paper)
        duplicate_key = next((identifier for identifier in identifiers if identifier in seen), None)
        if duplicate_key:
            if len(examples) < 10:
                examples.append({"id": paper.get("id", ""), "title": paper.get("title", ""), "reason": f"duplicate identifier: {duplicate_key}"})
            continue
        seen.update(identifiers)
        unique.append(paper)
    return unique, len(papers) - len(unique), examples


def filter_papers(papers: Sequence[Mapping[str, Any]], config: PrefilterConfig, *, today: date | None = None) -> PrefilterResult:
    today = today or datetime.now(timezone.utc).date()
    unique, duplicate_count, duplicate_examples = _deduplicate(papers)
    ranked = []
    filtered_examples = list(duplicate_examples)
    missing_abstract_count = 0
    low_relevance_count = 0

    for index, paper in enumerate(unique):
        abstract = str(paper.get("abstract") or "").strip()
        if len(abstract) < config.min_abstract_chars:
            missing_abstract_count += 1
            if len(filtered_examples) < 10:
                filtered_examples.append({"id": paper.get("id", ""), "title": paper.get("title", ""), "reason": "missing/very short abstract"})
            continue

        scored = _score_paper(paper, config, today)
        assert scored is not None
        if config.positive_match_required and not scored.positive_matches:
            low_relevance_count += 1
            if len(filtered_examples) < 10:
                filtered_examples.append({"id": paper.get("id", ""), "title": paper.get("title", ""), "reason": "no configured positive topical matches"})
            continue
        if scored.relevance_score < config.min_relevance_score:
            low_relevance_count += 1
            if len(filtered_examples) < 10:
                filtered_examples.append({"id": paper.get("id", ""), "title": paper.get("title", ""), "reason": f"relevance score {scored.relevance_score:.2f} below threshold {config.min_relevance_score:.2f}"})
            continue
        ranked.append((index, scored))

    ranked.sort(key=lambda item: (-item[1].score, -item[1].relevance_score, item[0]))
    details = [item[1] for item in ranked]
    return PrefilterResult(
        candidates=[detail.paper for detail in details],
        details=details,
        input_count=len(papers),
        output_count=len(details),
        duplicate_count=duplicate_count,
        missing_abstract_count=missing_abstract_count,
        low_relevance_count=low_relevance_count,
        filtered_examples=filtered_examples,
    )
