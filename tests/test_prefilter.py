from datetime import date

from src.prefilter import PrefilterConfig, filter_papers

CONFIG = PrefilterConfig(
    positive_keywords={"catalysis": 2.0, "electrochemistry": 2.0, "battery": 1.0},
    negative_keywords={"fluid dynamics": 3.0, "turbulence": 2.0},
    min_abstract_chars=40,
    min_relevance_score=-5.0,          # ← changed from 1.0
    positive_match_required=True,
    relevance_weight=1.0,
    recency_weight=1.0,
    recency_half_life_days=14.0,
)

def paper(id_, title, abstract, published="2026-09-05", doi=""):
    return {
        "id": id_,
        "title": title,
        "abstract": abstract,
        "authors": [],
        "published": published,
        "url": f"https://arxiv.org/abs/{id_}",
        "doi": doi,
        "source": "arXiv",
    }


def test_duplicate_doi_is_removed():
    papers = [
        paper("a", "Catalysis study", "This study investigates catalysis in a new chemical reaction." , doi="10.1234/example"),
        paper("b", "Same paper", "This is the same chemistry paper with a different identifier.", doi="10.1234/example"),
    ]

    result = filter_papers(papers, CONFIG, today=date(2026, 9, 6))

    assert result.input_count == 2
    assert result.output_count == 1
    assert result.duplicate_count == 1
    assert result.candidates[0]["id"] == "a"


def test_missing_abstract_is_rejected():
    papers = [paper("a", "Catalysis", "")]

    result = filter_papers(papers, CONFIG, today=date(2026, 9, 6))

    assert result.output_count == 0
    assert result.missing_abstract_count == 1
    assert "abstract" in result.filtered_examples[0]["reason"]


def test_positive_keyword_is_required():
    papers = [
        paper(
            "physics",
            "Fluid simulation",
            "We study numerical methods for turbulence and fluid dynamics in complex flows.",
        )
    ]

    result = filter_papers(papers, CONFIG, today=date(2026, 9, 6))

    assert result.output_count == 0
    assert result.low_relevance_count == 1
    assert "positive" in result.filtered_examples[0]["reason"]


def test_negative_keywords_penalize_relevance():
    papers = [
        paper(
            "mixed",
            "Catalysis in fluid dynamics",
            "We investigate catalysis alongside fluid dynamics and turbulence in a coupled system.",
        )
    ]

    result = filter_papers(papers, CONFIG, today=date(2026, 9, 6))

    assert result.output_count == 1
    assert result.details[0].relevance_score < 0
    assert "fluid dynamics" in result.details[0].negative_matches


def test_recency_changes_rank_deterministically():
    papers = [
        paper("old", "Battery research", "A study of battery materials and electrochemistry.", "2026-08-20"),
        paper("new", "Advanced battery research", "A study of battery materials and electrochemistry.", "2026-09-05"),
    ]

    result = filter_papers(papers, CONFIG, today=date(2026, 9, 6))

    assert [p["id"] for p in result.candidates] == ["new", "old"]
    assert result.details[0].recency_score > result.details[1].recency_score


def test_phrase_matching_is_case_insensitive_and_boundary_aware():
    papers = [
        paper("a", "Electrochemistry", "ELECTROCHEMISTRY is central to this study."),
        paper("b", "Not electrochemical",
              "The word electrochemicalxyz should not count as a match."),
    ]

    result = filter_papers(papers, CONFIG, today=date(2026, 9, 6))

    assert result.output_count == 1
    assert result.candidates[0]["id"] == "a"





def test_title_deduplication_removes_duplicate_titles():
    papers = [
        {"id": "a", "title": "Hydrogen storage in materials", "abstract": "x" * 100, "published": "2026-09-01"},
        {"id": "b", "title": "Hydrogen storage in materials:", "abstract": "y" * 100, "published": "2026-09-02"},
        {"id": "c", "title": "Hydrogen storage in materials!", "abstract": "z" * 100, "published": "2026-09-03"},
    ]
    config = PrefilterConfig(
        positive_match_required=False,
        min_relevance_score=0.0,
        min_abstract_chars=10,
    )
    result = filter_papers(papers, config)
    assert result.duplicate_count == 2  # two duplicates removed by title
    assert len(result.candidates) == 1
    assert result.candidates[0]["id"] == "a"


def test_candidate_cap_limits_candidates():
    papers = [
        {"id": f"p{i}", "title": f"Paper {i}", "abstract": f"abstract {i} " * 20, "published": "2026-09-01"}
        for i in range(20)
    ]
    config = PrefilterConfig(
        positive_match_required=False,
        min_relevance_score=0.0,
        min_abstract_chars=10,
        max_candidates=5,
    )
    result = filter_papers(papers, config)
    assert len(result.candidates) == 5
    # The first paper should still be present
    assert result.candidates[0]["id"] == "p0"
