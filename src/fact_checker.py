"""
Fact-checking stage for the Sci News pipeline.

This module runs after the Writer has produced a structured draft with
explicit claims. It validates the paper IDs in Python, then sends the
valid claims and the evidence abstracts to a single LLM call for
verification. The output is a structured FactCheckReport.
"""
from __future__ import annotations

import json
from typing import Dict, List, Literal, Optional, Set

from crewai import Agent, Crew, LLM, Task
from pydantic import BaseModel, Field, ValidationError

from config import FACT_CHECKER_ABSTRACT_CHARS, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TEMPERATURE
from src.crew_setup import Claim, WriterOutput, get_local_llm

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
Verdict = Literal[
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "NOT_CHECKABLE",
    "INVALID_CITATION",
]

OverallStatus = Literal["PASS", "PARTIAL", "FAIL"]


class ClaimAssessment(BaseModel):
    claim: str
    paper_ids: List[str] = Field(default_factory=list)
    verdict: Verdict
    justification: str = ""
    evidence_type: str = "abstract"  # always abstract for now


class FactCheckReport(BaseModel):
    assessments: List[ClaimAssessment]
    overall_status: OverallStatus
    summary: str = ""


def _validate_claim_ids(claims: List[Claim], valid_ids: Set[str]) -> tuple[List[ClaimAssessment], List[Claim]]:
    """
    Split claims into:
      - invalid assessments (INVALID_CITATION) for those referencing IDs not in valid_ids.
      - valid claims for the LLM to assess.
    """
    invalid_assessments = []
    valid_claims = []
    for claim in claims:
        invalid = [pid for pid in claim.paper_ids if pid not in valid_ids]
        if invalid:
            invalid_assessments.append(
                ClaimAssessment(
                    claim=claim.claim,
                    paper_ids=claim.paper_ids,
                    verdict="INVALID_CITATION",
                    justification=f"Referenced paper ID(s) not found: {invalid}",
                    evidence_type="abstract",
                )
            )
        else:
            valid_claims.append(claim)
    return invalid_assessments, valid_claims


def _compute_overall_status(assessments: List[ClaimAssessment]) -> OverallStatus:
    """Determine overall status from the list of assessments."""
    if not assessments:
        return "PASS"  # no claims -> nothing to fail
    if any(a.verdict in ["CONTRADICTED", "INVALID_CITATION", "UNSUPPORTED"] for a in assessments):
        return "FAIL"
    if any(a.verdict in ["PARTIALLY_SUPPORTED", "NOT_CHECKABLE"] for a in assessments):
        return "PARTIAL"
    return "PASS"


def _parse_fact_check_output(raw: str, valid_claims: List[Claim]) -> FactCheckReport:
    """
    Parse the LLM's raw JSON into a FactCheckReport.
    If parsing fails, fall back to NOT_CHECKABLE for all valid claims.
    """
    try:
        data = json.loads(raw)
        # Validate that each assessment corresponds to a known claim.
        assessments = []
        claims_by_text = {c.claim: c.paper_ids for c in valid_claims}
        for item in data.get("assessments", []):
            claim_text = item.get("claim")
            if claim_text not in claims_by_text:
                # Unknown claim: mark as NOT_CHECKABLE with a note.
                assessments.append(
                    ClaimAssessment(
                        claim=claim_text,
                        paper_ids=item.get("paper_ids", []),
                        verdict="NOT_CHECKABLE",
                        justification="Fact-checker returned an assessment for an unrecognized claim.",
                        evidence_type="abstract",
                    )
                )
                continue
            verdict = item.get("verdict")
            if verdict not in ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "NOT_CHECKABLE"]:
                verdict = "NOT_CHECKABLE"
            assessments.append(
                ClaimAssessment(
                    claim=claim_text,
                    paper_ids=claims_by_text[claim_text],
                    verdict=verdict,
                    justification=item.get("justification", ""),
                    evidence_type="abstract",
                )
            )
        # Ensure all valid claims are assessed; if missing, mark NOT_CHECKABLE.
        assessed_texts = {a.claim for a in assessments}
        for claim in valid_claims:
            if claim.claim not in assessed_texts:
                assessments.append(
                    ClaimAssessment(
                        claim=claim.claim,
                        paper_ids=claim.paper_ids,
                        verdict="NOT_CHECKABLE",
                        justification="Fact-checker did not provide an assessment for this claim.",
                        evidence_type="abstract",
                    )
                )
        overall = _compute_overall_status(assessments)
        summary = data.get("summary", "")
        return FactCheckReport(assessments=assessments, overall_status=overall, summary=summary)
    except Exception as e:
        # Fallback: mark all valid claims NOT_CHECKABLE.
        fallback = [
            ClaimAssessment(
                claim=c.claim,
                paper_ids=c.paper_ids,
                verdict="NOT_CHECKABLE",
                justification=f"Failed to parse fact-checker output: {e}",
                evidence_type="abstract",
            )
            for c in valid_claims
        ]
        return FactCheckReport(assessments=fallback, overall_status="PARTIAL", summary="Error parsing fact-checker output.")


def run_fact_check(
    writer_output: WriterOutput,
    selected_papers: List[Dict],
    llm: Optional[LLM] = None,
) -> FactCheckReport:
    """
    Run the Fact Checker on the writer's claims.
    Steps:
      1. Collect all claims from all articles.
      2. Validate paper IDs in Python.
      3. If no valid claims, return immediately.
      4. Otherwise, make ONE LLM call with all valid claims and the evidence abstracts.
      5. Parse the LLM response into a FactCheckReport.
    """
    llm = llm or get_local_llm()
    valid_ids = {p["id"] for p in selected_papers}

    all_claims: List[Claim] = []
    for article in writer_output.articles:
        all_claims.extend(article.claims)

    if not all_claims:
        return FactCheckReport(assessments=[], overall_status="PASS", summary="No claims to verify.")

    # Step 1: Validate IDs
    invalid_assessments, valid_claims = _validate_claim_ids(all_claims, valid_ids)

    if not valid_claims:
        # All claims had invalid IDs; no LLM call needed.
        return FactCheckReport(
            assessments=invalid_assessments,
            overall_status="FAIL",
            summary="All claims referenced invalid paper IDs.",
        )

    # Step 2: Prepare evidence and prompt
    evidence = [
        {
            "id": p["id"],
            "title": p["title"],
            "abstract": p["abstract"][:FACT_CHECKER_ABSTRACT_CHARS],
        }
        for p in selected_papers
    ]
    prompt = _build_fact_check_prompt(valid_claims, evidence)

    # Step 3: Call LLM once
    fact_checker_agent = Agent(
        role="Scientific Fact Checker",
        goal="Verify claims against provided evidence and assign a verdict.",
        backstory="You are a meticulous fact-checker with expertise in scientific literature. You only use the provided evidence and never rely on outside knowledge.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    fact_check_task = Task(
        description=prompt,
        expected_output="A JSON object with 'assessments' (list of claim assessments) and 'summary' string.",
        agent=fact_checker_agent,
    )
    crew = Crew(agents=[fact_checker_agent], tasks=[fact_check_task], verbose=False)
    result = crew.kickoff()
    raw = getattr(result, "raw", None) or str(result)

    # Step 4: Parse and merge
    llm_report = _parse_fact_check_output(raw, valid_claims)
    all_assessments = invalid_assessments + llm_report.assessments
    overall_status = _compute_overall_status(all_assessments)
    return FactCheckReport(
        assessments=all_assessments,
        overall_status=overall_status,
        summary=llm_report.summary,
    )


def _build_fact_check_prompt(claims: List[Claim], evidence: List[Dict]) -> str:
    claims_json = json.dumps([c.model_dump() for c in claims], ensure_ascii=False)
    evidence_json = json.dumps(evidence, ensure_ascii=False)
    return f"""You are a scientific fact-checker. Your job is to evaluate each claim against the provided paper abstracts.

IMPORTANT: All evidence supplied consists of paper abstracts only. Your verdicts reflect abstract-level support, not full-paper verification.

Evidence papers (JSON array):
{evidence_json}

Claims to verify (JSON array):
{claims_json}

For each claim, assign one of the following verdicts:
- SUPPORTED: the abstract directly and clearly supports the claim.
- PARTIALLY_SUPPORTED: part of the claim is supported, but another part is not mentioned or is unsupported.
- UNSUPPORTED: the abstract does not contain information that supports the claim.
- CONTRADICTED: the abstract explicitly contradicts the claim.
- NOT_CHECKABLE: the abstract is insufficient to verify the claim (e.g., missing details, claim too broad, numerical values absent).

Rules:
- Use ONLY the provided abstracts. Do not use outside knowledge.
- Do not infer beyond what is written. If uncertain, choose NOT_CHECKABLE.
- For each claim, provide a "justification" explaining your reasoning.

Return a JSON object with:
- "assessments": a list of objects, each with "claim" (string), "paper_ids" (list of strings), "verdict" (one of the above), and "justification" (string).
- "summary": a brief overall summary.
"""


def build_revision_feedback(report: FactCheckReport) -> str:
    """Generate structured feedback for the Writer based on failed claims."""
    failed = [a for a in report.assessments if a.verdict != "SUPPORTED"]
    if not failed:
        return ""
    lines = ["The fact-checker flagged the following issues. Please revise the article to address them:"]
    for a in failed:
        lines.append(f"- Claim: \"{a.claim}\" (paper IDs: {a.paper_ids})")
        lines.append(f"  Verdict: {a.verdict}")
        lines.append(f"  Reason: {a.justification}")
    lines.append(
        "\nWhen revising, strictly follow these rules:\n"
        "- Do NOT introduce new facts.\n"
        "- Do NOT introduce new numbers.\n"
        "- Do NOT introduce new papers.\n"
        "- Do NOT introduce unsupported applications or implications.\n"
        "- Remove unsupported details.\n"
        "- If something is NOT_CHECKABLE because the abstract lacks sufficient evidence, "
        "remove the unsupported specificity or rewrite it so that it is directly supported by the abstract.\n"
        "- Every claim must remain traceable to the supplied abstract."
    )
    return "\n".join(lines)