"""Code-enforced citation boundaries, separate from fallible semantic judgments."""

from risk_qa.contracts import (
    Claim,
    ClaimVerdict,
    EvidencePassage,
    VerificationReport,
    format_claims,
)


def verify_claims(claims: list[Claim], evidence: list[EvidencePassage],
                  semantic_verdicts: list[ClaimVerdict],
                  required_authorities: set[str]) -> VerificationReport:
    errors = []
    allowed = {p.passage_id: p for p in evidence}
    if len(allowed) != len(evidence) or len({p.release_id for p in evidence}) > 1:
        errors.append("Ambiguous or mixed evidence identity")
    if not claims:
        errors.append("No factual claims")
    indices = [v.claim_index for v in semantic_verdicts]
    if sorted(indices) != list(range(len(claims))):
        errors.append("Exactly one semantic verdict is required for every claim")
    judgments = {v.claim_index: v for v in semantic_verdicts}
    used_authorities: set[str] = set()
    for i, claim in enumerate(claims):
        ids = set(claim.citation_ids)
        if not ids or not ids.issubset(allowed):
            errors.append(f"Claim {i} refers to evidence outside the current retrieval")
            continue
        used_authorities.update(allowed[cid].authority for cid in ids)
        verdict = judgments.get(i)
        if verdict is None or verdict.status != "supported":
            errors.append(f"Claim {i} is not supported by its semantic verdict")
        elif set(verdict.evidence_ids) != ids:
            errors.append(f"Claim {i} citations differ from the verifier's supporting evidence")
    if not required_authorities.issubset(used_authorities):
        errors.append("Missing support from a required authority")
    return VerificationReport(passed=not errors, claims=claims, verdicts=semantic_verdicts, errors=errors)


def render_verified(report: VerificationReport) -> str:
    return format_claims(report.claims) if report.passed else ""
