import pytest

from risk_qa.contracts import Claim, ClaimVerdict
from tests.helpers import passage


@pytest.mark.parametrize("verdicts", [[], ["unsupported"], ["uncertain"], ["supported", "supported"]])
def test_missing_failed_or_duplicate_verdict_cannot_render(verdicts):
    from risk_qa.citations import render_verified, verify_claims

    claim = Claim(text="Risk controls are required.", citation_ids=["p1"])
    judgments = [ClaimVerdict(claim_index=0, status=s, evidence_ids=["p1"], rationale="fixture")
                 for s in verdicts]
    report = verify_claims([claim], [passage()], judgments, {"OSFI"})
    assert report.passed is False
    assert render_verified(report) == ""


def test_citation_membership_and_verifier_evidence_must_agree():
    from risk_qa.citations import verify_claims

    claim = Claim(text="A claim", citation_ids=["p1"])
    verdict = ClaimVerdict(claim_index=0, status="supported", evidence_ids=["p2"], rationale="x")
    report = verify_claims([claim], [passage(), passage("p2")], [verdict], {"OSFI"})
    assert not report.passed
    foreign = claim.model_copy(update={"citation_ids": ["absent"]})
    assert not verify_claims([foreign], [passage()], [verdict], {"OSFI"}).passed


def test_answer_is_only_verified_claims_and_comparison_requires_both_sources():
    from risk_qa.citations import render_verified, verify_claims

    claim = Claim(text="A claim", citation_ids=["p1"])
    verdict = ClaimVerdict(claim_index=0, status="supported", evidence_ids=["p1"], rationale="x")
    report = verify_claims([claim], [passage()], [verdict], {"OSFI"})
    assert render_verified(report) == "A claim [p1]"
    assert not verify_claims([claim], [passage()], [verdict], {"OSFI", "BCBS"}).passed
