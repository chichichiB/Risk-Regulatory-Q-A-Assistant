import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.live]


def test_live_answer_has_verified_source_citations():
    if os.environ.get("RUN_PAID_TESTS") != "1":
        pytest.skip("Explicit paid opt-in required: RUN_PAID_TESTS=1 and configured budget")
    from risk_qa.config import Settings
    from risk_qa.contracts import AskRequest
    from risk_qa.runtime import build_service

    service = build_service(Settings())
    assert service.readiness().ready
    result = service.answer(AskRequest(question="Under B-10, who retains accountability for outsourced services?", authority="OSFI"))
    assert result.status == "answered"
    assert any(p.doc_id == "osfi-b10" for p in result.citations)
    assert result.provider_calls <= 8
