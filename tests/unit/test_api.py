from fastapi.testclient import TestClient

from risk_qa.contracts import AnswerResult, InputValidationError, ReadinessReport


class FakeService:
    calls = 0
    available = True
    status = "insufficient_evidence"

    def readiness(self):
        return ReadinessReport(ready=self.available, reasons=[] if self.available else ["Unavailable"])

    def answer(self, request):
        self.calls += 1
        if request.question == "oversized tokens":
            raise InputValidationError("Too many tokens")
        return AnswerResult(status=self.status, corpus_release="r1", trace_id="trace")


def client_for(service):
    from risk_qa.api import create_app

    return TestClient(create_app(service))


def test_blank_question_returns_422_and_readiness_makes_no_answer_call():
    service = FakeService()
    with client_for(service) as client:
        assert client.post("/ask", json={"question": " "}).status_code == 422
        assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422
        assert client.get("/ready").status_code == 200
        assert "/ask" in client.get("/openapi.json").json()["paths"]
    assert service.calls == 0


def test_refusal_and_dependency_failure_have_distinct_http_statuses():
    service = FakeService()
    with client_for(service) as client:
        response = client.post("/ask", json={"question": "Risk question"})
        assert response.status_code == 200
        assert response.json()["status"] == "insufficient_evidence"
        service.status = "service_error"
        assert client.post("/ask", json={"question": "Risk question"}).status_code == 503
        assert client.post("/ask", json={"question": "oversized tokens"}).status_code == 422
        service.available = False
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200
        assert client.post("/ask", json={"question": "Risk question"}).status_code == 503
