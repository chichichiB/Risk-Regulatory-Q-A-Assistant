import os

import httpx
import pytest

pytestmark = pytest.mark.integration


def test_compose_serves_contract_and_explicit_readiness():
    url = os.environ.get("TEST_API_URL")
    if not url:
        pytest.skip("Set TEST_API_URL for the local Compose API")
    with httpx.Client(base_url=url, timeout=10) as client:
        assert client.get("/health").json() == {"status": "alive"}
        assert "/ask" in client.get("/openapi.json").json()["paths"]
        assert client.post("/ask", json={"question": " "}).status_code == 422
        ready = client.get("/ready")
        assert ready.status_code in {200, 503}
        assert ready.json()["ready"] == (ready.status_code == 200)
        # This smoke test never sends a valid question to a ready paid service.
        if ready.status_code == 503:
            response = client.post("/ask", json={"question": "Who manages third-party risk?"})
            assert response.status_code == 503
            assert response.json()["status"] == "service_error"
