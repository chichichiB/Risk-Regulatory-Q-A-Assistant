"""Thin HTTP adapter. Synchronous inference runs in FastAPI's worker pool."""

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response

from risk_qa.contracts import AnswerResult, AskRequest, InputValidationError, ReadinessReport

logger = logging.getLogger(__name__)


def create_app(service=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        if app.state.service is None:
            from risk_qa.config import Settings
            from risk_qa.runtime import build_service

            try:
                app.state.service = build_service(Settings())
            except Exception as exc:
                # Do not log connection strings, credentials or provider payloads.
                logger.error("Runtime unavailable (%s); check corpus, database and model cache", type(exc).__name__)
        yield

    app = FastAPI(title="Risk Regulatory Q&A", version="0.1.0", lifespan=lifespan,
                  description="Questions over a frozen five-document OSFI/Basel corpus.")
    app.state.service = service

    def readiness() -> ReadinessReport:
        if app.state.service is None:
            return ReadinessReport(ready=False, reasons=["Prepare models and corpus, ingest, then restart the API."])
        return app.state.service.readiness()

    @app.get("/health")
    def health():
        return {"status": "alive"}

    @app.get("/ready", response_model=ReadinessReport)
    def ready(response: Response):
        report = readiness()
        response.status_code = 200 if report.ready else 503
        return report

    @app.post("/ask", response_model=AnswerResult)
    def ask(request: AskRequest, response: Response):
        if not readiness().ready:
            response.status_code = 503
            return AnswerResult(status="service_error", corpus_release=getattr(app.state.service, "release_id", "unavailable"),
                                trace_id=uuid4().hex, reason="Service is not ready; inspect /ready.")
        try:
            answer = app.state.service.answer(request)
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        if answer.status == "service_error":
            response.status_code = 503
        return answer

    return app


app = create_app()
