"""A bounded LangGraph workflow, with per-request state and injected I/O."""

import json
import time
from collections.abc import Callable
from datetime import date
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from risk_qa import prompts
from risk_qa.citations import render_verified, verify_claims
from risk_qa.contracts import AnswerResult, AskRequest, InputValidationError, ReadinessReport
from risk_qa.llm import DraftAnswer, EvidenceGrade, QueryRewrite, RouteDecision, SemanticVerdicts


class AgentState(TypedDict, total=False):
    query: str
    authority: str | None
    evidence: list
    claims: list
    report: Any
    rewrites: int
    repairs: int
    status: str
    reason: str
    adequate: bool


class AnswerService:
    def __init__(self, retriever, llm_factory: Callable, release_id: str, reference_date: date,
                 timeout_seconds: float = 120, max_output_tokens: int = 1500,
                 readiness_check: Callable[[], list[str]] | None = None):
        self.retriever, self.llm_factory = retriever, llm_factory
        self.release_id, self.reference_date = release_id, reference_date
        self.timeout_seconds, self.max_output_tokens = timeout_seconds, max_output_tokens
        self.readiness_check = readiness_check

    def readiness(self) -> ReadinessReport:
        reasons = self.readiness_check() if self.readiness_check else []
        return ReadinessReport(ready=not reasons, reasons=reasons)

    def answer(self, request: AskRequest) -> AnswerResult:
        trace, calls = uuid4().hex, 0
        deadline = time.monotonic() + self.timeout_seconds

        def result(status, reason=None, **kwargs):
            return AnswerResult(status=status, reason=reason, corpus_release=self.release_id,
                                trace_id=trace, provider_calls=calls, **kwargs)

        if request.as_of and request.as_of != self.reference_date:
            return result("ambiguous_version", "This corpus supports only its frozen reference date.")

        def deadline_check():
            if time.monotonic() >= deadline:
                raise TimeoutError("Request deadline reached")

        try:
            limits = getattr(getattr(self.retriever, "embedder", None), "limits", None)
            if limits:
                limits.validate_query(request.question)
            llm = self.llm_factory()

            def call(schema, instruction, state):
                nonlocal calls
                deadline_check()
                if calls >= 8:
                    raise RuntimeError("Provider call bound reached")
                calls += 1
                payload = {"question": request.question, "requested_authority": request.authority,
                           "reference_date": str(self.reference_date), "state": state}
                messages = [{"role": "system", "content": instruction},
                            {"role": "user", "content": json.dumps(payload, default=lambda v: v.model_dump())}]
                output = llm.complete(schema, messages, self.max_output_tokens)
                deadline_check()
                return schema.model_validate(output)

            def route(state):
                routed = call(RouteDecision, prompts.ROUTE, {})
                authority = request.authority or routed.authority
                if routed.status != "answerable":
                    return {"status": routed.status, "reason": routed.reason}
                if authority is None:
                    return {"status": "ambiguous_authority", "reason": "Specify OSFI or Basel."}
                return {"status": "working", "authority": authority}

            def retrieve(state):
                deadline_check()
                authority = None if state["authority"] == "comparison" else state["authority"]
                evidence = self.retriever.search(state["query"], self.release_id, authority)
                if any(p.release_id != self.release_id or
                       (authority and p.authority != authority) for p in evidence):
                    raise ValueError("Retrieval crossed evidence boundaries")
                return {"evidence": evidence}

            def grade(state):
                if not state["evidence"]:
                    return {"adequate": False}
                graded = call(EvidenceGrade, prompts.GRADE, {"evidence": state["evidence"]})
                allowed = {p.passage_id for p in state["evidence"]}
                if not set(graded.evidence_ids).issubset(allowed):
                    raise ValueError("Evidence grader cited an unavailable passage")
                selected = [p for p in state["evidence"] if p.passage_id in graded.evidence_ids]
                return {"adequate": graded.adequate and bool(selected),
                        "evidence": selected if graded.adequate else state["evidence"]}

            def rewrite(state):
                rewritten = call(QueryRewrite, prompts.REWRITE, {"query": state["query"]})
                if not rewritten.search_query.strip():
                    raise ValueError("Empty rewritten query")
                if limits:
                    limits.validate_query(rewritten.search_query)
                return {"query": rewritten.search_query, "rewrites": state["rewrites"] + 1}

            def generate(state):
                drafted = call(DraftAnswer, prompts.GENERATE, {
                    "evidence": state["evidence"],
                    "repair_feedback": state["report"].errors if state.get("report") else [],
                })
                return {"claims": drafted.claims}

            def verify(state):
                if state["claims"]:
                    judgments = call(SemanticVerdicts, prompts.VERIFY,
                                     {"claims": state["claims"], "evidence": state["evidence"]})
                    verdicts = judgments.verdicts
                else:
                    verdicts = []
                required = {"OSFI", "BCBS"} if state["authority"] == "comparison" else {state["authority"]}
                report = verify_claims(state["claims"], state["evidence"], verdicts, required)
                return {"report": report}

            def repair(state):
                updated = generate(state)
                return {**updated, "repairs": state["repairs"] + 1}

            def refuse(state):
                return {"status": "insufficient_evidence",
                        "reason": "The selected documents did not provide sufficient verified support."}

            graph = StateGraph(AgentState)
            for name, node in [("route", route), ("retrieve", retrieve), ("grade", grade),
                               ("rewrite", rewrite), ("generate", generate), ("verify", verify),
                               ("repair", repair), ("refuse", refuse)]:
                graph.add_node(name, node)
            graph.add_edge(START, "route")
            graph.add_conditional_edges("route", lambda s: "retrieve" if s["status"] == "working" else END)
            graph.add_edge("retrieve", "grade")
            graph.add_conditional_edges("grade", lambda s: "generate" if s["adequate"] else
                                        ("rewrite" if s["rewrites"] == 0 else "refuse"))
            graph.add_edge("rewrite", "retrieve")
            graph.add_edge("generate", "verify")
            graph.add_conditional_edges("verify", lambda s: END if s["report"].passed else
                                        ("repair" if s["repairs"] == 0 else "refuse"))
            graph.add_edge("repair", "verify")
            graph.add_edge("refuse", END)
            final = graph.compile().invoke({"query": request.question, "rewrites": 0, "repairs": 0},
                                            {"recursion_limit": 20})
            deadline_check()
            if final.get("report") and final["report"].passed:
                ids = {cid for c in final["claims"] for cid in c.citation_ids}
                return result("answered", claims=final["claims"],
                              answer_text=render_verified(final["report"]),
                              citations=[p for p in final["evidence"] if p.passage_id in ids])
            return result(final["status"], final.get("reason"))
        except InputValidationError:
            raise
        except Exception:
            return result("service_error", "A dependency, output validation, timeout or configured budget prevented completion.")
