"""Structured OpenAI calls with explicit per-request and per-run budgets."""

import json
import threading
import time
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict

from risk_qa.contracts import Claim, ClaimVerdict


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RouteDecision(Output):
    status: Literal["answerable", "out_of_scope", "ambiguous_authority", "ambiguous_version"]
    authority: Literal["OSFI", "BCBS", "comparison"] | None
    reason: str


class EvidenceGrade(Output):
    adequate: bool
    evidence_ids: list[str]
    reason: str


class QueryRewrite(Output):
    search_query: str


class DraftAnswer(Output):
    claims: list[Claim]


class SemanticVerdicts(Output):
    verdicts: list[ClaimVerdict]


class BudgetExceeded(RuntimeError):
    pass


class SpendLedger:
    def __init__(self, maximum_usd: Decimal):
        if maximum_usd < 0:
            raise ValueError("Budget cannot be negative")
        self.maximum, self.spent = maximum_usd, Decimal("0")
        self.lock = threading.Lock()

    def reserve(self, amount: Decimal) -> None:
        with self.lock:
            if self.spent + amount > self.maximum:
                raise BudgetExceeded("Configured run budget exhausted")
            self.spent += amount

    def refund(self, amount: Decimal) -> None:
        with self.lock:
            self.spent -= amount


class CallBudget:
    def __init__(self, ledger: SpendLedger, input_rate: Decimal, output_rate: Decimal,
                 maximum_calls: int = 8):
        if input_rate < 0 or output_rate < 0 or not 1 <= maximum_calls <= 8:
            raise ValueError("Invalid cost/call limits")
        self.ledger, self.input_rate, self.output_rate = ledger, input_rate, output_rate
        self.maximum_calls, self.calls = maximum_calls, 0

    def reserve(self, input_tokens: int, max_output_tokens: int) -> Decimal:
        if input_tokens < 0 or max_output_tokens <= 0:
            raise ValueError("Invalid token estimate")
        if self.calls >= self.maximum_calls:
            raise BudgetExceeded("Per-request provider attempt limit reached")
        amount = (input_tokens * self.input_rate + max_output_tokens * self.output_rate) / 1_000_000
        self.ledger.reserve(amount)
        self.calls += 1
        return amount

    def settle(self, reserved: Decimal, observed_cost: Decimal | None) -> None:
        if observed_cost is None:
            return
        if observed_cost > reserved:
            # Never conceal a bad estimate: retain actual spend and stop further calls.
            with self.ledger.lock:
                self.ledger.spent += observed_cost - reserved
                self.ledger.maximum = self.ledger.spent
            raise BudgetExceeded("Actual usage exceeded its conservative reservation")
        self.ledger.refund(reserved - observed_cost)


class LLMClient:
    def __init__(self, settings, ledger: SpendLedger):
        from openai import OpenAI

        if not settings.openai_api_key or not settings.openai_model:
            raise RuntimeError("Configure OpenAI credentials and model locally")
        if settings.openai_input_usd_per_million is None or settings.openai_output_usd_per_million is None:
            raise RuntimeError("Configure model pricing before paid calls")
        self.settings = settings
        self.budget = CallBudget(ledger, settings.openai_input_usd_per_million,
                                 settings.openai_output_usd_per_million, settings.max_provider_calls)
        self.client = OpenAI(api_key=settings.openai_api_key.get_secret_value(), max_retries=0,
                             timeout=settings.provider_timeout_seconds)
        self.deadline = time.monotonic() + settings.request_timeout_seconds
        self.usage = []

    def complete(self, schema: type[BaseModel], messages: list[dict[str, str]],
                 max_output_tokens: int) -> BaseModel:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Request deadline reached")
        # UTF-8 bytes conservatively bound text tokens; include schema and framing allowance.
        byte_count = len(json.dumps([messages, schema.model_json_schema()]).encode("utf-8")) + 1000
        reserved = self.budget.reserve(byte_count, max_output_tokens)
        response = self.client.responses.parse(
            model=self.settings.openai_model, input=messages, text_format=schema,
            max_output_tokens=max_output_tokens, store=False,
            timeout=min(remaining, self.settings.provider_timeout_seconds),
        )
        usage = response.usage
        if usage:
            cost = (usage.input_tokens * self.budget.input_rate
                    + usage.output_tokens * self.budget.output_rate) / 1_000_000
            self.budget.settle(reserved, cost)
            self.usage.append({"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                               "cost_usd": str(cost), "model": response.model})
        if response.output_parsed is None:
            raise ValueError("Provider returned no complete structured result")
        return response.output_parsed
