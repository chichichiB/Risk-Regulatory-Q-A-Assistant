from decimal import Decimal

import pytest


def test_eighth_attempt_allowed_ninth_never_dispatched():
    from risk_qa.llm import BudgetExceeded, CallBudget, SpendLedger

    budget = CallBudget(SpendLedger(Decimal("1")), Decimal("1"), Decimal("1"))
    for _ in range(8):
        budget.reserve(10, 10)
    with pytest.raises(BudgetExceeded):
        budget.reserve(10, 10)
    assert budget.calls == 8


def test_timeouts_keep_reservation_and_run_budget_spans_requests():
    from risk_qa.llm import BudgetExceeded, CallBudget, SpendLedger

    ledger = SpendLedger(Decimal("0.00002"))
    first = CallBudget(ledger, Decimal("1"), Decimal("1"))
    reserved = first.reserve(10, 10)
    first.settle(reserved, None)
    second = CallBudget(ledger, Decimal("1"), Decimal("1"))
    with pytest.raises(BudgetExceeded):
        second.reserve(1, 1)
    assert second.calls == 0
