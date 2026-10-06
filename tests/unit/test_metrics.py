from risk_qa.contracts import AnswerResult
from tests.helpers import passage


def gold(spans, answerable=True):
    from risk_qa.eval_contracts import GoldItem

    return GoldItem(item_id="q1", split="benchmark", question="A question", authority="OSFI",
                    primary_doc="d1", answerable=answerable,
                    expected_status="answered" if answerable else "out_of_scope",
                    reference_answer="Reference" if answerable else "", support_sets=[spans] if answerable else [])


def span(page, start=0, end=8):
    from risk_qa.eval_contracts import GoldSpan

    return GoldSpan(version_id="v1", pdf_page_index=page, char_start=start, char_end=end,
                    text_sha256="0" * 64)


def test_recall_hit_complete_pages_and_exact_spans_are_distinct():
    from risk_qa.metrics import score_retrieval

    scores = score_retrieval([gold([span(1), span(2)])], {"q1": [passage(text="Evidence")]})
    assert scores.page_recall_at_5 == 0.5
    assert scores.hit_at_5 == 1
    assert scores.complete_page_coverage_at_5 == 0
    assert scores.complete_evidence_at_5 == 0
    scores = score_retrieval([gold([span(1, 100, 110)])], {"q1": [passage(text="Evidence")]})
    assert scores.complete_page_coverage_at_5 == 1
    assert scores.complete_evidence_at_5 == 0


def test_duplicate_pages_do_not_refill_five_slots_and_negatives_excluded():
    from risk_qa.metrics import score_retrieval

    found = [passage(pid=f"p{i}", text="Evidence") for i in range(5)]
    found.append(passage(pid="p6", page=2, text="Evidence"))
    scores = score_retrieval([gold([span(2)]), gold([], False)], {"q1": found})
    assert scores.answerable_queries == 1
    assert scores.page_recall_at_5 == 0


def test_adjacent_chunks_cover_a_gold_span():
    from risk_qa.metrics import score_retrieval

    found = [passage(text="abcd"), passage(pid="p2", text="efgh").model_copy(update={"char_start": 4, "char_end": 8})]
    assert score_retrieval([gold([span(1)])], {"q1": found}).complete_evidence_at_5 == 1


def test_errors_stay_in_denominator_and_zero_assessed_claims_is_null():
    from risk_qa.metrics import score_answers

    result = AnswerResult(status="service_error", corpus_release="r1", trace_id="t")
    scores = score_answers([gold([span(1)])], {"q1": result}, {})
    assert scores.attempted_queries == 1
    assert scores.service_errors == 1
    assert scores.answer_coverage == 0
    assert scores.claim_faithfulness is None
    assert scores.assessed_claims == 0
