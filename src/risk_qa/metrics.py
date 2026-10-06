"""Metrics preserve failed attempts and never substitute page hits for entailment."""

from risk_qa.contracts import AnswerResult, ClaimVerdict, EvidencePassage
from risk_qa.eval_contracts import AnswerMetrics, GoldItem, GoldSpan, RetrievalMetrics


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _covered(span: GoldSpan, passages: list[EvidencePassage]) -> bool:
    cursor = span.char_start
    for start, end in sorted((p.char_start, p.char_end) for p in passages if p.page_id == span.page_id):
        if start > cursor:
            break
        cursor = max(cursor, end)
        if cursor >= span.char_end:
            return True
    return False


def score_retrieval(items: list[GoldItem], ranked: dict[str, list[EvidencePassage]]) -> RetrievalMetrics:
    rows = []
    for item in items:
        if not item.answerable:
            continue
        found = ranked.get(item.item_id, [])[:5]
        found_pages = {p.page_id for p in found}
        # Each alternative is a complete valid support set; score the best valid alternative.
        alternatives = []
        for spans in item.support_sets:
            gold_pages = {s.page_id for s in spans}
            hit = len(found_pages & gold_pages)
            alternatives.append((hit / len(gold_pages), float(hit > 0),
                                 float(gold_pages <= found_pages), float(all(_covered(s, found) for s in spans))))
        rows.append(tuple(max(a[i] for a in alternatives) for i in range(4)))
    averages = [_ratio(sum(row[i] for row in rows), len(rows)) for i in range(4)]
    return RetrievalMetrics(answerable_queries=len(rows), page_recall_at_5=averages[0],
                            hit_at_5=averages[1], complete_page_coverage_at_5=averages[2],
                            complete_evidence_at_5=averages[3])


def score_answers(items: list[GoldItem], results: dict[str, AnswerResult],
                  judgments: dict[str, list[ClaimVerdict]]) -> AnswerMetrics:
    positive = sum(i.answerable for i in items)
    answered = errors = refusals = correct_negative = supported = assessed = total_claims = valid_citations = 0
    answered_positive = 0
    for item in items:
        result = results.get(item.item_id)
        if result is None or result.status == "service_error":
            errors += 1
            continue
        if result.status != "answered":
            refusals += 1
            correct_negative += int(not item.answerable and result.status == item.expected_status)
            continue
        answered += 1
        answered_positive += int(item.answerable)
        allowed = {p.passage_id for p in result.citations if p.release_id == result.corpus_release}
        total_claims += len(result.claims)
        valid_citations += sum(bool(c.citation_ids) and set(c.citation_ids) <= allowed for c in result.claims)
        verdicts = judgments.get(item.item_id, [])
        for index in range(len(result.claims)):
            matching = [v for v in verdicts if v.claim_index == index]
            if len(matching) == 1:
                assessed += 1
                verdict = matching[0]
                supported += int(verdict.status == "supported" and
                                 set(verdict.evidence_ids) == set(result.claims[index].citation_ids))
    negatives = len(items) - positive
    return AnswerMetrics(attempted_queries=len(items), answerable_queries=positive,
                         answered_queries=answered, service_errors=errors, refusals=refusals,
                         answer_coverage=_ratio(answered_positive, positive), negative_queries=negatives,
                         correct_negative_status=correct_negative,
                         negative_status_accuracy=_ratio(correct_negative, negatives),
                         assessed_claims=assessed, supported_claims=supported,
                         claim_faithfulness=_ratio(supported, assessed),
                         judgment_coverage=_ratio(assessed, total_claims),
                         citation_validity=_ratio(valid_citations, total_claims))
