import pytest
from pydantic import ValidationError


def test_blank_or_overlong_question_rejected():
    from risk_qa.contracts import AskRequest

    for text in [" ", "x" * 2001]:
        with pytest.raises(ValidationError):
            AskRequest(question=text)
    assert AskRequest(question="  What is risk? ").question == "What is risk?"


def test_answer_cannot_contain_unverified_free_prose():
    from risk_qa.contracts import AnswerResult

    with pytest.raises(ValidationError):
        AnswerResult(status="answered", answer_text="Invented answer", claims=[], citations=[],
                     corpus_release="release", trace_id="trace")


def test_nonanswer_cannot_smuggle_claims():
    from risk_qa.contracts import AnswerResult, Claim

    with pytest.raises(ValidationError):
        AnswerResult(status="insufficient_evidence", claims=[Claim(text="claim", citation_ids=["p"])],
                     corpus_release="release", trace_id="trace")


def test_page_and_passage_offsets_must_be_valid():
    from risk_qa.contracts import PassageRecord

    with pytest.raises(ValidationError):
        PassageRecord(passage_id="p", doc_id="doc", version_id="v", authority="OSFI",
                      pdf_page_index=0, text="text", char_start=4, char_end=2,
                      pdf_url="https://example.org/source.pdf")
