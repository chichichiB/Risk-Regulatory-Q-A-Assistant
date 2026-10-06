"""Evaluation labels use physical PDF pages and exact canonical character offsets."""

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from risk_qa.contracts import AskRequest, PageRecord, Record, Status


class GoldSpan(Record):
    version_id: str
    pdf_page_index: int = Field(gt=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @property
    def page_id(self):
        return f"{self.version_id}:p{self.pdf_page_index}"


class GoldItem(AskRequest):
    item_id: str
    split: Literal["dev", "benchmark"]
    primary_doc: str
    answerable: bool
    expected_status: Status
    reference_answer: str
    support_sets: list[list[GoldSpan]]
    label_provenance: str = "AI-authored and source-checked; no human audit"
    human_audited: bool = False

    @model_validator(mode="after")
    def valid_support(self) -> Self:
        if self.answerable and (not self.support_sets or any(not s for s in self.support_sets)
                                or not self.reference_answer or self.expected_status != "answered"):
            raise ValueError("Answerable labels require a reference and nonempty support sets")
        if not self.answerable and (self.support_sets or self.expected_status in {"answered", "service_error"}):
            raise ValueError("Negative labels cannot contain support or expect service failures")
        if any(s.char_end <= s.char_start for group in self.support_sets for s in group):
            raise ValueError("Invalid span offsets")
        return self


class RetrievalMetrics(Record):
    answerable_queries: int
    page_recall_at_5: float | None
    hit_at_5: float | None
    complete_page_coverage_at_5: float | None
    complete_evidence_at_5: float | None


class AnswerMetrics(Record):
    attempted_queries: int
    answerable_queries: int
    answered_queries: int
    service_errors: int
    refusals: int
    answer_coverage: float | None
    negative_queries: int
    correct_negative_status: int
    negative_status_accuracy: float | None
    assessed_claims: int
    supported_claims: int
    claim_faithfulness: float | None
    judgment_coverage: float | None
    citation_validity: float | None


def load_dataset(path: Path, pages: list[PageRecord] | None = None) -> list[GoldItem]:
    items = [GoldItem.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({i.item_id for i in items}) != len(items):
        raise ValueError("Duplicate evaluation item IDs")
    if pages is not None:
        lookup = {(p.version_id, p.pdf_page_index): p for p in pages}
        for item in items:
            for group in item.support_sets:
                for span in group:
                    page = lookup[(span.version_id, span.pdf_page_index)]
                    text = page.text[span.char_start:span.char_end]
                    if span.char_end > len(page.text) or hashlib.sha256(text.encode()).hexdigest() != span.text_sha256:
                        raise ValueError(f"Gold span changed: {item.item_id}")
    return items


def verify_dataset_hash(path: Path, manifest_path: Path = Path("eval/manifest.json")) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["datasets"][path.name]["sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Dataset differs from frozen evaluation manifest")
