"""Validated records shared by ingestion, retrieval, the agent and API."""

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Authority = Literal["OSFI", "BCBS"]
Status = Literal[
    "answered", "insufficient_evidence", "out_of_scope", "ambiguous_authority",
    "ambiguous_version", "service_error",
]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSpec(Record):
    doc_id: str
    title: str
    authority: Authority
    source_url: str
    pdf_url: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_count: int = Field(gt=0)
    retrieved_at: str
    published_at: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    language: str = "en"
    status: str = "final"

    @property
    def version_id(self) -> str:
        return f"{self.doc_id}:{self.sha256[:16]}"


class CorpusManifest(Record):
    release_id: str
    reference_date: date
    sources: list[SourceSpec]
    extraction_version: str
    chunk_tokens: int = 320
    overlap_tokens: int = 40
    model_revisions: dict[str, str]

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        if len(self.sources) != 5 or len({s.doc_id for s in self.sources}) != 5:
            raise ValueError("A corpus release must contain five unique sources")
        return self


class PageRecord(Record):
    doc_id: str
    version_id: str
    authority: Authority
    pdf_page_index: int = Field(gt=0)
    text: str
    pdf_url: str
    printed_page_label: str | None = None
    quality_flags: list[str] = Field(default_factory=list)


class PassageRecord(Record):
    passage_id: str
    doc_id: str
    version_id: str
    authority: Authority
    pdf_page_index: int = Field(gt=0)
    text: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    pdf_url: str
    section_heading: str = ""
    paragraph_label: str = ""
    printed_page_label: str | None = None

    @model_validator(mode="after")
    def valid_offsets(self) -> Self:
        if self.char_end <= self.char_start or len(self.text) != self.char_end - self.char_start:
            raise ValueError("Passage text must exactly cover its character interval")
        return self

    @property
    def page_id(self) -> str:
        return f"{self.version_id}:p{self.pdf_page_index}"


class EvidencePassage(PassageRecord):
    release_id: str
    score: float = 0.0
    ranks: dict[str, float] = Field(default_factory=dict)


class Claim(Record):
    text: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class ClaimVerdict(Record):
    claim_index: int = Field(ge=0)
    status: Literal["supported", "unsupported", "uncertain"]
    evidence_ids: list[str]
    rationale: str


class AskRequest(Record):
    question: str = Field(min_length=1, max_length=2000)
    authority: Literal["OSFI", "BCBS", "comparison"] | None = None
    as_of: date | None = None

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class VerificationReport(Record):
    passed: bool
    claims: list[Claim]
    verdicts: list[ClaimVerdict]
    errors: list[str]


def format_claims(claims: list[Claim]) -> str:
    return "\n\n".join(f"{c.text} [{', '.join(c.citation_ids)}]" for c in claims)


class AnswerResult(Record):
    status: Status
    answer_text: str = ""
    claims: list[Claim] = Field(default_factory=list)
    citations: list[EvidencePassage] = Field(default_factory=list)
    corpus_release: str
    trace_id: str
    reason: str | None = None
    provider_calls: int = 0

    @model_validator(mode="after")
    def enforce_answer_boundary(self) -> Self:
        if self.status == "answered":
            ids = {p.passage_id for p in self.citations}
            if not self.claims or self.answer_text != format_claims(self.claims):
                raise ValueError("Answers must be rendered from nonempty verified claims")
            if any(cid not in ids for c in self.claims for cid in c.citation_ids):
                raise ValueError("Every claim must have a returned citation")
            if any(p.release_id != self.corpus_release for p in self.citations):
                raise ValueError("Answer cannot mix corpus releases")
        elif self.answer_text or self.claims or self.citations:
            raise ValueError("Nonanswers cannot contain answer text or evidence claims")
        return self


class ReadinessReport(Record):
    ready: bool
    reasons: list[str]


class InputValidationError(ValueError):
    """Input fits JSON schema but exceeds a model-specific limit."""
