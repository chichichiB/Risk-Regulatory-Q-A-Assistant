"""Frozen PDF provenance and page-local text intervals."""

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

from pypdf import PdfReader

from risk_qa.contracts import CorpusManifest, PageRecord, PassageRecord, SourceSpec


class SnapshotHashMismatch(ValueError):
    pass


class ExtractionError(ValueError):
    pass


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> CorpusManifest:
    manifest = CorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))
    payload = manifest.model_dump(mode="json", exclude={"release_id"})
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    if manifest.release_id != expected:
        raise ValueError("Manifest release hash does not match its configuration")
    return manifest


def verify_snapshot(spec: SourceSpec, path: Path) -> None:
    if file_hash(path) != spec.sha256:
        raise SnapshotHashMismatch(f"Snapshot changed: {spec.doc_id}; preserve the original release")


def extract_pages(spec: SourceSpec, pdf_path: Path) -> list[PageRecord]:
    verify_snapshot(spec, pdf_path)
    reader = PdfReader(pdf_path)
    if len(reader.pages) != spec.page_count:
        raise ExtractionError(f"Page count mismatch: {spec.doc_id}")
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = "\n".join(line.rstrip() for line in
                         (page.extract_text(extraction_mode="layout") or "").splitlines())
        flags = []
        if not text.strip():
            flags.append("empty")
        if re.search(r"\S {4,}\S", text):
            flags.append("possible_table_or_columns")
        pages.append(PageRecord(doc_id=spec.doc_id, version_id=spec.version_id,
                                authority=spec.authority, pdf_page_index=i, text=text,
                                pdf_url=spec.pdf_url, quality_flags=flags))
    return pages


def validate_passage_offsets(passage: PassageRecord, page: PageRecord) -> None:
    if (passage.version_id, passage.pdf_page_index) != (page.version_id, page.pdf_page_index):
        raise ExtractionError("Passage refers to another page")
    if page.text[passage.char_start:passage.char_end] != passage.text:
        raise ExtractionError("Passage text differs from the frozen page interval")


def chunk_pages(pages: list[PageRecord], token_counter: Callable[[str], int],
                max_tokens: int = 320, overlap_tokens: int = 40) -> list[PassageRecord]:
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("Overlap must be nonnegative and smaller than the chunk limit")
    chunks = []
    for page in pages:
        words = list(re.finditer(r"\S+\s*", page.text))
        start = 0
        while start < len(words):
            char_start = 0 if start == 0 else words[start].start()
            low, high = start + 1, len(words)
            if token_counter(page.text[char_start:words[start].end()]) > max_tokens:
                raise ExtractionError(f"Unsplit token exceeds limit on {page.doc_id} p{page.pdf_page_index}")
            end = low
            while low <= high:
                mid = (low + high) // 2
                if token_counter(page.text[char_start:words[mid - 1].end()]) <= max_tokens:
                    end, low = mid, mid + 1
                else:
                    high = mid - 1
            char_end = words[end - 1].end()
            text = page.text[char_start:char_end]
            pid = f"{page.version_id}:p{page.pdf_page_index}:{char_start}-{char_end}"
            chunks.append(PassageRecord(
                passage_id=pid, doc_id=page.doc_id, version_id=page.version_id,
                authority=page.authority, pdf_page_index=page.pdf_page_index, text=text,
                char_start=char_start, char_end=char_end, pdf_url=page.pdf_url,
                printed_page_label=page.printed_page_label,
            ))
            if end == len(words):
                break
            next_start = end
            while next_start > start + 1 and overlap_tokens:
                candidate = next_start - 1
                if token_counter(page.text[words[candidate].start():char_end]) > overlap_tokens:
                    break
                next_start = candidate
            start = next_start
    return chunks
