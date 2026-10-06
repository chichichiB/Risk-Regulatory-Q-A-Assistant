from hashlib import sha256

import pytest

from risk_qa.contracts import PageRecord, SourceSpec


def test_changed_pdf_is_rejected(tmp_path):
    from risk_qa.corpus import SnapshotHashMismatch, verify_snapshot

    path = tmp_path / "source.pdf"
    path.write_bytes(b"changed")
    spec = SourceSpec(doc_id="a", title="title", authority="OSFI", source_url="https://x",
                      pdf_url="https://x/source.pdf", sha256=sha256(b"original").hexdigest(),
                      page_count=1, retrieved_at="2026-10-06")
    with pytest.raises(SnapshotHashMismatch):
        verify_snapshot(spec, path)


def test_chunks_cover_page_without_crossing_or_truncating():
    from risk_qa.corpus import chunk_pages, validate_passage_offsets

    pages = [PageRecord(doc_id="a", version_id="v", authority="OSFI", pdf_page_index=i,
                        text="alpha beta gamma delta epsilon zeta", pdf_url="https://x")
             for i in [1, 2]]
    chunks = chunk_pages(pages, token_counter=lambda x: len(x.split()),
                         max_tokens=3, overlap_tokens=1)
    assert len(chunks) >= 4
    for page in pages:
        own = [c for c in chunks if c.pdf_page_index == page.pdf_page_index]
        assert own[0].char_start == 0
        assert own[-1].char_end == len(page.text)
        for chunk in own:
            validate_passage_offsets(chunk, page)
            assert len(chunk.text.split()) <= 3
        assert all(b.char_start <= a.char_end for a, b in zip(own, own[1:]))


def test_unsplittable_token_and_empty_page_are_explicit():
    from risk_qa.corpus import ExtractionError, chunk_pages

    page = PageRecord(doc_id="a", version_id="v", authority="OSFI", pdf_page_index=1,
                      text="xxxxxxxxxxxx", pdf_url="https://x")
    with pytest.raises(ExtractionError):
        chunk_pages([page], token_counter=len, max_tokens=3, overlap_tokens=0)
    assert chunk_pages([page.model_copy(update={"text": ""})], len) == []
