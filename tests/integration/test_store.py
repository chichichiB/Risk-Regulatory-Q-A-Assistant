import os
from uuid import uuid4

import psycopg
import pytest

from risk_qa.contracts import CorpusManifest, SourceSpec
from risk_qa.store import CorpusStore
from tests.helpers import passage

pytestmark = pytest.mark.integration


def test_real_vector_order_idempotence_and_atomic_rollback(monkeypatch):
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set TEST_DATABASE_URL for a disposable database")
    store = CorpusStore(dsn)
    store.initialize()
    sources = [SourceSpec(doc_id=f"doc{i}", title="fixture", authority="OSFI",
                          source_url="https://example.org", pdf_url="https://example.org/source.pdf",
                          sha256=str(i) * 64, page_count=1, retrieved_at="2026-10-06") for i in range(5)]
    manifest = CorpusManifest(release_id="test-" + uuid4().hex, reference_date="2026-10-06",
                              sources=sources, extraction_version="fixture", model_revisions={})
    ps = [passage(f"p{i}").model_copy(update={"doc_id": "doc0", "version_id": sources[0].version_id})
          for i in range(2)]
    vectors = [[1.0, 0.0] + [0.0] * 382, [0.0, 1.0] + [0.0] * 382]
    store.ingest_release(manifest, ps, vectors)
    store.ingest_release(manifest, ps, vectors)
    assert len(store.all_passages(manifest.release_id)) == 2
    found = store.search_dense(vectors[0], manifest.release_id, "OSFI", 2)
    assert [p.passage_id for p in found] == ["p0", "p1"]
    assert found[0].score == pytest.approx(1)
    with pytest.raises(ValueError, match="immutable"):
        store.ingest_release(manifest, ps, list(reversed(vectors)))
    broken = manifest.model_copy(update={"release_id": manifest.release_id + "-broken"})

    def fail_write(*args, **kwargs):
        raise RuntimeError("simulated interruption after release insert")

    monkeypatch.setattr(psycopg.Cursor, "executemany", fail_write)
    with pytest.raises(RuntimeError, match="interruption"):
        store.ingest_release(broken, ps, vectors)
    assert store.active_release_id() == manifest.release_id
    with pytest.raises(ValueError, match="Unknown"):
        store.manifest(broken.release_id)
