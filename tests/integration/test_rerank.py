import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_real_models_retrieve_source_page():
    if not os.environ.get("RUN_MODEL_TESTS"):
        pytest.skip("Set RUN_MODEL_TESTS=1 for real cached model inference")
    from risk_qa.config import Settings
    from risk_qa.embeddings import Embedder
    from risk_qa.rerank import Reranker
    from risk_qa.retrieval import Retriever
    from risk_qa.store import CorpusStore

    settings = Settings()
    store = CorpusStore(settings.database_url)
    retriever = Retriever(store, Embedder.load(Path("models/manifest.json"), Path("data/models")),
                          Reranker.load(Path("models/manifest.json"), Path("data/models")))
    found = retriever.search("Who retains accountability for activities outsourced to third parties?",
                             store.active_release_id(), "OSFI")
    assert len(found) == 5
    assert any(p.doc_id == "osfi-b10" and "accountab" in p.text.lower() for p in found)
    assert all(p.pdf_page_index >= 1 for p in found)
