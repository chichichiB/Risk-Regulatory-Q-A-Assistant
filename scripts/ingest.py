"""Index a verified frozen release. Never called implicitly by API startup."""

from pathlib import Path

from risk_qa.config import Settings
from risk_qa.corpus import chunk_pages, extract_pages, load_manifest
from risk_qa.embeddings import Embedder
from risk_qa.store import CorpusStore


def main():
    settings = Settings()
    manifest = load_manifest(settings.corpus_manifest)
    embedder = Embedder.load(settings.model_manifest, settings.data_dir / "models")
    pages = [p for s in manifest.sources for p in extract_pages(
        s, settings.data_dir / "raw" / f"{s.doc_id}.pdf")]
    passages = chunk_pages(pages, embedder.limits.count, manifest.chunk_tokens, manifest.overlap_tokens)
    vectors = embedder.encode_passages([p.text for p in passages])
    store = CorpusStore(settings.database_url)
    store.initialize()
    release = store.ingest_release(manifest, passages, vectors, pages)
    print(f"Indexed {len(passages)} passages in release {release}")


if __name__ == "__main__":
    main()
