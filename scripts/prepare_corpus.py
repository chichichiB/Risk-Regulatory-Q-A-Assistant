"""Download approved sources explicitly, then freeze or verify their provenance."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pypdf
from pypdf import PdfReader

from risk_qa.contracts import CorpusManifest, SourceSpec
from risk_qa.corpus import chunk_pages, extract_pages, file_hash, load_manifest, verify_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    root = args.snapshot_dir
    root.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path("corpus/sources.json").read_text(encoding="utf-8"))
    manifest_path = Path("corpus/manifest.json")
    existing = load_manifest(manifest_path) if manifest_path.exists() else None
    if args.download:
        for source in config["documents"]:
            dest = root / f"{source['id']}.pdf"
            if dest.exists():
                continue
            response = httpx.get(source["pdf_url"], follow_redirects=True, timeout=90)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF") or len(response.content) > 30_000_000:
                raise ValueError("Expected an official PDF smaller than 30 MB")
            temp = dest.with_suffix(".download")
            temp.write_bytes(response.content)
            if existing:
                verify_snapshot(next(s for s in existing.sources if s.doc_id == source["id"]), temp)
            temp.replace(dest)
            print(f"Downloaded {source['id']}: {len(response.content)} bytes")
    model_path = Path("models/manifest.json")
    if args.download_models:
        from huggingface_hub import HfApi, snapshot_download

        if model_path.exists():
            models = json.loads(model_path.read_text(encoding="utf-8"))
        else:
            ids = {"embedding": "BAAI/bge-small-en-v1.5",
                   "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2"}
            models = {k: {"id": v, "revision": HfApi().model_info(v).sha} for k, v in ids.items()}
            model_path.parent.mkdir(exist_ok=True)
            model_path.write_text(json.dumps(models, indent=2) + "\n", encoding="utf-8")
        for model in models.values():
            snapshot_download(model["id"], revision=model["revision"], cache_dir="data/models",
                              allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
                              ignore_patterns=["onnx/*", "openvino/*"])
            print(f"Cached {model['id']} at {model['revision']}")
    if args.freeze:
        if existing:
            raise ValueError("A manifest already exists; refusing to replace the frozen release")
        models = json.loads(model_path.read_text(encoding="utf-8"))
        sources = [SourceSpec(
            doc_id=s["id"], title=s["title"], authority=s["authority"],
            source_url=s["source_url"], pdf_url=s["pdf_url"],
            sha256=file_hash(root / f"{s['id']}.pdf"),
            page_count=len(PdfReader(root / f"{s['id']}.pdf").pages),
            retrieved_at=datetime.now(UTC).isoformat(), effective_from=s.get("effective_from"),
        ) for s in config["documents"]]
        payload = dict(reference_date="2026-10-06", sources=[s.model_dump() for s in sources],
                       extraction_version=f"pypdf-{pypdf.__version__}-layout-v1",
                       chunk_tokens=320, overlap_tokens=40,
                       model_revisions={k: v["revision"] for k, v in models.items()})
        release = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
        existing = CorpusManifest(release_id=release, **payload)
        manifest_path.write_text(existing.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if args.verify_only or args.freeze:
        manifest = load_manifest(manifest_path)
        pages = [page for s in manifest.sources for page in extract_pages(s, root / f"{s.doc_id}.pdf")]
        from risk_qa.tokenization import TokenLimits

        limits = TokenLimits(model_path, Path("data/models"))
        chunks = chunk_pages(pages, limits.count, manifest.chunk_tokens, manifest.overlap_tokens)
        prepared = Path("data/prepared")
        prepared.mkdir(exist_ok=True)
        for name, records in [("pages", pages), ("passages", chunks)]:
            (prepared / f"{name}.jsonl").write_text(
                "\n".join(p.model_dump_json() for p in records) + "\n", encoding="utf-8")
        print(f"Verified {len(manifest.sources)} PDFs, {len(pages)} pages, {len(chunks)} chunks; release {manifest.release_id}")


if __name__ == "__main__":
    main()
