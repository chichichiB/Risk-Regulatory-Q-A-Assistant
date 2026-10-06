"""Explicit runtime assembly; no downloads or provider calls during startup."""

import json

from risk_qa.agent import AnswerService
from risk_qa.config import Settings
from risk_qa.corpus import load_manifest
from risk_qa.embeddings import Embedder
from risk_qa.llm import LLMClient, SpendLedger
from risk_qa.rerank import Reranker
from risk_qa.retrieval import Retriever
from risk_qa.store import CorpusStore


def build_retriever(settings: Settings) -> tuple[Retriever, str]:
    manifest = load_manifest(settings.corpus_manifest)
    models = json.loads(settings.model_manifest.read_text(encoding="utf-8"))
    if {k: v["revision"] for k, v in models.items()} != manifest.model_revisions:
        raise ValueError("Model revisions differ from frozen corpus")
    store = CorpusStore(settings.database_url)
    release = store.active_release_id()
    if release != manifest.release_id or store.manifest(release) != manifest:
        raise ValueError("Active database release differs from local manifest; ingest the frozen corpus")
    embedder = Embedder.load(settings.model_manifest, settings.data_dir / "models")
    reranker = Reranker.load(settings.model_manifest, settings.data_dir / "models")
    return Retriever(store, embedder, reranker), release


def build_service(settings: Settings) -> AnswerService:
    retriever, release = build_retriever(settings)
    manifest = load_manifest(settings.corpus_manifest)
    ledger = SpendLedger(settings.max_run_usd)

    def readiness():
        reasons = []
        if not settings.openai_api_key or not settings.openai_model:
            reasons.append("Set OPENAI_API_KEY and OPENAI_MODEL locally.")
        rates = (settings.openai_input_usd_per_million, settings.openai_output_usd_per_million)
        if any(rate is None or rate < 0 for rate in rates):
            reasons.append("Configure current input and output prices for the selected model.")
        if ledger.spent >= ledger.maximum:
            reasons.append("The process spending budget is zero or exhausted.")
        try:
            if retriever.store.active_release_id() != release:
                reasons.append("The active corpus changed; restart the API to load it.")
        except Exception:
            reasons.append("The corpus database is unavailable.")
        return reasons

    return AnswerService(retriever, lambda: LLMClient(settings, ledger), release,
                         manifest.reference_date, settings.request_timeout_seconds,
                         settings.max_output_tokens, readiness)
