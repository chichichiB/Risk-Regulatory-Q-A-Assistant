"""Run frozen, untuned retrieval ablations or an explicitly authorized paid answer run."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from risk_qa import prompts
from risk_qa.agent import AnswerService
from risk_qa.config import Settings
from risk_qa.contracts import AskRequest, PageRecord
from risk_qa.corpus import load_manifest
from risk_qa.eval_contracts import load_dataset, verify_dataset_hash
from risk_qa.llm import LLMClient, SemanticVerdicts, SpendLedger
from risk_qa.metrics import score_answers, score_retrieval
from risk_qa.runtime import build_retriever


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_evaluation(dataset_path: Path, mode: str, output_dir: Path) -> Path:
    settings = Settings()
    verify_dataset_hash(dataset_path)
    pages = [PageRecord.model_validate_json(line) for line in
             (settings.data_dir / "prepared/pages.jsonl").read_text(encoding="utf-8").splitlines()]
    items = load_dataset(dataset_path, pages)
    manifest = load_manifest(settings.corpus_manifest)
    if mode == "answers" and (not settings.openai_api_key or not settings.openai_model or
                               settings.max_run_usd <= 0 or
                               settings.openai_input_usd_per_million is None or
                               settings.openai_output_usd_per_million is None):
        raise ValueError("Paid evaluation requires a local key, exact model, prices and positive MAX_RUN_USD")
    retriever, release = build_retriever(settings)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"{stamp}-{mode}-{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True)
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()
    metadata = {
        "run_kind": "paid-answer-evaluation" if mode == "answers" else "real-component-retrieval",
        "started_at": datetime.now(UTC).isoformat(), "dataset": str(dataset_path),
        "dataset_sha256": digest(dataset_path), "corpus_release": release,
        "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
        "lock_sha256": digest("uv.lock"), "model_manifest": json.loads(settings.model_manifest.read_text()),
        "prompt_version": prompts.PROMPT_VERSION, "prompt_sha256": digest("src/risk_qa/prompts.py"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "sentence-transformers", "langgraph", "openai", "pgvector"]},
        "config": {"top_k_chunks": 5, "channel_candidates": 20, "rerank_candidates": 30,
                   "rrf_constant": 60, "query_rewriting": mode == "answers", "authority_source": "gold labels" if mode != "answers" else "request and agent route",
                   "max_provider_calls": settings.max_provider_calls,
                   "max_output_tokens": settings.max_output_tokens,
                   "max_run_usd": str(settings.max_run_usd)},
        "label_human_audit_count": sum(i.human_audited for i in items),
        "summary": {},
    }
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    started = time.perf_counter()
    rows = []
    if mode != "answers":
        modes = ["dense", "bm25", "tfidf", "hybrid", "reranked"] if mode == "all" else [mode]
        for variant in modes:
            ranked, errors, timings = {}, [], []
            for item in items:
                if not item.answerable:
                    continue
                before = time.perf_counter()
                error = None
                try:
                    ranked[item.item_id] = retriever.search(item.question, release,
                        None if item.authority == "comparison" else item.authority, mode=variant)
                except Exception as exc:
                    ranked[item.item_id] = []
                    error = type(exc).__name__
                    errors.append(item.item_id)
                elapsed = time.perf_counter() - before
                timings.append(elapsed)
                rows.append({"item_id": item.item_id, "variant": variant, "error": error,
                             "latency_seconds": elapsed,
                             "evidence": [p.model_dump(mode="json") for p in ranked[item.item_id]]})
            summary = score_retrieval(items, ranked).model_dump()
            summary.update({"retrieval_errors": len(errors), "error_item_ids": errors,
                            "mean_latency_seconds": sum(timings) / len(timings) if timings else None})
            metadata["summary"][variant] = summary
            print(variant, json.dumps(summary), flush=True)
            (run_dir / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    else:
        ledger = SpendLedger(settings.max_run_usd)
        clients = []
        def factory():
            client = LLMClient(settings, ledger)
            clients.append(client)
            return client
        service = AnswerService(retriever, factory, release, manifest.reference_date,
                                settings.request_timeout_seconds, settings.max_output_tokens)
        results, judgments = {}, {}
        for item in items:
            before = time.perf_counter()
            request = AskRequest(question=item.question, authority=item.authority, as_of=item.as_of)
            result = service.answer(request)
            results[item.item_id] = result
            judge_error = None
            if result.status == "answered":
                try:
                    judge = factory()
                    judged = judge.complete(SemanticVerdicts, [
                        {"role": "system", "content": prompts.VERIFY + " Independently assess this final answer; do not assume prior verification was correct."},
                        {"role": "user", "content": json.dumps({"question": item.question,
                         "claims": [c.model_dump() for c in result.claims],
                         "evidence": [p.model_dump() for p in result.citations]})},
                    ], settings.max_output_tokens)
                    judgments[item.item_id] = SemanticVerdicts.model_validate(judged).verdicts
                except Exception as exc:
                    judge_error = type(exc).__name__
            rows.append({"item_id": item.item_id, "result": result.model_dump(mode="json"),
                         "judgments": [v.model_dump() for v in judgments.get(item.item_id, [])],
                         "judge_error": judge_error, "latency_seconds": time.perf_counter() - before})
            (run_dir / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            print(item.item_id, result.status, flush=True)
        metadata["summary"]["agent_with_rewrite"] = score_answers(items, results, judgments).model_dump()
        metadata["judge_status"] = "Same-model separate-call estimate; no human audit; measures supplied claim units"
        metadata["openai_model"] = settings.openai_model
        metadata["pricing_usd_per_million"] = {"input": str(settings.openai_input_usd_per_million), "output": str(settings.openai_output_usd_per_million)}
        metadata["usage"] = [u for c in clients for u in c.usage]
        metadata["provider_attempts"] = sum(c.budget.calls for c in clients)
        metadata["spend_including_unknown_usage_reservations_usd"] = str(ledger.spent)
    metadata["duration_seconds"] = time.perf_counter() - started
    metadata["completed_at"] = datetime.now(UTC).isoformat()
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(run_dir, flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/benchmark.jsonl"))
    parser.add_argument("--mode", choices=["all", "dense", "bm25", "tfidf", "hybrid", "reranked", "answers"], default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--allow-paid", action="store_true", help="Explicitly authorize use of the configured run budget")
    args = parser.parse_args()
    if args.mode == "answers" and not args.allow_paid:
        parser.error("Answer evaluation requires --allow-paid and an explicitly configured budget")
    run_evaluation(args.dataset, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
