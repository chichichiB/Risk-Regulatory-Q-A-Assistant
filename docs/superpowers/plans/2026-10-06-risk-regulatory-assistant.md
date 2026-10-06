# Risk Regulatory Q&A Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. **Recommended for this project:** native implementation in this chat, because the ingestion, retrieval, graph and evaluation tasks share evolving interfaces; obtain one independent whole-branch code review before delivery. The user's Sol request concerned technical documentation, not code delegation.

**Goal:** Deliver a runnable local portfolio system that answers English questions from five frozen official OSFI/Basel PDFs with verifiable page citations, bounded refusals, and a reproducible 50-item evaluation.

**Architecture:** A versioned corpus manifest and preserved PDF snapshots feed one-page passages, PostgreSQL/pgvector dense search and an atomic BM25 index. A retrieval service fuses and reranks candidates; a bounded LangGraph service generates verified atomic claims, and FastAPI exposes the result. A separate evaluation runner measures retrieval and answer behavior against frozen page/span labels.

**Tech stack:** Python 3.12 CPU container, FastAPI, Pydantic, LangGraph, Sentence Transformers/Hugging Face, `BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L6-v2`, PostgreSQL/pgvector, `rank_bm25`, scikit-learn TF-IDF baseline, OpenAI Responses with structured outputs, `pytest`, Docker Compose, GitHub Actions. Use `uv.lock` to lock actual resolved dependency versions; pin actual model revisions during execution, not invented version identifiers.

**Spec:** `docs/design.md` in the target repository (`work/repository/docs/design.md` in this workspace). Read the approved spec before implementation and use its details when a task below refers to a policy.

**Status:** Implemented and locally verified, 6 October 2026. See docs/verification.md for measured checks. Remote CI passed offline validation and Docker build on commit `6bc150a`; paid OpenAI validation remains unrun.

## Global constraints

- English documentation/questions/results; five sources are OSFI E-21, B-10, B-13, Corporate Governance, and the 15 December 2019 in-force Basel SRP30 chapter. Freeze the 6 October 2026 corpus reference and verify official PDF bytes and metadata at ingestion. Raw PDFs, local snapshot archive and Hugging Face cache remain gitignored.
- One immutable `corpus_release` joins SQL passages, dense vectors and sparse index. Download-time URL identity alone is insufficient: a changed official generated PDF SHA-256 fails closed, never silently rebaselines a benchmark. Preserve exact snapshots locally for reproduction; a fresh download is accepted only when its hash matches.
- Every chunk stays within one physical PDF page and one version. Its evidence text must respect the selected model's maximum token length: split before embedding/reranking or fail with an explicit extraction/index error; never silently truncate support text.
- Exact pgvector cosine search first; query-only BGE prefix `Represent this sentence for searching relevant passages: `; BM25 production sparse channel, TF-IDF baseline, rank fusion and cross-encoder. First five ranked **chunks** define page metrics even if multiple chunks map to one page.
- Graph: at most one rewrite and one repair, eight OpenAI attempts per request including timeouts; disable implicit SDK retries and enforce token/deadline limits plus conservative paid-run reservation. Construct `answer_text` only from verified claims. Retrieved text is data, never an instruction. No shell, browser, external-write tools or open-ended loops.
- Benchmark: 50 held-out questions (40 answerable, eight primary per source; 10 refusal/version/out-of-scope) plus separate 15 development questions (three primary per source). AI-authored labels must say so. Page recall@5, hit@5, complete-page-coverage@5 and exact-span complete-evidence@5 are distinct. Judge-assisted faithfulness is an estimate until human-audited.
- First version is local Swagger UI, no custom frontend/auth/cloud. Deterministic offline tests require no key, network, model cache or database. Real HF/pgvector integration and paid OpenAI end-to-end tests are explicit separate commands. Pending key/spend affects only paid runs.

Implementation defaults within the approved design: stripped question length 1–2,000 characters and no more than 128 tokens under either model tokenizer; page chunks at most 320 tokens under either tokenizer with up to 40 tokens of same-page overlap. Check the full reranker pair against its actual model limit. Oversized questions produce an input-validation error before any paid call. Use 120 seconds as the overall request deadline, 30 seconds as each provider timeout, and at most 1,500 output tokens per call. These are development settings recorded in the run configuration, not measured optimums. `pytest` discovers only offline tests by default; integration files require an explicit path and fixture configuration. Use a Python 3.12 project environment for commands below, never the host's Python 3.14 by accident.

## File ownership and fixed interfaces

All paths below are relative to `work/repository/`. Create `src/risk_qa/` as the package. The file owner is the task named in the right column; later tasks may consume its public interfaces without reimplementing its responsibility.

| Files | Owner / responsibility |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.gitignore`, `.env.example`, `src/risk_qa/__init__.py`, `src/risk_qa/contracts.py`, `src/risk_qa/config.py` | Task 1: dependencies and shared typed records/configuration. |
| `corpus/manifest.json`, `models/manifest.json`, `src/risk_qa/corpus.py`, `src/risk_qa/tokenization.py`, `scripts/prepare_corpus.py` | Task 2: official snapshots, pinned tokenizers, page extraction and provenance. |
| `src/risk_qa/store.py`, `src/risk_qa/embeddings.py`, `src/risk_qa/sparse.py`, `scripts/ingest.py`, initial database service in `compose.yaml` | Task 3: transactional pgvector persistence and matching immutable BM25 release. |
| `src/risk_qa/retrieval.py`, `src/risk_qa/rerank.py` | Task 4: filtered dense/sparse retrieval, RRF, rerank, candidate traces. |
| `src/risk_qa/citations.py` | Task 5: deterministic citation membership/locator checks and verified-claim rendering. |
| `src/risk_qa/agent.py`, `src/risk_qa/llm.py`, `src/risk_qa/prompts.py` | Task 6: bounded LangGraph and structured provider calls. |
| `src/risk_qa/api.py`, `Dockerfile`, API service added to `compose.yaml` | Task 7: HTTP contract and local services. |
| `eval/dev.jsonl`, `eval/benchmark.jsonl`, `src/risk_qa/eval_contracts.py`, `src/risk_qa/metrics.py`, `scripts/evaluate.py` | Task 8: labeled sets, fair retrieval comparisons and honest reports. |
| `.github/workflows/ci.yml`, `README.md`, `docs/evaluation.md` | Task 9: reproducible commands, CI and delivery evidence. |

Task 1 defines the shared records in `contracts.py`: `SourceSpec`, `CorpusManifest`, `PageRecord`, `PassageRecord`, `EvidencePassage`, `Claim(text, citation_ids)`, `ClaimVerdict(claim_index, status, evidence_ids, rationale)`, `AskRequest(question, authority?, as_of?)`, `AnswerResult(status, answer_text?, claims, citations, corpus_release, trace_id, reason?)`, `ReadinessReport(ready, reasons)` and `VerificationReport(passed, claims, verdicts, errors)`. A verdict's status is `supported`, `unsupported` or `uncertain`; missing, duplicated or out-of-range claim indices fail verification. `page_id := (version_id, pdf_page_index)`; intervals are half-open character offsets into preserved extracted page text. Authority is `OSFI`, `BCBS`, or explicit `comparison`; the last permits retrieving both sources without treating them as interchangeable.

The owning task implements these signatures; Task 1 does not create empty function bodies:

- Task 2: `load_manifest(path: Path) -> CorpusManifest`; `extract_pages(spec: SourceSpec, pdf_path: Path) -> list[PageRecord]`; `chunk_pages(pages: list[PageRecord], token_counter: Callable[[str], int], max_tokens: int = 320, overlap_tokens: int = 40) -> list[PassageRecord]`. The injected counter returns the larger token count from the two pinned tokenizers; offline fixtures inject a fake. Keep offsets against the frozen page text.
- Task 3: `CorpusStore.ingest_release(manifest: CorpusManifest, passages: list[PassageRecord], vectors: list[list[float]]) -> str`; `CorpusStore.active_release_id() -> str`; `CorpusStore.search_dense(query_vector: list[float], release_id: str, authority: str | None, limit: int) -> list[EvidencePassage]`; `SparseIndex.search(query: str, release_id: str, authority: str | None, limit: int) -> list[EvidencePassage]`.
- Task 4: `Retriever.search(question: str, release_id: str, authority: str | None, limit: int = 5) -> list[EvidencePassage]`. Pass `None` only for an explicit comparison; otherwise supply the routed authority.
- Task 5: `verify_claims(claims: list[Claim], evidence: list[EvidencePassage], semantic_verdicts: list[ClaimVerdict], required_authorities: set[str]) -> VerificationReport`.
- Task 6: `AnswerService.answer(request: AskRequest) -> AnswerResult`; `AnswerService.readiness() -> ReadinessReport`, with no provider call for readiness.
- Task 7: `create_app(service: AnswerService) -> FastAPI`.
- Task 8 defines `GoldItem`, `RetrievalMetrics` and `AnswerMetrics` in `eval_contracts.py`, and implements `score_retrieval(items: list[GoldItem], ranked: dict[str, list[EvidencePassage]]) -> RetrievalMetrics` plus `score_answers(items, results, judgments) -> AnswerMetrics` with those records. Undefined rates are `None` with denominator zero.

## Review Focus

1. **Regenerated official PDF with same URL:** Task 2 test must reject a SHA mismatch and leave the active corpus untouched.
2. **Dense and BM25 releases diverge after interrupted ingestion:** Task 3 test must prove queries see either the previous complete release or the new complete release, never a mixture.
3. **Long page/chunk near model token limit:** Task 2/3 tests must prove support text is fully indexed or explicitly rejected, not truncated invisibly.
4. **Retrieved passage contains “ignore instructions” or requests secrets:** Task 6 test must prove the graph treats it as evidence data and cannot trigger tools or leak configuration.
5. **Claim cites a valid page but wrong passage, or citation ID outside top evidence:** Task 5/6 tests must reject membership/locator mismatch and prevent unchecked text from reaching `answer_text`.

---

### Task 1: Shared contracts and deterministic test base

**Files:** Create `pyproject.toml`, `uv.lock`, `.gitignore`, `.env.example`, `src/risk_qa/__init__.py`, `src/risk_qa/contracts.py`, `src/risk_qa/config.py`, `tests/unit/test_contracts.py`. **Interfaces:** Produces shared records in the ownership section. `Settings` loads the active release, model IDs/revisions and bounded budgets from environment without reading secrets at import time. Separate optional heavy-model dependencies from offline test dependencies; configure Ruff, mypy and `pytest` in `pyproject.toml`.

- [x] **Red:** After creating the isolated Python 3.12 development environment and test dependency, write `test_blank_question_is_rejected` and an otherwise-valid answer fixture whose empty claims and nonempty free prose fail validation. The former includes `with pytest.raises(ValidationError): AskRequest(question=" ")`; the latter must supply all other required response fields so the assertion tests the intended invariant. Run `python -m pytest tests/unit/test_contracts.py -q`; expect failures because contracts are absent.
- [x] **Green:** Add only the shared records, enums and settings needed by these assertions. Resolve and lock supported Python 3.12 package versions from the actual environment; Task 2 pins tokenizer/model revision identifiers and Task 3 verifies downloaded weights against them. Do not make offline test imports download weights. Run the same test; expect pass.
- [x] **Verify/commit:** Run `python -m pytest tests/unit -q` with network and key unavailable; expect pass. Commit `chore: establish typed contracts and locked environment` with only Task 1 files.

### Task 2: Frozen official corpus and page provenance

**Files:** Create `corpus/manifest.json`, `models/manifest.json`, `src/risk_qa/corpus.py`, `src/risk_qa/tokenization.py`, `scripts/prepare_corpus.py`, `tests/unit/test_corpus.py`, `tests/fixtures/pages/`. **Interfaces:** `load_manifest`, `extract_pages`, `chunk_pages` as defined above; `verify_snapshot(spec: SourceSpec, path: Path) -> None` compares bytes to pinned SHA-256 before parsing; `validate_passage_offsets(passage: PassageRecord, page: PageRecord) -> None` checks offsets and page identity. Each `PassageRecord` carries version, one-based physical page, extracted-page half-open offsets, optional printed label, heading, paragraph and official PDF URL. `tokenization.py` implements a local-cache-only adapter for the pinned model tokenizers.

- [x] **Red:** Fixture tests assert exactly five manifest IDs/URLs, one-page chunk boundaries, preserved page offsets, detected empty/table extraction flags, and rejection of a same-URL changed PDF hash. A long-page fixture asserts `chunk_pages(..., max_tokens=N)` returns pieces individually at or below the tokenizer limit without losing indexed text; if a table cannot be safely split, it raises a named extraction error. Include this focused assertion:

  ```python
  with pytest.raises(SnapshotHashMismatch):
      verify_snapshot(source_spec, changed_pdf_with_same_url)
  ```

  Run `python -m pytest tests/unit/test_corpus.py -q`; expect fail.
- [x] **Green:** Download the five official PDFs to gitignored `data/raw/`, capture hash/retrieval metadata and observed page counts, inspect extraction quality, then freeze the manifest and local snapshot archive. Use `pypdf` for extraction, inspecting source pages visually wherever tables, footnotes or extraction flags need review. Record repairs with their provenance. Resolve each model to an actual immutable Hugging Face commit in `models/manifest.json`, download its tokenizer explicitly, and implement page-constrained chunking using the injected counter for both models. A repeated preparation with matching bytes succeeds; changed bytes require an explicit new corpus release and benchmark review, not overwrite.
- [x] **Verify/commit:** Run the unit test plus `python scripts/prepare_corpus.py --verify-only --snapshot-dir data/raw`; expect five verified hashes and page-map checks. Preserve the archive locally, keep PDFs out of Git, and commit source/manifest/tests as `feat: freeze official corpus and page provenance`.

### Task 3: Atomic dense and BM25 corpus release

**Files:** Create `src/risk_qa/store.py`, `src/risk_qa/embeddings.py`, `src/risk_qa/sparse.py`, `scripts/ingest.py`, database-only `compose.yaml`, `tests/unit/test_index_release.py`, `tests/integration/test_store.py`. **Interfaces:** `CorpusStore.ingest_release`, `CorpusStore.active_release_id`, `CorpusStore.search_dense`, `SparseIndex.search`; `Embedder.encode_query(text: str) -> list[float]` applies the BGE prefix, while `Embedder.encode_passages(texts: list[str]) -> list[list[float]]` does not. Both reject silent truncation and incompatible 384-dimensional vectors.

- [x] **Red:** Unit tests with fake embedder/store assert document-only text has no query prefix, query text does; repeated same-hash ingestion is idempotent. In `test_failed_release_does_not_replace_active_corpus`, inject a failure during sparse publication and assert `store.active_release_id() == previous_release_id`; query both channels and assert every result retains that release. Add a real pgvector fixture test for cosine ordering and foreign-key provenance. Run `python -m pytest tests/unit/test_index_release.py -q`; expect fail.
- [x] **Green:** Build SQL tables for versions, pages, passages and vectors; stage a full release, build a deterministic BM25 artifact keyed by release hash, verify record counts/IDs, then atomically switch the active release pointer only after both are complete. Each query captures that pointer once and passes it through all retrieval stages. Publish immutable sparse artifacts before the database pointer transaction; retain the old artifacts so interrupted or in-flight queries remain consistent. Refuse a release with a missing sparse artifact. Use exact pgvector cosine search and verify weight revisions against `models/manifest.json`. Make `scripts/ingest.py` an explicit one-shot command, never an API startup side effect. Start the existing Docker Desktop installation if needed and use `docker compose up -d db` for the real database checks; do not remove pre-existing user volumes.
- [x] **Verify/commit:** Run unit tests offline; run `python -m pytest tests/integration/test_store.py -q` only with disposable PostgreSQL/pgvector and cached model weights. Expect matching active release IDs and stable second ingestion. Commit `feat: persist atomic dense and sparse corpus release`.

### Task 4: Hybrid retrieval and bounded reranking

**Files:** Create `src/risk_qa/retrieval.py`, `src/risk_qa/rerank.py`, `tests/unit/test_retrieval.py`, `tests/integration/test_rerank.py`. **Interfaces:** `Retriever.search` above and `Reranker.rank(question: str, candidates: list[EvidencePassage], limit: int) -> list[EvidencePassage]`. Expose a trace containing dense/BM25 ranks, RRF score, rerank score, passage IDs and release ID; no answer judgment occurs here.

- [x] **Red:** Test RRF with ranks 1 and 2 computes `1/(60+rank)` per present channel, stable ties by `passage_id`, duplicate removal by ID, authority filtering, at most 20 per channel/30 reranked/five returned, and rejection of a candidate from a different release. Test TF-IDF as a separate baseline adapter, not a score mixed into production RRF. Run `python -m pytest tests/unit/test_retrieval.py -q`; expect fail.
- [x] **Green:** Implement dense/BM25 candidate gathering, direct section-identifier match, RRF fusion, and pinned MiniLM cross-encoder reranking. Check each pair's token length; split at ingestion or reject overlong evidence rather than silently truncating the sentence needed for citation. Keep release and authority filters through every stage.
- [x] **Verify/commit:** Run unit tests, then real model integration with cached weights via `python -m pytest tests/integration/test_rerank.py -q`. Commit `feat: retrieve and rerank frozen evidence`.

### Task 5: Citation membership and verified rendering

**Files:** Create `src/risk_qa/citations.py`, `tests/unit/test_citations.py`. **Interfaces:** `verify_claims(claims, evidence, semantic_verdicts, required_authorities) -> VerificationReport`; `render_verified(report: VerificationReport) -> str` returns answer text only when all assessable claims pass structural and semantic checks. The latter accepts verifier decisions but never accepts an arbitrary draft prose field. Citation URLs and page locators come from immutable evidence records, not model-written metadata.

- [x] **Red:** Test a citation outside the current five evidence IDs, an ID with a mismatched version/page/official URL, a real page but wrong passage, and an empty citation list. Assert all fail. Supply a fake semantic verdict marking a numeric claim with a missing qualifier `unsupported`, and assert it cannot be rendered; do not pretend deterministic membership tests can discover factual entailment. Assert a two-source comparison with one missing source fails its explicit support requirement. One focused check is:

  ```python
  report = verify_claims(
      [Claim(text="unsupported", citation_ids=["other-release:p1"])],
      top_five, semantic_verdicts=[], required_authorities={"OSFI"},
  )
  assert report.passed is False
  assert render_verified(report) == ""
  ```

  Run `python -m pytest tests/unit/test_citations.py -q`; expect fail.
- [x] **Green:** Implement deterministic membership/provenance/locator checks and explicit per-claim semantic verdict input. Require exactly one verdict for every claim and reject duplicates, unknown claim indices, foreign evidence IDs or missing required authorities. A model judge may propose `supported/unsupported/uncertain` with rationale, but `uncertain` fails closed and human audit status is reported separately. Automated span containment proves only text coverage, not factual entailment. Render all claim text only when the full report passes; otherwise return an empty string for the graph to repair or refuse.
- [x] **Verify/commit:** Run citation unit tests offline and commit `feat: validate citations and render verified claims`.

### Task 6: Bounded LangGraph answer service

**Files:** Create `src/risk_qa/agent.py`, `src/risk_qa/llm.py`, `src/risk_qa/prompts.py`, `tests/unit/test_agent.py`, `tests/integration/test_agent_live.py`. **Interfaces:** `AnswerService.answer(request: AskRequest) -> AnswerResult` and `readiness() -> ReadinessReport`; inject `Retriever`, citation verifier and `LLMClient.complete(schema: type[BaseModel], messages: list[dict[str, str]], max_output_tokens: int) -> BaseModel` so offline tests use fakes. Define the stage schemas in `llm.py`: `RouteDecision(status, authority, reason)`, `EvidenceGrade(adequate, evidence_ids, reason)`, `QueryRewrite(search_query)`, `DraftAnswer(claims)` and `SemanticVerdicts(verdicts)`. A rewrite supplies retrieval text only; immutable request authority/date are never replaced. `LLMClient` makes one provider attempt per call with SDK retries disabled. `CallBudget.reserve(input_tokens: int, max_output_tokens: int) -> Decimal` reserves the conservative dollar cost and increments attempts before dispatch; `settle(reserved: Decimal, observed_cost: Decimal | None) -> None` refunds only a proven unused amount. A timeout with unknown usage retains its reservation. A run-level ledger covers the entire paid evaluation, not just one question; run the first evaluation sequentially to keep reservation accounting explicit.

- [x] **Red:** Fake-client graph tests cover out-of-scope and ambiguous authority/date routes; zero evidence refusal; one rewrite and one repair maximum; an eighth provider attempt allowed but a ninth rejected; timeout counted; invalid structured output handled; provider/index failures become `service_error`. Test that evidence is placed only in the data payload, that no secret reaches any provider prompt, and that adversarial provider outputs cannot bypass enforced citation/authority/budget rules. These tests establish application boundaries, not a proof that a real model resists every prompt injection. Add missing/duplicate claim verdict cases and budget exhaustion across two consecutive questions. A verifier rejection must prevent unchecked prose or a wrong-citation claim from entering `answer_text`. Run `python -m pytest tests/unit/test_agent.py -q`; expect fail.
- [x] **Green:** Implement typed LangGraph nodes `validate -> route -> retrieve -> grade -> generate -> verify -> terminal`, with bounded rewrite and repair edges. Require provider outputs to fit structured schemas, enforce evidence ID allowlists in code, and treat evidence as quoted data. Apply token caps and deadline. Reserve conservatively before paid calls; if price data is unavailable, do not run automated paid evaluation. Keep model ID configurable and prompts versioned.
- [x] **Verify/commit:** Run offline graph tests and a graph topology/recursion test. Only with approved API access/budget run `python -m pytest tests/integration/test_agent_live.py -q`; record model ID, calls, tokens, latency and spend. Commit `feat: add bounded evidence-grounded answer graph` independently of whether paid validation is available; document that status accurately.

### Task 7: FastAPI and local Docker demonstration

**Files:** Create `src/risk_qa/api.py`, `Dockerfile`, `tests/unit/test_api.py`, `tests/integration/test_compose.py`; extend Task 3's `compose.yaml` with the API service. **Interfaces:** `create_app(service: AnswerService) -> FastAPI` exposes `POST /ask` using `AskRequest`/`AnswerResult` and `GET /ready` calling `AnswerService.readiness()` without an LLM call. Answer/refusal outcomes return HTTP 200; invalid input returns 422; unavailable dependencies/provider failures return 503 with sanitized `service_error`. No route owns retrieval or provider logic.

- [x] **Red:** FastAPI `TestClient` tests assert OpenAPI exposes `/ask`, blank/overlong question returns validation error, mocked answer/refusal/status serializes correctly, readiness never calls the LLM, and no trace secrets appear in response. Include `assert client.post("/ask", json={"question": " "}).status_code == 422` in `test_blank_question_returns_422`, plus `assert fake_llm.call_count == 0` after a readiness request. Run `python -m pytest tests/unit/test_api.py -q`; expect fail.
- [x] **Green:** Implement dependency-injected routes, Python 3.12 CPU image, PostgreSQL/pgvector health check and API service in Compose. Provide a one-shot ingestion invocation; startup fails clearly when corpus/index is absent instead of fetching live PDFs. Use Swagger UI for interaction; no custom frontend, auth or cloud deployment.
- [x] **Verify/commit:** Run unit tests; when Docker engine is available run `docker compose build`, `docker compose up -d`, `python -m pytest tests/integration/test_compose.py -q`, then stop the test stack. If the engine remains stopped, report build/runtime as unverified, not passed. Commit `feat: expose local API and compose stack`.

### Task 8: Frozen labels, metrics and report runner

**Files:** Create `eval/dev.jsonl`, `eval/benchmark.jsonl`, `src/risk_qa/eval_contracts.py`, `src/risk_qa/metrics.py`, `scripts/evaluate.py`, `tests/unit/test_metrics.py`, `docs/evaluation.md`. **Interfaces:** `score_retrieval(items: list[GoldItem], ranked: dict[str, list[EvidencePassage]]) -> RetrievalMetrics`; `score_answers(items, results, judgments) -> AnswerMetrics`; `run_evaluation(dataset_path: Path, mode: str, output_dir: Path) -> Path` writes per-question JSONL and a run manifest. `GoldItem` carries page IDs, exact span intervals, minimal supporting page and span sets, answerability and AI-authorship/audit metadata. Keep retrieval and answer-score records separate.

- [x] **Red:** On a fixture with two gold pages and five ranked chunks that hit only one, assert page recall `0.5`, hit `1`, complete-page-coverage `0`, complete-evidence `0`. On a second fixture where chunks from the correct page omit the gold span, assert page coverage `1` but complete-evidence `0`; duplicate-page chunks do not open extra slots. Negative items do not enter the 40-answerable retrieval denominator. Assert a metric with zero assessed claims is `null`, not `0` or `100%`; service-error queries are counted in attempted queries and a separate error bucket, not quietly dropped or scored as refusals. Include:

  ```python
  assert retrieval_scores.complete_page_coverage_at_5 == 1
  assert retrieval_scores.complete_evidence_at_5 == 0  # gold span outside retrieved intervals
  assert answer_scores.claim_faithfulness is None  # zero assessed claims
  ```

  Run `python -m pytest tests/unit/test_metrics.py -q`; expect fail.
- [x] **Green:** Draft and source-check 15 separate dev and 50 held-out items; validate counts, five-source primary balance, page/span IDs and answerability. Record AI-authored provenance and any actual human audit coverage. Implement macro page metrics, exact-span union coverage, refusal metrics, citation checks and judge-assisted claim faithfulness labeling. Exact-span coverage is deterministic, but semantic support remains a judge-assisted estimate unless audited by a human. Emit `null` plus denominator `0` for undefined percentages, with service errors separate and included in total attempted queries. Freeze dataset hash before tuning. Compare dense, BM25, TF-IDF, hybrid and reranked modes with query rewriting off; run rewriting as a separately named end-to-end variant. Persist git/corpus/model/prompt/config hashes, calls/cost and predictions. No claimed 88→94 recall or 95% faithfulness before a real run.
- [x] **Verify/commit:** Run metric tests offline, then a fixture evaluation to verify output schema. Run real-component retrieval benchmark only when snapshots/weights/database are ready; paid answer evaluation only after approved API budget. Commit `feat: add frozen benchmark and reproducible metrics`, clearly distinguishing fixture, real-component and paid-result statuses.

### Task 9: CI, user documentation and final evidence

**Files:** Create `.github/workflows/ci.yml`; modify `README.md`; finish `docs/evaluation.md`; add `tests/unit/test_docs_contract.py` only if a command/schema example would otherwise drift. **Interfaces:** Document the exact commands for environment setup, snapshot verification, one-shot ingestion, Compose API/Swagger use, offline tests, real-component tests and explicit paid evaluation. Record where the preserved snapshot archive lives and how hashes are checked; do not imply a fresh network download always reproduces historical bytes.

- [x] **Red:** A smoke check against a clean checkout must identify any undocumented env var, absent snapshot archive, model cache or Docker dependency before executing a paid call. Run the README commands in a clean local environment where available and record failures honestly.
- [x] **Green:** CI runs lockfile install, lint/type checks, offline `pytest`, and Docker build; it never requires secrets for pull requests. Keep real-component and paid jobs explicit and separate. README explains architecture, source manifest, metrics, label provenance, limitations and precise test status; only measured results appear. Provide local artifact paths for the corpus manifest, run manifest and per-question outputs. Do not deploy or publish a service.
- [x] **Verify/commit:** Run `python -m ruff check .`, `python -m mypy src/risk_qa`, `python -m pytest tests/unit -q`, real-component checks and `docker compose build` when the engine is running; inspect CI results remotely if a push is authorized. Distinguish **local pass**, **remote pass**, **not run** and **blocked by key/engine**. Request an independent final code review focused on the five Review Focus cases and spec coverage, fix findings, rerun affected gates, then commit `docs/ci: document and verify local reproduction`.

## Spec coverage and execution handoff

Tasks 1–3 implement immutable provenance and corpus release; Task 4 implements hybrid retrieval and reranking; Tasks 5–6 implement citation/claim checks, refusal, routing and provider budgets; Task 7 implements the local API; Task 8 implements the frozen evaluation and honest metric labels; Task 9 implements reproducibility and CI. The five Review Focus cases each have a named failing-test gate above. Each commit is a reviewable boundary; any changes to shared contracts require updating consuming tasks' tests in the same commit.

The user approved this plan and native execution. Local implementation and the independent whole-branch review are complete. Conditional paid checks were not run because API setup and spending authorization are pending. The user authorized publication: branch `feat/regulatory-rag` is pushed and PR #1 is open against `main`. Reference-answer correctness and per-category answer aggregates are deferred; the delivered evaluation documents this limitation explicitly.
