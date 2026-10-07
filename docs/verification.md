# Verification record

Local verification on 6 October 2026 used Python 3.12.15 and CPU models.

| Check | Result |
| --- | --- |
| Offline unit suite | 40 passed |
| Ruff | Passed |
| mypy | Passed, 18 source modules |
| Real pgvector | Passed: ordering, immutable releases, idempotence, rollback after interrupted ingestion |
| Real model retrieval | Passed with pinned cached embedding/reranking models |
| Frozen retrieval benchmark | 40 positive questions, five variants, zero errors; [run record](reports/retrieval-run.json) |
| Docker build | Passed locally |
| Initial Docker HTTP smoke | Passed before credential setup; dependencies loaded and `/ready` correctly reported missing paid configuration |
| Remote GitHub Actions | [Passed](https://github.com/chichichiB/Risk-Regulatory-Q-A-Assistant/actions/runs/37546511817) on main commit `8f1efc9`: offline validation and Docker build after PR #1 merged |
| Live OpenAI development smoke | Five questions, one per source, all answered with structurally valid citations; 20 provider calls, token-based cost estimate US$0.0146208 at configured standard rates (no cached-input discount or account-billing confirmation). No reference-answer or human semantic audit. [Portable run](reports/development-smoke-run.json). |
| Frozen 50-question answer benchmark | Completed at clean application commit `8f1efc9`: 50 attempted, 38/40 positive answered, 10/10 negatives declined but 8/10 exact status, zero service errors; 105/105 structural citations and same-model judged claim units, zero human audit. [Portable run](reports/answer-run.json). |
| Docker HTTP readiness | Running API returned HTTP 200 for `/health`, `/ready` with `ready=true`, and `/docs`; `tests/integration/test_compose.py` passed (1 test, 0.13 s). This HTTP check made zero provider calls. |
| Human audit | None |

Run from an activated Python 3.12 environment:

```bash
python -m ruff check .
python -m mypy src/risk_qa
python -m pytest tests/unit -q
```

The Windows sandbox blocked a test-client worker thread; the same suite passed with normal local execution. No test required a paid API call. On hosts where Application Control rejects temporary isolated build interpreters, create the environment with `uv venv --python 3.12`, install helpers with `uv pip install hatchling editables`, then `uv sync --locked --extra models --no-build-isolation`. Normal Linux/Docker installs use `uv sync --frozen --extra models`.

Explicit integrations (set environment variables with `$env:NAME='value'` in PowerShell or `export NAME=value` in Bash):

- `TEST_DATABASE_URL`: use a **disposable database**, then run `python -m pytest tests/integration/test_store.py -q`. This test changes that database's active release; do not point it at your corpus database.
- `RUN_MODEL_TESTS=1`: run `python -m pytest tests/integration/test_rerank.py -q` after ingestion. Reads the configured corpus database and local model cache.
- `TEST_API_URL=http://127.0.0.1:8000`: run `python -m pytest tests/integration/test_compose.py -q`. This passed locally (1 test, 0.13 s) with a ready API and made zero provider calls; it never sends a valid question to a paid service.
- `RUN_PAID_TESTS=1`: enables `tests/integration/test_agent_live.py`; requires an explicit local key, model, prices and budget. The five-question development smoke is a separate completed check; this specific test's status should be reported only after it is run.

The API is local-only and uses development database credentials. Its budget is a process-local estimate based on user-configured rates, not an account billing control; restarting resets it. Unknown usage after a timeout retains the reservation. The live smoke used 20 calls (estimated US$0.0146208); the frozen benchmark used 200 calls (estimated US$0.1621644); combined estimates were 220 calls and US$0.1767852. Estimates use observed token counts at configured standard rates, without cached-input discounts or account-billing reconciliation. Citation membership is structural; a same-model separate-call judge may still accept an unsupported claim. The benchmark preserved four routing mismatches: two false refusals on answerable questions and two wrong refusal statuses on negatives. Reference-answer correctness, category breakdowns, an independently authored routing challenge set and human label/answer review remain future evaluation work.

Implementation decisions: reuse the dedicated clone on a feature branch; store sparse source rows in the same PostgreSQL transaction as vectors and rebuild the small sparse index in memory; use PowerShell bookkeeping because the bundled Bash helper could not start; keep generated PDF bytes and model cache outside Git; preserve conservative errors instead of attributing a bad model rewrite to user input. The source extraction review is in [the corpus record](../corpus/extraction-review.json).
