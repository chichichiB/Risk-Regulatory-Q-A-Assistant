# Risk Regulatory Q&A Assistant

A local, English-language portfolio project for question answering over five frozen OSFI/Basel documents. It combines dense and exact-term retrieval, reranks evidence, and uses a bounded LangGraph workflow to answer with source citations or refuse when support is insufficient. It is a technical demonstration, not advice about an institution's regulatory obligations.

## Corpus and reproduction

The five official PDF renditions are OSFI [E-21](https://www.osfi-bsif.gc.ca/en/print/pdf/node/2332), [B-10](https://www.osfi-bsif.gc.ca/en/print/pdf/node/567), [B-13](https://www.osfi-bsif.gc.ca/en/print/pdf/node/570), [Corporate Governance](https://www.osfi-bsif.gc.ca/en/print/pdf/node/592), and [Basel Framework SRP30](https://www.bis.org/committees/bcbs/basel-framework/81535/chapter.pdf). The frozen release contains **110 PDF pages and 181 one-page chunks**. [corpus/manifest.json](corpus/manifest.json) records the 6 October 2026 reference date, URLs, page counts, PDF SHA-256 values, extraction settings and release ID `fa82be6139b9eb4af71bce55`. [models/manifest.json](models/manifest.json) pins both Hugging Face model revisions. The source snapshot date does not establish current legal applicability. OSFI and Basel are separate authorities; comparisons require evidence from each.

Raw PDFs and model files are under gitignored `data/`. **Preserve the PDF bytes used for a reported run.** Official generated PDFs can change at the same URL; a new file with a hash different from the frozen manifest is rejected. A fresh network download is not guaranteed to reproduce this release. After verification, create a local source archive and save a copy outside the repository:

```bash
tar -czf data/corpus-snapshot-fa82be6139b9eb4af71bce55.tar.gz -C data raw
```

Restore by extracting into `data/` so the five files are again in `data/raw/`, then run the verification command below:

```bash
tar -xzf path/to/corpus-snapshot-fa82be6139b9eb4af71bce55.tar.gz -C data
python -m scripts.prepare_corpus --download-models --verify-only
```

The archive is gitignored; the manifest alone cannot reconstruct changed source bytes. The second command fetches pinned models if absent, then verifies the restored PDF hashes and prepares chunks. With models already cached, omit `--download-models` for a network-free check.

## Architecture

```text
verified PDFs -> page-local chunks -> exact pgvector search + BM25
                                    -> reciprocal rank fusion -> cross-encoder
                                    -> LangGraph route / grade / answer / verify
                                    -> FastAPI response or refusal
```

`BAAI/bge-small-en-v1.5` supplies 384-dimensional vectors; only queries receive its documented retrieval prefix. Exact pgvector cosine search avoids an approximate-index recall trade-off in this small corpus. BM25 helps with identifiers and exact wording; TF-IDF is a separate baseline. Queries containing digit-bearing tokens also receive an identifier channel: matching passages are ordered by passage ID, capped at 20, and included in fusion. Reciprocal rank fusion combines ranks without treating their raw scores as equivalent. `cross-encoder/ms-marco-MiniLM-L6-v2` reranks at most 30 candidates and returns at most five chunks. Token limits are checked to prevent silent evidence truncation.

The LangGraph flow can rewrite once and repair once, with at most eight provider attempts per request. It treats retrieved passages as untrusted data. Code checks citation membership in the current evidence set; a separate model verdict estimates semantic support. `answer_text` is built from verified atomic claims. A citation on the right page still may not support a claim; the evaluation distinguishes page coverage from exact supporting text. Provider/dependency failures are service errors, not evidence-based refusals.

## Run locally

Requirements: Python 3.12, `uv`, Docker with a running engine, and network access for the initial pinned model/PDF download. Normal Docker/Linux installation uses the checked-in lockfile:

```bash
uv sync --locked --extra models
```

On a Windows host restricted by Application Control, the local environment needed build helpers installed into the Python 3.12 virtual environment, followed by `uv sync --extra models --no-build-isolation`; this is a host-specific workaround, not part of the container build. Activate the environment with `source .venv/bin/activate` (Bash) or `.venv\Scripts\Activate.ps1` (PowerShell). If activation is restricted, prefix each `python` command below with `uv run --no-sync`. From the repository root with the environment active:

```bash
python -m scripts.prepare_corpus --download --download-models --verify-only
docker compose up -d db
python -m scripts.ingest
python -m scripts.evaluate --mode all
docker compose up -d --build api
```

The first command checks all five PDF hashes and prepares extracted pages/chunks; repeat with `--verify-only` when preserved PDFs and models are present. `scripts.ingest` indexes the verified release into pgvector and is idempotent for the same release; BM25 is built from that active release when retrieval starts. The evaluation command runs real local retrieval modes and writes timestamped `reports/.../run.json` and per-question predictions. The API container mounts `./data` read-only, listens on `127.0.0.1:8000`, and uses one worker because its spend ledger is process-local. Open [Swagger UI](http://127.0.0.1:8000/docs); `/health` checks liveness and `/ready` returns 503 until dependencies and local API configuration are ready. `POST /ask` is the question endpoint. The Docker image built successfully and the local Compose API passed its HTTP smoke test. These checks did not invoke OpenAI.

For answer generation, **copy `.env.example` to `.env` on your own computer**, then put your API key only in `.env`. The tracked `.env.example` is a public template and must keep `OPENAI_API_KEY` empty. `.env` and other `.env.*` files are ignored by Git; only `.env.example` is permitted as a tracked template. GitHub Actions secrets or a hosting provider's secret settings are the appropriate places for credentials if you later run paid jobs or deploy this service.

The example selects `gpt-4.1-mini-2025-04-14` with input/output rates of US$0.40/US$1.60 per million tokens. Check the [official model pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini) before a paid run. Set a positive `MAX_RUN_USD` in your local `.env` to enable calls; the public template defaults to zero. The process-local spend ledger resets on restart; use one API worker and separate account spending controls. If a real key has ever been committed, [revoke it and create a replacement](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety). Deleting it from the latest file does not remove it from Git history or invalidate copies. Paid evaluation requires an explicit command:

```bash
python -m scripts.evaluate --mode answers --allow-paid
```

No paid API calls or claim-faithfulness measurement have been reported yet. Omit the paid command until access, pricing and budget are configured. Answers are constrained to the frozen 6 October 2026 reference date; this project does not implement a historical regulatory timeline.

## Measured retrieval result

On the frozen, AI-authored benchmark's **40 answerable items**, all with known sufficient support labels and **zero human-audited labels**, the real pgvector/Hugging Face run at clean commit `3cf92ab` measured:

| Method | Macro page recall@5 | Complete evidence@5 |
| --- | ---: | ---: |
| Dense, BM25, TF-IDF (each) | 0.950 | 0.950 |
| Dense + BM25 + identifier RRF | 0.975 | 0.975 |
| Fusion + cross-encoder reranking | 1.000 | 1.000 |

The run recorded zero retrieval errors. Page hit and complete-page coverage had the same values in this dataset. These scores measure recovery of **annotated sufficient support**, not all relevant pages in the regulations, general regulatory QA accuracy, or answer faithfulness. [docs/evaluation.md](docs/evaluation.md) defines the metrics and limitations. The frozen dataset SHA-256 is `914d4af83b8876ee2758f4d63ada858fb6cd02f04b9aacf3dcbd062f5bd0cc13`; the portable [run summary](docs/reports/retrieval-run.json) and [ranked citation records](docs/reports/retrieval-predictions.jsonl) are checked in.

## Verification and scope

At the latest reported checkpoint, **40 offline unit tests passed**, and real pgvector ingestion plus Hugging Face reranking integration checks passed. Docker build and API smoke tests passed locally. [GitHub Actions](https://github.com/chichichiB/Risk-Regulatory-Q-A-Assistant/actions/runs/37542136078) passed offline validation and the Docker build for commit `6bc150a`. Paid end-to-end tests have not run. Offline tests use fixtures/mocks and need no key, model download or database:

```bash
python -m pytest tests/unit -q
```

The local demo has Swagger UI, PostgreSQL/pgvector and no custom frontend, authentication or cloud deployment. See [docs/design.md](docs/design.md) for design boundaries and [docs/evaluation.md](docs/evaluation.md) for the measurement protocol.

See [docs/interview-guide.md](docs/interview-guide.md) for design rationale and interview preparation, and [docs/verification.md](docs/verification.md) for precise validation and remaining limitations.

Extraction dispositions for the 13 heuristic layout flags are in [corpus/extraction-review.json](corpus/extraction-review.json); review was performed by an AI agent, not a human regulatory expert.
