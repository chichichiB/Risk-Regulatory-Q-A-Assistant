# Interview guide: design choices, evidence and limits

This guide explains the engineering decisions in the Risk Regulatory Q&A Assistant. It separates implemented behavior and measured retrieval from answer-quality claims that still require an authorized paid run and human review.

## What problem does the project solve?

It demonstrates how to answer English questions from a **fixed, auditable regulatory corpus** while showing the exact source version and page behind each claim. Five official PDFs cover OSFI operational resilience, third-party risk, technology/cyber risk, corporate governance, and Basel SRP30 risk management. That breadth creates realistic vocabulary overlap and authority distinctions, but the corpus is intentionally small enough for reproducible local experiments. It is not a general regulatory research system and does not decide legal applicability for a particular institution.

**Why freeze PDF bytes rather than fetch pages on every request?** Regulators may regenerate a PDF at an unchanged URL. The source manifest pins SHA-256, page count, retrieval time, model revisions and a release ID; the exact PDF bytes are preserved in a gitignored snapshot archive. This makes a reported retrieval result interpretable. If the same URL later serves different bytes, verification fails and a new corpus release needs explicit review. A timestamp alone would not preserve the source text or prove a rule was in force.

**Why one-page chunks and character offsets?** A chunk cannot cite text across a physical page boundary. Offsets let the benchmark check whether the first five retrieved chunks actually include a labeled supporting sentence, rather than merely landing on its page. This also makes citations inspectable. The trade-off is that a condition or footnote on the next page may require multiple chunks; complete-evidence@5 tests that. The extraction uses `pypdf` layout text and flags suspicious layouts, but tables and PDF reading order still merit manual inspection. The frozen release has 110 pages and 181 chunks, with a 320-token chunk limit and 40-token overlap.

## Why this retrieval pipeline?

**Dense search:** `BAAI/bge-small-en-v1.5` places paraphrases near one another. Its 384-dimensional vectors are stored with versioned passages in PostgreSQL/pgvector. The query-only retrieval prefix is applied as the model documents; passage text is not prefixed. Exact cosine search is suitable for this small corpus and avoids the recall/latency trade-off of approximate HNSW indexing. The limitation is that a dense model can miss exact clause IDs and may rank a conceptually similar but wrong authority passage highly.

**BM25:** Regulatory questions often contain identifiers, section numbers and exact phrases. BM25 complements dense retrieval on such terms. The sparse view is built from the same active corpus release as pgvector; this avoids mixing editions. TF-IDF exists as a simple separate baseline, not as another name for BM25. Sparse search can miss paraphrases, which is why both channels are measured.

**Reciprocal rank fusion (RRF):** Dense cosine distance and BM25 scores have different scales. RRF adds rank contributions (`1/(60 + rank)`), so the combination does not depend on unvalidated score calibration. The retrieval code also handles direct identifier matches and deduplicates passage IDs. Fusion can still surface too many related but non-answering passages; it is candidate selection, not evidence verification.

**Cross-encoder reranking:** `cross-encoder/ms-marco-MiniLM-L6-v2` scores a question paired with each of at most 30 fused passages, then selects up to five chunks. It is more expensive than precomputed embeddings, so it only runs on a bounded candidate set. A high pair score is still not a factual-support verdict. In the frozen 40-answerable-item benchmark, dense/BM25/TF-IDF each recovered the annotated support on 95% of items, fusion on 97.5%, and fusion plus reranking on 100%. All current positives have one AI-authored known sufficient support set and **zero human-audited labels**; these scores do not prove perfect regulatory QA.

## Why use a graph and two verification layers?

LangGraph makes routing, retrieval, evidence grading, one optional rewrite, generation, semantic checking, one optional repair and terminal refusal explicit. Each request is limited to eight provider attempts, with implicit SDK retries disabled. The graph has no shell, browsing or external-write tools; retrieved documents are untrusted evidence, not instructions. A deadline, token bounds and spend reservation limit runaway or unexpectedly costly behavior. Graph structure is useful for inspecting failure paths; it does not itself make model judgments correct.

Citation checks serve different purposes. Code enforces that a claim cites passage IDs from the current evidence set and release. This is a deterministic boundary against invented IDs or edition mixing. A model verdict then evaluates whether the cited passages actually support the claim. `answer_text` is rendered from verified atomic claims, avoiding an extra unchecked prose field. The semantic verdict may still be wrong, especially for negation, numerical conditions, scope or cross-authority comparisons. The paid evaluation's separate judge call uses the **same configured model**, so its faithfulness score must be called a same-model judge-assisted estimate until people audit claims.

Refusal is a first-class outcome: outside-corpus or wrong-date questions, unclear authority, and insufficient evidence should not be answered as if the corpus were comprehensive. Dependency/provider failures are `service_error`, not a correct refusal. The release supports one frozen reference date; it does not build a historical timeline or establish which text applied to a particular institution on another date.

## How is the project evaluated honestly?

The 15 development questions are separate from the frozen 50-item benchmark. Of the 50, 40 are answerable (eight primary questions per source) and 10 test refusal/version/out-of-scope handling. Retrieval ablations use the same questions, source release and labels, with agent rewriting disabled. Rewriting is a separate end-to-end experiment. Every run records its git commit, dataset/corpus hashes, model revisions, configuration, per-question evidence and errors.

Page recall@5 means coverage of the **annotated known sufficient support set** in the pages of the first five chunks; it is not a census of every relevant page in all regulations. Hit@5 asks only whether one support page was found. Complete-page-coverage@5 asks whether all pages of a sufficient support path were found. Complete-evidence@5 goes further: it checks whether the retrieved chunks' actual character intervals cover every labeled support span. None of these proves the final answer's claims are entailed by the sources. The reported 100% reranked retrieval is 40/40 on this limited AI-authored benchmark, not an answer-faithfulness result or broad performance guarantee.

The paid answer run has **not** been executed. It requires a local key, chosen OpenAI model, current input/output prices, a positive `MAX_RUN_USD`, and explicit `--mode answers --allow-paid`. Refusal accuracy, answer coverage and same-model judge-assisted claim faithfulness remain unmeasured. Human audit coverage is zero. A future result should publish both rate and denominator; undefined rates are `null`, and service errors stay visible in attempted-query counts.

## How can someone run or extend it?

FastAPI exposes Swagger UI at `/docs`, `/health` for liveness, `/ready` for dependency readiness, and `/ask` for questions. Docker Compose runs a Python 3.12 CPU API with PostgreSQL/pgvector; the API mounts local `data/` read-only and uses one worker because its spend ledger lives in process memory. A restart resets that ledger, so an external account budget remains important. The first version is a local demo without a custom frontend, authentication or cloud deployment.

The next technically useful extension is a human audit of the frozen gold spans and a sampled set of answer claims, especially multi-page conditions and OSFI/Basel comparisons. If the corpus expands, measure latency before introducing an approximate index, preserve version filters, and repeat the same retrieval ablations on a separately reviewed dataset. Do not describe a newly generated PDF or tuned benchmark as the same frozen experiment.
