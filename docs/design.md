# Risk Regulatory Q&A Assistant Technical Design

**Status:** Approved by the project owner on 6 October 2026. This describes intended behavior, not implemented behavior or measured results. Product code, source ingestion, and evaluation have not yet been verified. All example thresholds and models below are starting settings to validate on development data.

## Purpose and scope

Build a locally runnable, reproducible portfolio demonstration that answers English questions about a deliberately small, fixed set of five public OSFI/Basel regulatory documents. Every substantive answer must cite the exact source version and location that supports it. The system must decline questions when the selected corpus or requested time frame cannot support an answer. It does not determine an institution's legal obligations or replace regulatory advice.

**Confirmed brief:** this is an English-language portfolio project, with a fully runnable local demonstration, reproducible evaluation, and clear technical explanations. The user has no source collection or question set and has authorized selecting and creating both. The target repository currently has only a README and MIT license. **Engineering default:** use a modular FastAPI service with its Swagger UI and PostgreSQL/pgvector in Docker Compose; no custom frontend, authentication or cloud deployment in the first version. A Python 3.12 CPU container avoids making the host's Python 3.14 installation a compatibility requirement. The Docker engine is currently stopped, so build and runtime claims require later verification. An OpenAI API key has not been found in the current process; the user is being asked about API access and budget. "Reproducible" means pinned code, model revisions, source snapshots/hashes, run configuration and evaluators; hosted model outputs cannot be guaranteed byte-for-byte identical.

| Approach | Assessment |
| --- | --- |
| **Recommended: modular local stack** — FastAPI, PostgreSQL/pgvector, BM25, LangGraph, OpenAI, Docker Compose | Demonstrates the requested retrieval, agent, citation and evaluation boundaries while remaining runnable on one machine. |
| Stripped retrieval baseline only | Useful as an experiment, but it does not demonstrate bounded agent routing, grading and citation verification. |
| Production platform with UI, auth and cloud services | Adds operational scope that does not improve the portfolio's core evidence or reproducibility claims. |

## System boundary and interfaces

```text
fixed source manifest + immutable source snapshots
              | ingestion / extraction / page map / chunking
              v
 PostgreSQL: document versions, passages, embeddings, provenance
              |                         |
              | dense pgvector top-k    | sparse BM25 or TF-IDF top-k
              +------------+------------+
                           v
                   rank fusion / de-duplication
                           v
                    cross-encoder reranking
                           v
           bounded LangGraph decision + generation graph
                           v
              claim/citation verification + answer/refusal
                           v
                 FastAPI response / evaluation record
```

Keep ingestion, retrieval, graph orchestration, citation validation, and HTTP as independently callable units. The API invokes the same query service used by evaluation; no retrieval logic lives in route handlers. The generator receives only numbered, immutable `EvidencePassage` objects, never a free-form collection of web results. This constrains what a citation can refer to and makes the run auditable.

Suggested interfaces, stated as contracts rather than implementation classes:

| Interface | Input | Output / invariant |
| --- | --- | --- |
| `ingest(manifest)` | Approved five-entry source manifest | Immutable document-version records, extracted pages/sections, passages and vectors; repeat ingestion is idempotent by snapshot hash. |
| `retrieve(question, corpus_release, as_of)` | Normalized question and source filters | Dense and sparse ranked candidate lists with scores, version IDs and provenance. |
| `fuse_and_rerank(question, candidates)` | Candidates from both channels | De-duplicated evidence list, stable tie-breaker, top five passage IDs. |
| `answer(question, as_of?)` | User question and optional date | Structured answer, citations, decision status, corpus release and trace ID. |
| `verify(answer, evidence)` | Draft answer with claim-to-citation links | Per-claim support decisions and overall pass/fail; failed drafts are corrected once or refused. |

The wire response should distinguish `status = answered | insufficient_evidence | out_of_scope | ambiguous_authority | ambiguous_version | service_error`. An answered response has `answer_text`, `claims[]`, `citations[]`, `corpus_release`, and `trace_id`; construct its answer text from the verified claims, so an additional prose field cannot introduce unchecked assertions. A refusal has a clear reason and no unsupported answer. `POST /ask` should validate bounded question length and optional `as_of` date. A separate readiness endpoint may check model/index/database availability without making a model call. Do not expose internal prompts, API keys or full traces in public responses.

## Corpus, provenance and version policy

The selected five-source corpus has official HTML landing pages and official PDF renditions:

| Source | Official landing page | Official PDF | Pages observed in source review |
| --- | --- | --- | ---: |
| OSFI E-21 Operational Risk Management and Resilience | [E-21](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/operational-risk-management-resilience-guideline) | [PDF](https://www.osfi-bsif.gc.ca/en/print/pdf/node/2332) | 21 |
| OSFI B-10 Third-Party Risk Management | [B-10](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/third-party-risk-management-guideline) | [PDF](https://www.osfi-bsif.gc.ca/en/print/pdf/node/567) | 30 |
| OSFI B-13 Technology and Cyber Risk Management | [B-13](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/technology-cyber-risk-management) | [PDF](https://www.osfi-bsif.gc.ca/en/print/pdf/node/570) | 24 |
| OSFI Corporate Governance | [Guideline](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/corporate-governance-guideline-2018) | [PDF](https://www.osfi-bsif.gc.ca/en/print/pdf/node/592) | 20 |
| Basel Framework SRP30, 15 December 2019 in-force version | [SRP30](https://www.bis.org/committees/bcbs/basel-framework/standard/srp/30/inforce/2019-12-15/published/2019-12-15) | [Chapter PDF](https://www.bis.org/committees/bcbs/basel-framework/81535/chapter.pdf) | 15 |

The observed total is 110 PDF pages. The Basel chapter PDF was generated on 6 October 2026; the chapter's stated in-force date is distinct from that generation date. Use the chapter itself, not the 1,428-page full Basel Framework PDF. Ingestion must confirm titles, publication/effective metadata, page counts, downloaded bytes and hashes before freezing a release. The snapshot reference date is **6 October 2026**. It does not itself establish legal currency or applicability.

List each source in a checked-in manifest with official URL, jurisdiction, title, identifier/chapter, publication date, effective-from/to dates where stated, source format, language, retrieval timestamp, content SHA-256, and local snapshot path. For page-based scoring, ingest an official PDF rendition of each selected source and pin its bytes; an HTML page can remain the canonical public link, but a locally generated PDF would not provide authoritative page provenance. Pin a `corpus_release` derived from the manifest. Prefer final primary sources and reject draft/consultation variants. Treat OSFI and Basel as distinct authorities: Basel's standard is not automatically the Canadian rule, and an OSFI answer must not be inferred solely from a Basel passage. Explicit OSFI/Basel comparisons are allowed only when each side has independent cited evidence.

The first version answers from this one frozen corpus release. It does **not** build a historical regulatory timeline. If a question asks about a date or version outside what this snapshot can establish, refuse or ask for clarification; do not extrapolate current validity from the download date. If the authority is ambiguous, ask the user to specify it. Do not mix effective versions or authorities in an unqualified answer.

`DocumentVersion` minimally contains `doc_id`, `version_id`, `authority`, `title`, `official_url`, `effective_from`, `effective_to`, `published_at`, `retrieved_at`, `sha256`, `status`, and `corpus_release`. `Passage` contains `passage_id`, `version_id`, `text`, `section_heading`, `paragraph_label`, `pdf_page_index` (one-based physical PDF page), `printed_page_label` (if present), `html_anchor` (if present), `char_start/end`, extraction quality flags, and embedding model/revision. The passage ID is derived from version hash plus location and chunk sequence, so it cannot silently refer to altered text. Keep a page map: printed page numbers and physical PDF page indices may differ, and an HTML section may have no page at all. Citation display must state which location convention it uses and link to the official source; a PDF `#page=N` link is supplementary because browser support varies.

Extract tables, footnotes, headings and numbered paragraphs without losing their relationship. Flag OCR failures, empty pages, and tables whose row/column structure cannot be preserved; exclude or manually repair such passages before labeling. Each chunk stays within **one physical PDF page**, as well as one document version; keep the heading and paragraph label as metadata. Do not copy explanatory text or a footnote from another page into the cited chunk: preserve a separate linked passage and retrieve both when needed. Page-constrained chunks can lose cross-page context, so annotate minimal multi-page support sets and measure complete-evidence coverage. Record extraction software/version and chunking parameters in the corpus release. Reconcile source-page, extracted-page and passage counts before publishing an index.

## Hybrid retrieval and reranking

Use [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) as the proposed 384-dimensional Hugging Face/Sentence Transformers bi-encoder, with its exact model revision pinned during implementation. Apply its documented retrieval query prefix, `Represent this sentence for searching relevant passages: `, to questions only; do not prefix document passages. Store vectors in PostgreSQL/pgvector and use exact cosine search initially. pgvector's HNSW option is unnecessary until latency measurements justify its recall trade-off. [pgvector documentation](https://github.com/pgvector/pgvector)

Use `rank_bm25` for the production sparse channel. Fit its corpus index only on the frozen release and rebuild deterministically when the release changes. Use scikit-learn TF-IDF as a separately reported baseline. Record tokenization, case folding, stopwords, punctuation handling, and treatment of identifiers such as `E-21`, `B-10`, `B-13` and `SRP30`. Keep a direct identifier/section match path for explicitly named clauses. Do not compare raw dense and sparse scores as if calibrated.

Retrieve a configurable top `k` from each channel (initially 20 each), filter by authority and frozen release **before** ranking where feasible, de-duplicate on `passage_id`, then use reciprocal rank fusion (initially `1/(60 + rank)` per channel) to form a bounded candidate set. Use `cross-encoder/ms-marco-MiniLM-L6-v2` at a pinned revision to rerank at most 30 fused candidates, then return at most five evidence passages. A cross-encoder score ranks pairs; it does not prove that a passage supports an answer. [Sentence Transformers cross-encoder guidance](https://www.sbert.net/docs/cross_encoder/usage/usage.html)

Log dense rank, sparse rank, fused rank, rerank score, passage ID and selected version for each candidate. Tune `k`, fusion constant, chunk size and rerank cutoff on the separate development questions only. Compare dense-only, sparse-only, TF-IDF, hybrid, and hybrid-plus-rerank on the **same** fixed corpus, questions, gold labels and evaluation script, with agent query rewriting disabled for a fair retrieval comparison. Evaluate rewrite as a separate end-to-end intervention. If reranking removes a required passage, record the failure; do not relabel the gold set.

## Bounded LangGraph answer flow

Use a `StateGraph` with typed state and conditional edges. State holds `question`, normalized `as_of`, `corpus_release`, route decision, retrieved candidate IDs, evidence IDs/text, draft claims, verification results, `rewrite_count`, `repair_count`, and terminal status. External clients and secrets are runtime dependencies, not serialized graph state. LangGraph's state/nodes/edges model and its recursion limit support this structure, but application-level counters are still needed to make refusal behavior deliberate. [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)

Proposed transitions:

```text
validate -> route
route: out of corpus/date ambiguous -> refuse
route: answerable -> retrieve -> grade evidence
grade: adequate -> generate structured claims + citations -> verify
grade: inadequate and rewrite_count=0 -> rewrite query -> retrieve
grade: still inadequate -> refuse
verify: pass -> answer
verify: fail and repair_count=0 -> repair from same evidence -> verify
verify: still fail -> refuse
all terminal paths -> END
```

The route and evidence grade may use a configurable OpenAI model, subject to confirmed API access and budget, but must return bounded structured decisions. Code enforces enum values, counters, allowed evidence IDs, source-version consistency and terminal transitions. Allow at most one query rewrite and one answer repair, with an overall limit of eight provider calls per request, including verifier calls; any provider retries consume that budget. Query rewrite changes search terms, never the user's requested authority or date. A missing API key, provider error or unavailable index is a service error, not an evidence-based refusal. OpenAI Structured Outputs constrain shape, not factual accuracy. [OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)

Disable implicit SDK retries and count every explicit attempt, including timeouts, against the request limit. Apply input/output token limits and a request deadline. Record provider usage, and stop a paid evaluation before the next request could exceed its configured budget using conservative token-cost reservation; unavailable pricing means no automatic paid run. Treat retrieved text as untrusted evidence, never as instructions to change the agent's role, reveal secrets or invoke tools. The graph has no shell, browsing or external write tools. Test document-embedded instruction attempts and citations to passages outside the current evidence set.

## Citations and refusal standard

Require the generator to break its response into checkable factual claims, each linked to one or more allowed passage IDs and exact location metadata. Verification has two layers:

1. **Syntactic:** every citation ID exists in the retrieved evidence set; source URL, version and locator match the immutable corpus record; citation links resolve to the intended official document. This can be deterministic.
2. **Semantic:** a cited passage actually entails the claim, including qualifiers, scope, dates, negations and numeric values. This requires review or a fallible judge with human adjudication; a real passage and well-formed link are insufficient. For a claim that combines rules, all necessary evidence must be cited.

The verifier should reject unsupported statements, version blending, authority blending, and claims that omit material conditions. A model-based verifier can flag risk but should not be advertised as a proof of truth. If supporting text is absent or conflicting, return `insufficient_evidence` or `ambiguous_version` with the missing point stated plainly. Avoid a generic answer followed by a disclaimer. The system should not assert a conclusion based on the fact that a source appears in top five.

## Evaluation design: 50 questions

Create a frozen **50-item benchmark**: 40 answerable questions, eight assigned a primary source document in each of the five-source corpus, plus 10 refusal/version/out-of-scope questions. Cross-document comparisons can be among the 40 if one primary document is assigned for balancing and both authorities are labeled. Include exact clause lookup, paraphrase and multi-page questions. Create a separate **15-item development set**, three primarily assigned to each document, for tuning and prompt refinement; none of these 15 items enters the held-out 50. For each item label `question_id`, category, primary document, authority, as-of scope, answerability, gold answer or refusal rationale, relevant **version-plus-physical-page IDs**, exact supporting text spans, minimal supporting page sets, required qualifiers and source version. AI may draft questions and labels, but that is not independent human validation. Manually audit source spans and ambiguous labels where capacity allows; describe the actual audit coverage in results. Freeze the benchmark and its hash before tuning. Report counts and uncertainty, especially for the 40-item retrieval denominator.

The primary retrieval metric is **macro page recall@5** on the 40 answerable benchmark items. Map the first five ranked **chunks** to their unique `(version_id, physical_pdf_page_index)` IDs without retrieving extra chunks to fill duplicate-page slots. For question `q`, let `R_q` be these unique pages and `G_q` all labeled relevant pages: `page_recall@5(q) = |R_q ∩ G_q| / |G_q|`. Average the 40 per-question ratios. **Page hit@5** is `1` if the intersection is nonempty and `0` otherwise. Their averages differ for multiple relevant pages; reporting hit as recall is incorrect. Also report **complete-page-coverage@5**: at least one annotated minimal supporting page set is a subset of `R_q`. Page matches are coarse: the retrieved chunk could miss the relevant sentence on that page. Therefore separately compute **complete-evidence@5** from the gold character spans: at least one minimal evidence set must have every required span fully covered by the union of the first five chunks' character intervals on the same frozen page. Retain these spans for citation and answer adjudication. Exclude the 10 negative questions from retrieval denominators and report refusal accuracy on them separately. Fix the page map, labels and chunk ranking rules before comparing systems.

For final answers, segment nontrivial factual assertions into atomic claims. **Claim-level faithfulness** is supported claims divided by all assessable claims in answered responses, under a written rubric that checks claim meaning against cited source text, including version and qualifiers. Count an unsupported claim as a failure even when its citation is syntactically valid. Until claims are human-audited, call this a **judge-assisted faithfulness estimate**, not a validated rate. Report the number of claims and answered questions alongside it. Empty answers/refusals do not enter the denominator; report answer rate, correct refusal rate and incorrect refusal rate separately to prevent refusal gaming. Also report citation validity (syntactic and semantic separately), answer correctness against the reference rubric, and per-category failures. A 95% faithfulness target or a change from 88% to 94% retrieval recall is an **unverified ambition**, not a result. Never put such numbers in the README as achieved until a frozen run supports them.

Store run metadata: git revision, corpus release/hash, embedding and reranker model revisions, OpenAI model ID, prompts/schema versions, retrieval parameters, random seeds where applicable, evaluation set hash, timestamp, provider call counts/cost and per-question predictions/scores. Keep the 50 benchmark labels sealed during tuning on the separate 15; after a benchmark run, improvements need a new versioned protocol rather than repeated selection on that set.

## Verification and delivery boundary

Deterministic, offline `pytest` tests should cover manifest/version filtering, page-map integrity, one-page chunk boundaries, sparse and fusion ranking on fixtures, citation ID/location validation, graph loop and provider-call limits, refusal paths, and FastAPI request/response contracts with mocked providers. They must run without OpenAI credentials, model downloads, PostgreSQL or network. A distinct real-component integration tier should test Hugging Face embeddings/reranking and disposable PostgreSQL/pgvector with small fixtures. FastAPI's official `TestClient` supports direct `pytest` endpoint tests. [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)

Paid end-to-end integration/evaluation runs should be explicit commands requiring an API key, pinned source snapshots, a healthy database and pinned local model weights. They record cost and latency and do not run implicitly on every pull request. Docker Compose should include the Python 3.12 CPU API and PostgreSQL/pgvector services, health checks, one-shot ingestion, and a documented benchmark command. Use FastAPI's generated Swagger UI for the local demo. Never fetch live regulatory pages at application startup. GitHub Actions should run formatting/type checks, deterministic offline tests and the Docker build; optional real-component and paid end-to-end jobs require an explicit trigger and credentials. This is a CI boundary, not a claim that the service or evaluation currently passes. [FastAPI container guidance](https://fastapi.tiangolo.com/deployment/docker/) · [GitHub Actions workflow documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)

## Acceptance criteria and pending runtime choice

A reviewer can start the local API and database from documented Docker Compose commands, ingest exactly the five frozen official-source snapshots, ask a question through Swagger UI, and inspect an answer with versioned page citations or an explicit refusal. The deterministic test tier runs with no credentials; real-component tests and paid end-to-end evaluation run by explicit command. The repository contains the 15-item development set, frozen 50-item benchmark, gold page/span labels, metric definitions, per-question outputs, and a run manifest sufficient to reproduce reported numbers. A README states measured results only after the relevant run has completed and identifies judge-assisted estimates and any human audit coverage.

The only user-dependent runtime choice still pending is OpenAI API access and an acceptable spend/call budget. Keep the model ID configurable and do not claim a paid end-to-end result until access exists. Source metadata, downloadable formats, model revisions and local hardware fit are implementation checks, not unresolved feature choices.

## Primary technical references

- [LangGraph Graph API and state/edges/recursion behavior](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [pgvector search, exact/HNSW indexing and hybrid search notes](https://github.com/pgvector/pgvector)
- [Sentence Transformers bi-encoder usage](https://www.sbert.net/docs/sentence_transformer/usage/usage.html) and [cross-encoder reranking usage](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/) and [containers](https://fastapi.tiangolo.com/deployment/docker/)
- [OSFI guidance library](https://www.osfi-bsif.gc.ca/en/guidance/guidance-library) and [Basel Framework](https://www.bis.org/committees/bcbs/basel-framework) for authoritative corpus and version checks
