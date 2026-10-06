# Evaluation guide

This guide separates measured retrieval from unmeasured answer quality. The frozen corpus release is `fa82be6139b9eb4af71bce55`, with five official PDF snapshots, 110 pages and 181 chunks. The held-out dataset hash is in [eval/manifest.json](../eval/manifest.json). Preserve the exact gitignored PDF snapshots; a later generated PDF from the same official URL may have different bytes and must fail the manifest hash check rather than silently replace the evaluation corpus.

## Labeled sets and what the labels mean

The 15-item development set has three questions primarily assigned to each source. The separate 50-item benchmark has 40 answerable questions (eight primary per source) and 10 refusal/version/out-of-scope questions. `eval/manifest.json` freezes their hashes and records **zero human-audited labels**. Questions and support spans were AI-authored and source-checked. A support set is a **known sufficient way to answer**, not an exhaustive judgment of every relevant regulatory page. All current positive items have one annotated support set. Do not tune on the held-out 50 or relabel a miss to make a system score higher.

Each positive item identifies its reference answer and one or more minimal support sets of exact spans, with physical PDF page and version IDs. The span text hash is checked against preserved extracted pages before a run. Negative items specify the expected refusal status. These machine checks establish corpus and label consistency; they do not constitute independent human review of regulatory meaning.

## Retrieval metrics as implemented

For an answerable question, score the **first five ranked chunks**. Convert them to unique `(version_id, physical_pdf_page_index)` page IDs; if two chunks are on the same page, do not add a sixth chunk. For each annotated minimal support set `S`, let `P(S)` be its pages and `R` be the pages of those five chunks. The code computes each metric for every alternative support set and takes the best value for that question, then macro-averages over answerable questions. Today each positive item has one support set, so the alternative selection has no effect on the reported run.

| Metric | Per-question calculation | Interpretation |
| --- | --- | --- |
| `page_recall_at_5` | `max_S |R ∩ P(S)| / |P(S)|` | Fraction of a known sufficient support set's pages found. This is **not** recall over all relevant pages. |
| `hit_at_5` | `1` if any known support page is found; else `0` | At least one annotated support page found. It must not be labeled recall. |
| `complete_page_coverage_at_5` | `1` if the pages of any support set are all in `R`; else `0` | All necessary pages for a labeled support path found. |
| `complete_evidence_at_5` | `1` if the union of retrieved chunks' half-open character intervals on the frozen page fully covers every span in any support set; else `0` | The actual retrieved text contains a labeled complete support path. |

Page coverage can be `1` while complete evidence is `0`: a chunk from the right page might miss the relevant sentence. Even complete span coverage is not proof that a generated claim is true. The 10 negative items are excluded from retrieval metric denominators. An undefined rate is `null`, never `0%` or `100%`.

Run all five real local retrieval methods on the same frozen questions, labels and corpus, with agent query rewriting disabled:

```bash
python -m scripts.evaluate --mode all
```

The accepted `--mode` values are `all`, `dense`, `bm25`, `tfidf`, `hybrid`, `reranked`, and `answers`; `--dataset` and `--output-dir` can override their defaults (`eval/benchmark.jsonl` and `reports/`). The run writes timestamped `run.json` and `predictions.jsonl`, including dataset/corpus/lock/model/prompt identifiers, git state, timings, per-query evidence and errors. The default command makes **no OpenAI calls**.

## Frozen retrieval result

The clean commit `3cf92aba3ee8bfad40eb281730bb15124f8ffbe8` produced `reports/20261006T222337Z-all-909316/run.json` on Python 3.12.15 / Windows 11 with the actual pinned HF models and pgvector. The dataset SHA-256 was `914d4af83b8876ee2758f4d63ada858fb6cd02f04b9aacf3dcbd062f5bd0cc13`; there were 40 answerable scored items and zero retrieval errors.

| Method | Page recall@5 | Hit@5 | Complete page@5 | Complete evidence@5 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.950 | 0.950 | 0.950 | 0.950 |
| BM25 | 0.950 | 0.950 | 0.950 | 0.950 |
| TF-IDF | 0.950 | 0.950 | 0.950 | 0.950 |
| Hybrid RRF (identifier channel when applicable) | 0.975 | 0.975 | 0.975 | 0.975 |
| Hybrid + cross-encoder | 1.000 | 1.000 | 1.000 | 1.000 |

Because these labels identify known sufficient support rather than all relevant pages and have no human audit, `1.000` means all 40 labeled support paths were recovered in this small test. It is not a general recall guarantee and says nothing about answer correctness, citation truth or claim-level faithfulness. The 10 negative items are for answer/refusal evaluation, not retrieval scoring. No 88→94% target or 95% faithfulness target is claimed as achieved.

## Answer, refusal and citation metrics

Answer evaluation is a **separate paid run**. It requires a local `OPENAI_API_KEY`, chosen `OPENAI_MODEL`, current `OPENAI_INPUT_USD_PER_MILLION` and `OPENAI_OUTPUT_USD_PER_MILLION` values, a positive `MAX_RUN_USD`, and an explicit opt-in:

```bash
python -m scripts.evaluate --mode answers --allow-paid
```

The agent may rewrite a query once and repair an answer once, with at most eight provider attempts per question and SDK implicit retries disabled. The budget ledger reserves an upper-bound cost before a call and is **process-local**, so restart resets it; use one API worker and an external account limit. The evaluation stores provider attempts, usage, latency and errors. No paid run or answer metric has been reported yet.

`score_answers` records attempted queries, service errors, answered queries, refusals, answer coverage on positives, exact expected-status accuracy on negatives, claim count and judgment coverage. Service errors remain in attempted counts and are not credited as correct refusals. For answered responses, `citation_validity` in the summary is the fraction of claims with nonempty citation IDs found among returned passages from the same release. This is a **structural membership check**, not semantic citation validity; the agent's verifier also checks its own evidence boundary before release.

The evaluator makes a separate call to the **same configured OpenAI model** for claim verdicts. `claim_faithfulness = supported judged claims / assessed claims`, where a verdict counts as supported only if its supporting IDs match the claim's citation IDs. `judgment_coverage = assessed claims / returned claims`. A zero denominator yields `null`. This is a **same-model judge-assisted estimate over supplied claim units**, not independent validation. Human audit count is currently zero. A defensible factual accuracy claim requires human review of sampled answer claims and source spans, including qualifiers, dates, negations and authority. Refusals are outside the claim denominator, so answer coverage and negative-status accuracy must accompany any faithfulness rate.

## Reproduction and verification tiers

1. **Offline:** `python -m pytest tests/unit -q` uses fixtures and fake services, with no key, model download or database. At the latest reported checkpoint, 40 tests passed.
2. **Real local components:** explicit integration tests exercise pinned Hugging Face embeddings/reranking and disposable PostgreSQL/pgvector. Real ingestion and reranking tests passed locally. The frozen retrieval run above belongs to this tier.
3. **Paid end-to-end:** only the opt-in answer command above, after key, model, pricing and spend cap are configured. It has not run and must not be inferred from retrieval scores.

For each reported result, retain the run directory's `run.json` and per-question predictions, the preserved PDF archive, the dataset hash and exact code commit. Distinguish local passes from Docker-build and remote-CI results; Docker build and the local HTTP smoke test passed; [remote CI passed](https://github.com/chichichiB/Risk-Regulatory-Q-A-Assistant/actions/runs/37542136078) for commit `6bc150a`. Never commit raw PDFs, model cache, `.env`, API keys or unreviewed paid outputs by accident.

## Measurement boundaries

Reference answers are available for a future human correctness/completeness rubric. The current runner does not compute answer correctness against those references or per-category aggregates. It measures retrieval, structural citations, supplied-claim faithfulness, answer coverage and refusal outcomes. All positive labels are English and largely source-focused; the small corpus, AI authorship and absence of human audit can inflate apparent performance. Use a separately authored and audited test set before making broader claims.

The hybrid variants include an extra identifier rank list for queries containing digit-bearing tokens. Matching passage text/document IDs are sorted by passage ID and capped at 20 before RRF. This is an explicit heuristic, not another learned model. The graph has an eight-attempt cap; a separate evaluation judge may add one call, so an answered item can use up to nine attempts in total, all sharing the run budget.
