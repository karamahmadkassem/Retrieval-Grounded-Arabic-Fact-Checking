# Evidence localization task (in-article)

**Status:** Implementation track (replaces open-domain BM25 as the primary retrieval experiment)  
**Contrast with:** git tag `bm25-baseline-v1` (claim → 15.1M Wikipedia chunks)

---

## Task

Given a **claim** and a **known Arabic Wikipedia article** (`source` id), retrieve the passage in that article that corresponds to ARAFA `evidence`.

This is FEVER’s **sentence-selection** stage with document retrieval already solved (Thorne et al., 2018). FEVER still had to find the wiki page; we skip that. FEVER left NOTENOUGHINFO without evidence sentences; we likewise **exclude NEI** from this module.

### Inference vs training

| | Inference | Training / eval labels |
|---|-----------|------------------------|
| Input | `claim` + `source` + that article’s chunks | same + gold `chunk_id` |
| Must not use | `evidence` text | `evidence` is supervision only |
| Output | ranked `chunk_id`s | hit if gold id is in top-k |

---

## Filters

**Include** a claim iff all of:

- `judgement` ∈ {`supported`, `refuted`}
- `gold_passages.match_type` ∈ {`exact`, `fuzzy`}
- `gold_chunk_id` exists in `wikipedia_chunks.json` for that `source`

**Exclude:** `nei`; `match_type` ∈ {`none`, `error`}; missing chunk ids.

Candidates are the existing overlapping **1–3 sentence windows** in `data/arafa/wikipedia_chunks.json`. Do not re-chunk gold articles (labels are `chunk_id`s).

---

## Splits

- Unit of split: **Wikipedia `source` (article)**, not claim.
- Ratio: **80 / 10 / 10** train / val / test, seed **42**.
- Stratify articles by claim-count buckets so large pages are not all in one split.
- Two claims from the same article never appear in different splits.

Outputs: `data/arafa/localization/{train,val,test}.jsonl` + `split_manifest.json`.

Chunk lists stay in `wikipedia_chunks.json` (load by `source`); they are not duplicated in every JSONL row.

---

## Metrics

**Primary:** Recall@1 / @5 / @10 and MRR on **exact** `gold_chunk_id`.

**Diagnostic:** span-relaxed hit if predicted window IoU with gold sentence range ≥ 0.5 (overlapping windows can punish a correct neighbor).

**Breakdowns:** judgement (supported vs refuted); mutation `type` (refuted only); exact vs fuzzy gold.

---

## Pipeline (accuracy)

1. **In-article BM25** (Phase 2) — lexical baseline and hard-negative source.
2. **Bi-encoder** (Phase 3) — score all chunks in the article; merge with BM25 → top-50.
3. **Cross-encoder reranker** (Phase 4) — rerank top-50. Reportable neural result.

Hard negatives are **in-article near-misses**, not random Wikipedia pages.

---

## What we do not rebuild

Open-domain 1.3M distractor corpus, 15M Lucene index, and claim-only full-Wikipedia eval as the main number. Those stay under `bm25-baseline-v1`.
