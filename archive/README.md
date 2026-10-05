# Archive

Stopped tracks and one-off diagnostics. **Not** the active pipeline.

## What is active now

In-article evidence localization: known Wikipedia page → gold span.

- BM25 over overlapping **1–3 sentence** windows (kept as the lexical baseline).
- Next neural method: **extractive BERT** (one span), not bi-encoder ranking.

Live code: `scripts/`. Live data: `data/arafa/` (ARAFA, verification, cache, `wikipedia_chunks.json`, `gold_passages.json`, `localization/`). Live notes: `results/evidence_localization_task.md`, `results/inarticle_bm25_findings.md`.

## What lives here

| Path | Why archived |
|------|----------------|
| `scripts/` | Open-domain BM25/Pyserini, distractor dump, e5 bi-encoder + cross-encoder reranker, pilots |
| `slurm/` | Octopus jobs for the ranking models |
| `results/` | Open-domain BM25 write-up, old ranking slides, gold-build spotchecks |
| `data/` | Small analysis JSON from the open-domain / evidence-match pilots |
| `artifacts/` | Large local files (Lucene indexes, wiki dump, e5 checkpoints). Gitignored |

Do not train `train_inarticle_biencoder.py` as the main method unless ranking is revived as a baseline.
