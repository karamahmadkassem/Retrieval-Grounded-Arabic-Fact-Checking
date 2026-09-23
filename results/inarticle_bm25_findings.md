# In-article BM25 baseline (Phase 2)

**Date:** 2026-09-21  
**Task:** claim + known article → gold chunk (supported/refuted only)  
**Contrast:** open-domain BM25 (`bm25-baseline-v1`) searched 15.1M chunks; here candidates are **only that article’s windows**.

## Dataset (Phase 1)

| Split | Articles | Claims | Supported | Refuted |
|-------|----------|--------|-----------|---------|
| train | 2,634 | 118,664 | 85,975 | 32,689 |
| val | 330 | 14,993 | 10,885 | 4,108 |
| test | 329 | 14,814 | 10,781 | 4,033 |
| **total** | **3,293** | **148,471** | | |

Split unit = Wikipedia `source`, seed 42, **0 article overlap**.  
Skipped: 28,564 NEI; 4,854 none; 46 missing chunk_id; 41 no gold row.

## Test — claim query (primary baseline)

| Metric | Score |
|--------|-------|
| Recall@1 | **0.716** |
| Recall@5 | **0.919** |
| Recall@10 | **0.962** |
| MRR | **0.801** |
| n | 14,814 |

Span-relaxed (IoU ≥ 0.5) is essentially identical (R@1 0.716), so exact `chunk_id` is not overly harsh here.

## Val — claim vs evidence-oracle ceiling

| Query | Recall@1 | Recall@10 | MRR |
|-------|----------|-----------|-----|
| claim | 0.712 | 0.961 | 0.798 |
| **evidence (oracle)** | **0.878** | **0.9995** | **0.918** |

Oracle R@10 ≈ 1.0: gold chunks are lexically recoverable from evidence text. The gap vs claim query is the headroom for dense/rerank models.

## Test breakdowns

| Slice | n | Recall@1 | Recall@10 | MRR |
|-------|---|----------|-----------|-----|
| supported | 10,781 | 0.721 | 0.964 | 0.805 |
| refuted | 4,033 | 0.703 | 0.957 | 0.790 |
| exact gold | 9,145 | **0.819** | 0.972 | 0.877 |
| fuzzy gold | 5,669 | **0.552** | 0.947 | 0.678 |

Exact–fuzzy **Recall@1 gap: 26.7 pp** (still present when the article is given). Quantitative Precision remains the weakest large mutation class (R@1 0.674).

## vs open-domain BM25

Open-domain (15.1M docs, all judgements with gold): R@1=0.527, R@10=0.915.  
In-article (known page, supported/refuted): R@1=0.716, R@10=0.962.

Giving the article removes the “find the page” problem; remaining errors are **within-page** (especially Wikipedia drift / fuzzy gold).

**Neural models must beat test R@1=0.716 / MRR=0.801.**
