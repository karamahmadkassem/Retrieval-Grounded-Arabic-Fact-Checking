# BM25 Retrieval Baseline — Findings Note

**Date:** 2026-09-15  
**Status:** Locked baseline for Phase 3 retrieval evaluation  
**Prepared for:** Thesis results chapter (draft reference) / Dr. Shady review

---

## Setup (what was measured)

| Item | Value |
|------|-------|
| Retriever | BM25 via Pyserini (Lucene, Arabic analyzer) |
| Corpus | 15,138,203 chunks (2.73M gold + 12.40M distractor) |
| Distractor source | Arabic Wikipedia dump, single-sentence windows (`--max-window 1`) |
| Gold labels | `gold_passages.json` — 176,026 scorable claims (exact/fuzzy) |
| Query | Claim text from ARAFA (standard FC retrieval formulation) |
| Metrics | Recall@1 / @5 / @10, MRR |
| Claims evaluated | 176,026 (of 181,928 gold-label runs) |

**Source files:** `data/arafa/bm25_eval_results.json`, `data/arafa/bm25_breakdown.json`, `data/arafa/bm25_overlap_analysis.json`  
**Scripts:** `scripts/analyze_bm25_breakdown.py`, `scripts/analyze_overlap_recall.py`

### Methods & limitations

- **Corpus freshness mismatch:** The distractor haystack is built from a **live Wikipedia dump** (Sep 2026). Gold labels were matched against article text that may reflect **older wording** — the 70,628 "fuzzy" gold cases are evidence of this drift. The exact/fuzzy retrieval gap is therefore partly a **temporal mismatch artifact**: BM25 queries the claim against current Wikipedia text, while fuzzy gold labels already required approximate matching to find the evidence span. This is a known, named source of the fuzzy-match penalty — not merely inferred from the pattern.
- **Excluded claims:** This evaluation excludes **5,902 claims (3.2%)** with `match_type = none` — no resolvable gold passage in the indexed corpus (Wikipedia gaps, stubs, or missing sources). The Phase 0 gold-label run processed 181,928 claims; only the 176,026 with exact/fuzzy matches are scored here. See Phase 0 notes / `results/missing_chunks_sources.txt` for bookkeeping on none cases.
- **Corpus overlap:** **Verified: zero source-ID overlap** between the 3,301 gold corpus articles (`wikipedia_chunks.json`) and the 1,322,604 distractor articles (`wiki_corpus_chunks.jsonl`). Known ARAFA sources are excluded at build time via `source_verification.json`.

---

## Headline result

| Metric | Score |
|--------|-------|
| **Recall@1** | **0.527** |
| **Recall@5** | **0.847** |
| **Recall@10** | **0.915** |
| **MRR** | **0.657** |
| Claims evaluated | 176,026 |

BM25 retrieves the correct evidence chunk in the top 10 for ~91% of claims against a full-Wikipedia-scale haystack. This is the number all future retrieval improvements (dense, hybrid) are measured against.

---

## Breakdown 1 — By claim judgement

Does BM25 struggle more on refuted claims than supported ones?

| Judgement | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|-----------|---|----------|----------|-----------|-----|
| supported | 107,675 | 0.559 | 0.872 | 0.933 | 0.686 |
| refuted | 40,842 | 0.513 | 0.836 | 0.906 | 0.644 |
| **nei** | **27,509** | **0.423** | **0.767** | **0.854** | **0.560** |

**Takeaway:** Refuted claims are only slightly harder than supported (−4.6 pp Recall@1). The real gap is **NEI** — Recall@1 drops 13.6 pp vs. supported and 8.9 pp vs. refuted. This is expected: NEI claims often lack a direct supporting passage in the source article, so lexical retrieval has less to latch onto. Dense retrieval should be evaluated on this slice specifically; improvements here would be meaningful, but some NEI misses may be irreducible (genuine absence of evidence in Wikipedia).

---

## Breakdown 2 — By mutation type (refuted claims only)

Do certain kinds of refuted claims retrieve worse?

| Mutation type | n | Recall@1 | Recall@10 | MRR |
|---------------|---|----------|-----------|-----|
| Temporal Nuance | 13,533 | 0.544 | 0.924 | 0.673 |
| Contextual Reframing | 2,931 | 0.562 | 0.924 | 0.685 |
| Relationship Reconfiguration | 2,409 | 0.526 | 0.921 | 0.658 |
| Scope Refinement | 6,235 | 0.502 | 0.904 | 0.635 |
| Qualitative Shift | 8,170 | 0.488 | 0.890 | 0.622 |
| **Quantitative Precision** | **7,467** | **0.469** | **0.880** | **0.605** |

*(Types with n < 100 omitted — too sparse for reliable interpretation.)*

**Takeaway:** Among refuted claims, **Quantitative Precision** is the hardest mutation class (lowest Recall@1 and MRR among large groups). Claims that alter numbers, dates, or magnitudes produce claims whose wording diverges most from the source passage — exactly where lexical matching breaks down. **Temporal Nuance** and **Contextual Reframing** perform near the supported-claim average, suggesting the retrieval difficulty is tied to *what changed* in the claim, not refutation per se. This connects directly to the earlier ARAFA metadata analysis on mutation strategies.

---

## Breakdown 3 — By gold match type (exact vs. fuzzy)

Did Wikipedia drift hurt retrieval?

| Gold match type | n | Recall@1 | Recall@5 | Recall@10 | MRR |
|-----------------|---|----------|----------|-----------|-----|
| exact | 105,398 | 0.591 | 0.879 | 0.928 | 0.708 |
| **fuzzy** | **70,628** | **0.432** | **0.800** | **0.895** | **0.579** |
| **Gap (exact − fuzzy)** | | **+15.9 pp** | **+7.9 pp** | **+3.3 pp** | **+0.129** |

**Takeaway:** This is the strongest diagnostic finding. Claims whose gold evidence required fuzzy matching (Wikipedia text has drifted since ARAFA annotation) are systematically harder to retrieve — **16 pp lower Recall@1** than exact-match claims. The gap persists at Recall@10 (+3.3 pp), so it is not just a rank-1 tie-breaking issue. Combined with the corpus freshness note above, this is a reportable result about how **indexing live Wikipedia against older-aligned gold labels degrades lexical retrieval**. Dense retrieval is the natural next test: semantic similarity should partially close this gap.

---

## Breakdown 4 — Claim–evidence word overlap vs. Recall@1

*Measured, not inferred:* for each claim, token overlap = |claim tokens ∩ evidence tokens| / |claim tokens|, after Arabic normalization (strip tashkeel, collapse whitespace).

| Overlap (claim-side) | n | Recall@1 |
|----------------------|---|----------|
| 0–10% | 1,464 | 0.122 |
| 10–20% | 3,828 | 0.204 |
| 20–30% | 11,040 | 0.273 |
| 30–40% | 20,677 | 0.375 |
| 40–50% | 28,701 | 0.465 |
| 50–60% | 42,518 | 0.548 |
| 60–70% | 35,788 | 0.624 |
| 70–80% | 20,654 | 0.676 |
| 80–90% | 9,132 | 0.713 |
| 90–100% | 2,224 | 0.729 |

**Quintile summary:** Q1 (lowest overlap) Recall@1 = **31.1%** → Q5 (highest) = **68.6%** (+37.4 pp).

**Headline thresholds:**
- Overlap **< 30%** (n = 16,332): Recall@1 = **24.3%**
- Overlap **≥ 50%** (n = 110,316): Recall@1 = **61.4%**

**Subgroup mean overlap** (claim vs. ARAFA evidence text):
| Subgroup | Mean overlap | n |
|----------|--------------|---|
| supported | 0.559 | 107,675 |
| refuted | 0.522 | 40,842 |
| nei | 0.458 | 27,509 |
| exact gold | 0.533 | 105,398 |
| fuzzy gold | 0.537 | 70,628 |

**Takeaway:** Low claim–evidence lexical overlap strongly predicts BM25 failure — the pattern in Breakdowns 1–3 is **demonstrated**, not assumed. NEI claims have the lowest mean overlap (0.458), consistent with their weaker retrieval. Notably, **exact and fuzzy gold cases have nearly identical mean claim–evidence overlap** (0.533 vs 0.537), so the fuzzy retrieval penalty is not simply because those claims use different words from their evidence. It is more likely driven by **corpus-side drift**: the indexed Wikipedia chunk wording diverges from ARAFA evidence even when the claim itself overlaps the evidence text normally. That distinction matters for interpreting dense retrieval gains.

---

## Summary for thesis / advisor

1. **BM25 is a strong but incomplete baseline** — 91.5% Recall@10 overall, but uneven across subgroups (176,026 claims; 5,902 none cases excluded).
2. **NEI claims** are the weakest judgement class (42.3% Recall@1), with the lowest claim–evidence overlap (0.458).
3. **Quantitative refutations** are the hardest mutation type among refuted claims.
4. **Wikipedia drift (fuzzy gold)** imposes a large retrieval penalty (+15.9 pp Recall@1) — likely corpus-side wording drift, not lower claim–evidence overlap.
5. **Lexical overlap is a direct predictor:** claims with <30% overlap achieve 24.3% Recall@1 vs. 61.4% at ≥50%.
6. **Zero gold/distractor source overlap** verified — no methodological leakage from duplicate articles.
7. All future retrieval work should report **overall + these four breakdowns** for comparability.

---

## Next step

Move to **dense retrieval** (when GPU/Octopus available). Primary success criterion: close the exact/fuzzy gap and improve NEI Recall@1 without regressing on supported claims. Re-run `scripts/analyze_bm25_breakdown.py` on dense eval output using the same breakdown structure.
