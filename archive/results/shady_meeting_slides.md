# Retrieval-Grounded Arabic Fact-Checking

In-article evidence localization — meeting slides (Dr. Shady Elbassuoni)

---

## Slide 1 — Title

**Retrieval-Grounded Arabic Fact-Checking**  
In-article evidence localization

- Karam Kassem · supervisor Dr. Shady Elbassuoni
- Dataset: ARAFA (~182K claim–evidence pairs)
- This meeting: task, BM25 baseline, first neural result, what is blocked on the cluster

---

## Slide 2 — Why this task

ARAFA tests verification when the evidence text is already given.

A real checker must first find that evidence inside the source page.

| Stage | Input | Output |
|--------|--------|--------|
| Document known | claim + Wikipedia article id | the page is given |
| **This module** | claim + that page’s passages | ranked evidence chunk |
| Later | claim + retrieved chunk | supported / refuted |

Same idea as FEVER sentence selection, with page retrieval already solved.

---

## Slide 3 — What we retrieve

- **Input:** Arabic claim + known article
- **Candidates:** overlapping 1–3 sentence windows already stored for that article
- **Gold:** the window that matches ARAFA evidence (`chunk_id`)
- **Score:** is the gold window in the top 1 / 5 / 10? (Recall@k and MRR)

Evidence text is used only as a label. The model never sees it at test time.

---

## Slide 4 — Data we train and test on

Keep a claim only if the judgement is **supported** or **refuted**, and the gold span is **exact** or **fuzzy**.

| Split | Articles | Claims |
|--------|----------|--------|
| Train | 2,634 | 118,664 |
| Val | 330 | 14,993 |
| Test | 329 | 14,814 |
| **Total** | **3,293** | **148,471** |

- Split is by **article** (80/10/10, seed 42). No article appears in two splits.
- Left out: NEI, unmatched spans, missing chunk ids.

---

## Slide 5 — Pipeline

1. **BM25** inside the article — lexical baseline, and the source of hard negatives
2. **Bi-encoder** (`multilingual-e5-small`) — claim and each window become vectors; rank by similarity
3. **Cross-encoder** (next) — rerank the top 50 from BM25 ∪ dense

| Term | Meaning |
|------|---------|
| **Hard negative** | A window on the same page that BM25 ranks high, but that is not the gold span |
| **Bi-encoder** | Encode claim and passage separately, then compare vectors. Fast over one article |
| **Cross-encoder** | Read claim and passage together. Slower; used only on the shortlist |

---

## Slide 6 — BM25 baseline (full test, n = 14,814)

| Metric | Score |
|--------|--------|
| Recall@1 | **0.716** |
| Recall@5 | 0.919 |
| Recall@10 | 0.962 |
| MRR | **0.801** |

Neural models must beat **R@1 = 0.716** and **MRR = 0.801**.

Oracle check (val): search with the evidence text itself → R@1 **0.878**, R@10 **≈ 1.0**. The gold window is in the article. The gap is matching the **claim** wording, not a missing page.

---

## Slide 7 — Where BM25 fails

| Slice | n | Recall@1 | MRR |
|--------|---|----------|-----|
| Exact gold | 9,145 | **0.819** | 0.877 |
| Fuzzy gold | 5,669 | **0.552** | 0.678 |
| Supported | 10,781 | 0.721 | 0.805 |
| Refuted | 4,033 | 0.703 | 0.790 |

- Exact vs fuzzy is a **27-point** gap at R@1. Supported vs refuted is small.
- Main error: Wikipedia wording drifted from the ARAFA evidence span (fuzzy), so keyword overlap is weak.
- Open-domain BM25 (search 15.1M chunks, page unknown) was R@1 **0.527**. Giving the article raises R@1 to **0.716**. Remaining errors are **inside the page**.

---

## Slide 8 — Model we are training

| Choice | Why |
|--------|-----|
| **multilingual-e5-small** | Arabic-capable embedding model; fits one 32 GB V100 |
| 1 epoch, batch 32, 7 hard negatives | MultipleNegativesRankingLoss |
| Prefixes `query:` / `passage:` | Required by E5 |
| **e5-large** | Stronger in principle; ran out of GPU memory on our V100 |

**Small vs large** = model size, not dataset size. Full training still uses **e5-small** on all 118,664 claims.

---

## Slide 9 — Smoke result (done)

GPU job on a V100. **4,000** train claims, **500** test claims. Wall time **~11 minutes**.

| | BM25 (full test) | e5-small smoke (500) |
|--|--|--|
| Recall@1 | 0.716 | **0.776** |
| Recall@5 | 0.919 | 0.928 |
| Recall@10 | 0.962 | 0.964 |
| MRR | 0.801 | **0.842** |

- Direction is right: dense retrieval beats lexical BM25 on this sample.
- This is **not** the full-test number. The 500-claim slice is easier to over-read.

---

## Slide 10 — Full training — status

| Item | State |
|------|--------|
| Job | `917259`, script `train_biencoder.sh` |
| Work | all 118,664 claims + full test (~14.8k) |
| Hardware | 1× V100 32 GB, `gpu` partition, **6 h** limit |
| Queue | **Pending ~4 days.** Reason: `(Resources)` — no free V100 |
| Account | `kma88` had expired; SSH works again; the job is still waiting |

- Estimate once a GPU starts: **~2 h** to mine hard negatives + **~3–5 h** train and eval. It may finish inside 6 h, or hit the time limit.
- No mid-run checkpoint in the current script, so a timeout means a full restart.
- K20 nodes are free and too small. We are not submitting a second copy of the same job.

---

## Slide 11 — Pending and next steps

**Pending now**

- Full e5-small run (job 917259) and full-test Recall / MRR vs BM25
- If it times out: save checkpoints (and reuse mined negatives) so the next 6 h continues instead of starting over

**Next, in order**

1. Report full-test bi-encoder vs BM25, including exact vs fuzzy
2. Hybrid shortlist: BM25 ∪ dense, top 50 (reciprocal rank fusion)
3. Fine-tune a **cross-encoder** on that shortlist and compare three systems: BM25, dense, hybrid+rerank
4. Write the localization results note

**Later (not this GPU job)**

- Open-domain retrieval over all of Arabic Wikipedia (already tagged, not the main number)
- Claim verification with the retrieved span (including LoRA on a generative model)
