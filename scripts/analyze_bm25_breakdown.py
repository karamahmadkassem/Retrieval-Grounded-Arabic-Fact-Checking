"""
analyze_bm25_breakdown.py

Breaks down the BM25 evaluation results by claim judgement, mutation
type, and gold match type (exact/fuzzy), to see WHERE retrieval
struggles -- not just the overall average.

Usage:
    py scripts/analyze_bm25_breakdown.py
"""

import json
from collections import defaultdict
from pathlib import Path

BM25_RESULTS = Path("data/arafa/bm25_eval_results.json")
ARAFA_PATH = Path("data/arafa/ARAFA.json")
K_VALUES = [1, 5, 10]


def compute_metrics(rows):
    n = len(rows)
    if n == 0:
        return None
    hits_at_k = {k: 0 for k in K_VALUES}
    rr_sum = 0.0
    for r in rows:
        rank = r["rank"]
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1
        rr_sum += (1.0 / rank) if rank else 0.0
    return {
        "n": n,
        **{f"recall@{k}": round(hits_at_k[k] / n, 4) for k in K_VALUES},
        "mrr": round(rr_sum / n, 4),
    }


def main():
    bm25_data = json.loads(BM25_RESULTS.read_text(encoding="utf-8"))
    per_claim = bm25_data["results"]

    claims = json.loads(ARAFA_PATH.read_text(encoding="utf-8"))
    claim_info = {c["final_id"]: c for c in claims}

    for r in per_claim:
        info = claim_info.get(r["final_id"], {})
        r["judgement"] = info.get("judgement")
        r["type"] = info.get("type")

    print("=" * 60)
    print("Overall")
    print("=" * 60)
    overall = compute_metrics(per_claim)
    for k, v in overall.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("By judgement (supported / refuted / nei)")
    print("=" * 60)
    by_judgement = defaultdict(list)
    for r in per_claim:
        by_judgement[r["judgement"]].append(r)
    for label, rows in sorted(by_judgement.items(), key=lambda x: -len(x[1])):
        m = compute_metrics(rows)
        print(f"  {label:12s} n={m['n']:6d}  Recall@1={m['recall@1']:.3f}  "
              f"Recall@10={m['recall@10']:.3f}  MRR={m['mrr']:.3f}")

    print("\n" + "=" * 60)
    print("By mutation type (refuted claims only)")
    print("=" * 60)
    by_type = defaultdict(list)
    for r in per_claim:
        if r["judgement"] == "refuted" and r["type"]:
            by_type[r["type"]].append(r)
    for label, rows in sorted(by_type.items(), key=lambda x: -len(x[1])):
        m = compute_metrics(rows)
        print(f"  {label:32s} n={m['n']:6d}  Recall@1={m['recall@1']:.3f}  "
              f"Recall@10={m['recall@10']:.3f}  MRR={m['mrr']:.3f}")

    print("\n" + "=" * 60)
    print("By gold match type (exact vs. fuzzy -- did Wikipedia drift hurt retrieval?)")
    print("=" * 60)
    by_match = defaultdict(list)
    for r in per_claim:
        by_match[r["gold_match_type"]].append(r)
    for label, rows in sorted(by_match.items(), key=lambda x: -len(x[1])):
        m = compute_metrics(rows)
        print(f"  {label:12s} n={m['n']:6d}  Recall@1={m['recall@1']:.3f}  "
              f"Recall@10={m['recall@10']:.3f}  MRR={m['mrr']:.3f}")

    output_path = Path("data/arafa/bm25_breakdown.json")
    breakdown = {
        "overall": overall,
        "by_judgement": {k: compute_metrics(v) for k, v in by_judgement.items()},
        "by_mutation_type": {k: compute_metrics(v) for k, v in by_type.items()},
        "by_gold_match_type": {k: compute_metrics(v) for k, v in by_match.items()},
    }
    output_path.write_text(json.dumps(breakdown, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten to {output_path}")


if __name__ == "__main__":
    main()
