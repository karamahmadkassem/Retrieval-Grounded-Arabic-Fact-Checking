"""Quick claim-evidence token overlap vs Recall@1 analysis."""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arabic_text_utils import normalize


def tokenize(text: str) -> set[str]:
    return set(normalize(text).split())


def claim_evidence_overlap(claim: str, evidence: str) -> float:
    ct, et = tokenize(claim), tokenize(evidence)
    if not ct:
        return 0.0
    return len(ct & et) / len(ct)


def main():
    bm25 = json.loads(Path("data/arafa/bm25_eval_results.json").read_text(encoding="utf-8"))
    claim_info = {
        c["final_id"]: c
        for c in json.loads(Path("data/arafa/ARAFA.json").read_text(encoding="utf-8"))
    }

    rows = []
    for r in bm25["results"]:
        info = claim_info.get(r["final_id"], {})
        ov = claim_evidence_overlap(info.get("claim", ""), info.get("evidence", ""))
        hit1 = 1 if (r["rank"] is not None and r["rank"] <= 1) else 0
        rows.append({
            "overlap": ov,
            "hit1": hit1,
            "judgement": info.get("judgement"),
            "gold_match_type": r["gold_match_type"],
        })

    bins = [
        (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01),
    ]
    by_bin = []
    for lo, hi in bins:
        sub = [x for x in rows if lo <= x["overlap"] < hi]
        if not sub:
            continue
        by_bin.append({
            "bin": f"{lo:.0%}-{min(hi, 1.0):.0%}" if hi <= 1 else f"{lo:.0%}-100%",
            "n": len(sub),
            "recall_at_1": round(sum(x["hit1"] for x in sub) / len(sub), 4),
        })

    sorted_ov = sorted(rows, key=lambda x: x["overlap"])
    n = len(sorted_ov)
    quintiles = []
    for q in range(5):
        start, end = q * n // 5, (q + 1) * n // 5
        sub = sorted_ov[start:end]
        quintiles.append({
            "quintile": f"Q{q + 1}",
            "overlap_min": round(sub[0]["overlap"], 3),
            "overlap_max": round(sub[-1]["overlap"], 3),
            "n": len(sub),
            "recall_at_1": round(sum(x["hit1"] for x in sub) / len(sub), 4),
        })

    thresholds = {}
    for label, pred in [("<30%", lambda x: x["overlap"] < 0.3),
                        (">=50%", lambda x: x["overlap"] >= 0.5)]:
        sub = [x for x in rows if pred(x)]
        thresholds[label] = {
            "n": len(sub),
            "recall_at_1": round(sum(x["hit1"] for x in sub) / len(sub), 4),
        }

    subgroup_overlap = {}
    for name, key_fn in [("by_judgement", lambda x: x["judgement"]),
                         ("by_gold_match_type", lambda x: x["gold_match_type"])]:
        buckets = defaultdict(list)
        for x in rows:
            buckets[key_fn(x)].append(x["overlap"])
        subgroup_overlap[name] = {
            k: {"n": len(v), "mean_overlap": round(sum(v) / len(v), 3)}
            for k, v in sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        }

    out = {
        "metric": "claim-side token overlap: |claim_tokens & evidence_tokens| / |claim_tokens|",
        "normalization": "arabic_text_utils.normalize (strip tashkeel, collapse whitespace)",
        "by_overlap_bin": by_bin,
        "by_quintile": quintiles,
        "thresholds": thresholds,
        "subgroup_mean_overlap": subgroup_overlap,
    }
    out_path = Path("data/arafa/bm25_overlap_analysis.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
