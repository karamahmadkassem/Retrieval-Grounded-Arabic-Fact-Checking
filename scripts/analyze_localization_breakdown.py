"""Breakdowns for in-article localization eval JSON (same slices as open-domain BM25)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localization_metrics import summarize


def print_block(title: str, groups: dict):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for label, rows in sorted(groups.items(), key=lambda x: -len(x[1])):
        m = summarize(rows)
        if not m:
            continue
        print(f"  {str(label)[:32]:32s} n={m['n']:6d}  Recall@1={m['recall@1']:.3f}  "
              f"Recall@10={m['recall@10']:.3f}  MRR={m['mrr']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--results-key", default="results")
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    rows = data[args.results_key]
    overall = summarize(rows)
    print("Overall", overall)

    by_j = defaultdict(list)
    by_t = defaultdict(list)
    by_m = defaultdict(list)
    for r in rows:
        by_j[r.get("judgement")].append(r)
        by_m[r.get("gold_match_type")].append(r)
        if r.get("judgement") == "refuted" and r.get("type"):
            by_t[r["type"]].append(r)
    print_block("By judgement", by_j)
    print_block("By mutation type (refuted)", by_t)
    print_block("By gold match type", by_m)

    out = {
        "overall": overall,
        "by_judgement": {k: summarize(v) for k, v in by_j.items()},
        "by_mutation_type": {k: summarize(v) for k, v in by_t.items()},
        "by_gold_match_type": {k: summarize(v) for k, v in by_m.items()},
    }
    output = args.output or args.results.with_name(args.results.stem + "_breakdown.json")
    output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWritten to {output}")


if __name__ == "__main__":
    main()
