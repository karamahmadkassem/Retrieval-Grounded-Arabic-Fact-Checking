"""
run_bm25_eval.py

Evaluates BM25 retrieval against gold_passages.json: for each claim,
does the correct chunk appear in the top-k results? Reports Recall@1,
Recall@5, Recall@10, and MRR.

Usage:
    py scripts/run_bm25_eval.py --sample-size 2000   # smoke test first
    py scripts/run_bm25_eval.py                      # full eval
"""

import argparse
import json
import os
import random
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher

K_VALUES = [1, 5, 10]

# Portable JDK bundled under data/tools/jdk for Pyserini (requires Java 21+).
_JDK_CANDIDATES = [
    Path("data/tools/jdk/jdk-21.0.6+7"),
    Path("data/tools/jdk/jdk-17.0.14+7"),
]


def ensure_java_home():
    if os.environ.get("JAVA_HOME"):
        return
    for candidate in _JDK_CANDIDATES:
        if (candidate / "bin" / "java.exe").exists():
            os.environ["JAVA_HOME"] = str(candidate.resolve())
            os.environ["PATH"] = str(candidate / "bin") + os.pathsep + os.environ.get("PATH", "")
            return


def load_claims_by_id(arafa_path: Path) -> dict:
    claims = json.loads(arafa_path.read_text(encoding="utf-8"))
    return {c["final_id"]: c["claim"] for c in claims}


def main():
    ensure_java_home()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/arafa/bm25_index"))
    parser.add_argument("--gold", type=Path, default=Path("data/arafa/gold_passages.json"))
    parser.add_argument("--arafa", type=Path, default=Path("data/arafa/ARAFA.json"))
    parser.add_argument("--output", type=Path, default=Path("data/arafa/bm25_eval_results.json"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    print("Loading gold passages ...")
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    evaluable = {
        fid: v for fid, v in gold.items()
        if v.get("match_type") in ("exact", "fuzzy") and v.get("chunk_id")
    }
    print(f"{len(evaluable)}/{len(gold)} claims have a scorable gold chunk.")

    print("Loading claim text ...")
    claims_by_id = load_claims_by_id(args.arafa)

    items = list(evaluable.items())
    if args.sample_size:
        random.seed(args.seed)
        items = random.sample(items, min(args.sample_size, len(items)))
    print(f"Evaluating {len(items)} claims.")

    print("Opening Lucene index ...")
    searcher = LuceneSearcher(str(args.index))
    searcher.set_language("ar")

    hits_at_k = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []
    per_claim_results = []

    for i, (final_id_str, gold_info) in enumerate(items, start=1):
        final_id = int(final_id_str)
        claim_text = claims_by_id.get(final_id)
        gold_chunk_id = gold_info["chunk_id"]

        if not claim_text:
            continue

        hits = searcher.search(claim_text, k=args.top_k)
        retrieved_ids = [h.docid for h in hits]

        rank = None
        if gold_chunk_id in retrieved_ids:
            rank = retrieved_ids.index(gold_chunk_id) + 1

        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1

        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        per_claim_results.append({
            "final_id": final_id,
            "gold_chunk_id": gold_chunk_id,
            "rank": rank,
            "gold_match_type": gold_info["match_type"],
        })

        if i % 2000 == 0:
            print(f"  [{i}/{len(items)}] running Recall@10 so far: "
                  f"{hits_at_k[10] / i:.3f}")

    n = len(per_claim_results)
    summary = {
        "total_evaluated": n,
        **{f"recall_at_{k}": round(hits_at_k[k] / n, 4) for k in K_VALUES},
        "mrr": round(sum(reciprocal_ranks) / n, 4),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": per_claim_results}, f,
                  ensure_ascii=False, indent=2)

    print("\n=== BM25 Retrieval Evaluation ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
