"""
run_inarticle_bm25_eval.py

BM25 ranking over chunks of the known Wikipedia article only.

Usage:
    py scripts/run_inarticle_bm25_eval.py --split test
    py scripts/run_inarticle_bm25_eval.py --split val --query-field evidence  # oracle
    py scripts/run_inarticle_bm25_eval.py --split val --query-field claim --oracle
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arabic_text_utils import normalize
from localization_metrics import K_VALUES, ranks_from_ids, summarize


def tokenize(text: str) -> list[str]:
    return [t for t in normalize(text).split() if t]


class BM25Article:
    def __init__(self, chunk_ids: list[str], texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.chunk_ids = chunk_ids
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(t) for t in texts]
        self.n = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.n, 1)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {
            t: math.log((self.n - f + 0.5) / (f + 0.5) + 1.0)
            for t, f in df.items()
        }

    def rank(self, query: str, top_k: int) -> list[str]:
        q = tokenize(query)
        if not q or self.n == 0:
            return []
        scores = []
        q_tf = Counter(q)
        for i, doc in enumerate(self.docs):
            dl = len(doc)
            if dl == 0:
                scores.append((0.0, i))
                continue
            tf = Counter(doc)
            s = 0.0
            for t, qf in q_tf.items():
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf * freq * (self.k1 + 1) / denom * qf
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.chunk_ids[i] for _, i in scores[:top_k]]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(rows: list[dict], chunks_by_source: dict[str, list[dict]], query_field: str, top_k: int):
    by_source: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    hits = {k: 0 for k in K_VALUES}
    hits_rel = {k: 0 for k in K_VALUES}
    results = []
    n = 0
    for source, group in by_source.items():
        article = chunks_by_source.get(str(source), [])
        ids = [c["chunk_id"] for c in article]
        texts = [c["text"] for c in article]
        bm25 = BM25Article(ids, texts)
        for r in group:
            ranked = bm25.rank(r[query_field] or "", top_k)
            exact, relaxed = ranks_from_ids(ranked, r["gold_chunk_id"])
            n += 1
            for k in K_VALUES:
                if exact is not None and exact <= k:
                    hits[k] += 1
                if relaxed is not None and relaxed <= k:
                    hits_rel[k] += 1
            results.append({
                "final_id": r["final_id"],
                "source": r["source"],
                "gold_chunk_id": r["gold_chunk_id"],
                "rank": exact,
                "rank_relaxed": relaxed,
                "gold_match_type": r["gold_match_type"],
                "judgement": r["judgement"],
                "type": r.get("type"),
                "query_field": query_field,
            })
            if n % 2000 == 0:
                print(f"  [{n}] Recall@1={hits[1]/n:.3f}  Recall@10={hits[10]/n:.3f}", flush=True)

    summary = {
        "query_field": query_field,
        "total_evaluated": n,
        **{f"recall_at_{k}": round(hits[k] / n, 4) for k in K_VALUES},
        "mrr": round(sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in results) / n, 4),
        **{f"relaxed_recall_at_{k}": round(hits_rel[k] / n, 4) for k in K_VALUES},
        "relaxed_mrr": round(
            sum((1.0 / r["rank_relaxed"]) if r["rank_relaxed"] else 0.0 for r in results) / n, 4
        ),
    }
    return summary, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--data-dir", type=Path, default=Path("data/arafa/localization"))
    parser.add_argument("--chunks", type=Path, default=Path("data/arafa/wikipedia_chunks.json"))
    parser.add_argument("--query-field", choices=["claim", "evidence"], default="claim")
    parser.add_argument("--oracle", action="store_true",
                        help="Also evaluate evidence-oracle BM25 on this split.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    jsonl = args.data_dir / f"{args.split}.jsonl"
    print(f"Loading {jsonl} ...")
    rows = load_jsonl(jsonl)
    needed = {str(r["source"]) for r in rows}
    print(f"Loading chunks for {len(needed)} articles ...")
    all_chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    chunks_by_source = {sid: all_chunks[sid]["chunks"] for sid in needed if sid in all_chunks}
    del all_chunks

    print(f"BM25 query={args.query_field} split={args.split} n={len(rows)}")
    summary, results = evaluate(rows, chunks_by_source, args.query_field, args.top_k)

    out = {"summary": summary, "results": results}
    if args.oracle and args.query_field != "evidence":
        print("Oracle BM25 (query=evidence) ...")
        o_sum, o_res = evaluate(rows, chunks_by_source, "evidence", args.top_k)
        out["oracle_summary"] = o_sum
        out["oracle_results"] = o_res

    default_name = f"inarticle_bm25_{args.split}_{args.query_field}.json"
    output = args.output or (args.data_dir / default_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print("\n=== In-article BM25 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if "oracle_summary" in out:
        print("\n=== Oracle (evidence query) ===")
        for k, v in out["oracle_summary"].items():
            print(f"  {k}: {v}")
    print(f"\nWritten to {output}")


if __name__ == "__main__":
    main()
