"""
build_localization_dataset.py

Filter ARAFA + gold_passages to supported/refuted claims with exact/fuzzy
gold chunks, then split by Wikipedia article (80/10/10, seed 42).

Usage:
    py scripts/build_localization_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10


def claim_count_bucket(n: int) -> str:
    if n <= 20:
        return "1-20"
    if n <= 50:
        return "21-50"
    if n <= 80:
        return "51-80"
    return "81+"


def stratified_article_split(source_to_n: dict[int, int], seed: int):
    rng = random.Random(seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for source, n in source_to_n.items():
        buckets[claim_count_bucket(n)].append(source)

    train, val, test = [], [], []
    for bucket in sorted(buckets):
        ids = buckets[bucket]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # keep at least 1 in val/test when the bucket is large enough
        if n >= 10:
            n_train = min(n_train, n - 2)
            n_val = max(1, min(n_val, n - n_train - 1))
        train.extend(ids[:n_train])
        val.extend(ids[n_train : n_train + n_val])
        test.extend(ids[n_train + n_val :])
    return set(train), set(val), set(test)


def judgement_mix(rows: list[dict]) -> dict:
    c = Counter(r["judgement"] for r in rows)
    m = Counter(r["gold_match_type"] for r in rows)
    return {
        "n_claims": len(rows),
        "n_articles": len({r["source"] for r in rows}),
        "judgement": dict(c),
        "gold_match_type": dict(m),
    }


def print_split_table(name: str, rows: list[dict]):
    m = judgement_mix(rows)
    print(f"\n{name}")
    print(f"  articles={m['n_articles']}  claims={m['n_claims']}")
    print(f"  judgement={m['judgement']}")
    print(f"  gold_match={m['gold_match_type']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arafa", type=Path, default=Path("data/arafa/ARAFA.json"))
    parser.add_argument("--gold", type=Path, default=Path("data/arafa/gold_passages.json"))
    parser.add_argument("--chunks", type=Path, default=Path("data/arafa/wikipedia_chunks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/arafa/localization"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("Loading ARAFA, gold passages, wikipedia chunks ...")
    claims = json.loads(args.arafa.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    chunks_data = json.loads(args.chunks.read_text(encoding="utf-8"))
    chunk_ids_by_source: dict[str, set[str]] = {}
    for sid, article in chunks_data.items():
        chunk_ids_by_source[str(sid)] = {
            c["chunk_id"] for c in article.get("chunks", [])
        }
    del chunks_data

    rows = []
    skipped = Counter()
    for c in claims:
        judgement = c.get("judgement")
        if judgement not in ("supported", "refuted"):
            skipped["not_supported_refuted"] += 1
            continue
        g = gold.get(str(c["final_id"])) or gold.get(c["final_id"])
        if not g:
            skipped["no_gold"] += 1
            continue
        if g.get("match_type") not in ("exact", "fuzzy"):
            skipped[f"match_{g.get('match_type')}"] += 1
            continue
        chunk_id = g.get("chunk_id")
        if not chunk_id:
            skipped["no_chunk_id"] += 1
            continue
        source = int(c["source"])
        ids = chunk_ids_by_source.get(str(source), set())
        if chunk_id not in ids:
            skipped["chunk_id_missing"] += 1
            continue
        rows.append({
            "final_id": c["final_id"],
            "claim": c["claim"],
            "evidence": c["evidence"],
            "source": source,
            "judgement": judgement,
            "type": c.get("type"),
            "gold_chunk_id": chunk_id,
            "gold_match_type": g["match_type"],
            "similarity": g.get("similarity"),
        })

    print(f"Eligible claims: {len(rows)}")
    print(f"Skipped: {dict(skipped)}")

    source_to_n = Counter(r["source"] for r in rows)
    train_s, val_s, test_s = stratified_article_split(source_to_n, args.seed)
    split_of = {}
    for s in train_s:
        split_of[s] = "train"
    for s in val_s:
        split_of[s] = "val"
    for s in test_s:
        split_of[s] = "test"

    by_split = defaultdict(list)
    for r in rows:
        by_split[split_of[r["source"]]].append(r)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        path = args.output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in by_split[name]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print_split_table(name, by_split[name])

    manifest = {
        "seed": args.seed,
        "split_unit": "source (Wikipedia article)",
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": round(1 - TRAIN_FRAC - VAL_FRAC, 2)},
        "skipped": dict(skipped),
        "train": {
            "article_ids": sorted(train_s),
            **judgement_mix(by_split["train"]),
        },
        "val": {
            "article_ids": sorted(val_s),
            **judgement_mix(by_split["val"]),
        },
        "test": {
            "article_ids": sorted(test_s),
            **judgement_mix(by_split["test"]),
        },
    }
    man_path = args.output_dir / "split_manifest.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nManifest: {man_path}")
    overlap = (train_s & val_s) | (train_s & test_s) | (val_s & test_s)
    print(f"Article overlap across splits: {len(overlap)} (must be 0)")


if __name__ == "__main__":
    main()
