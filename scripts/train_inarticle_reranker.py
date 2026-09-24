"""
train_inarticle_reranker.py

Fine-tune a cross-encoder on hybrid (BM25 ∪ bi-encoder) hard negatives,
then rerank Stage-1 top-50.

Usage:
    py scripts/train_inarticle_reranker.py --max-train-claims 2000 --epochs 1
    py scripts/train_inarticle_reranker.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localization_metrics import K_VALUES, ranks_from_ids
from run_inarticle_bm25_eval import BM25Article, load_jsonl
from train_inarticle_biencoder import load_chunks_for_sources


def rrf_merge(lists: list[list[str]], k: int = 60, top_n: int = 50) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in lists:
        for i, cid in enumerate(ranked):
            scores[cid] += 1.0 / (k + i + 1)
    return [c for c, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]


def bm25_top(row, chunks, top_k) -> list[str]:
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    return BM25Article(ids, texts).rank(row["claim"], top_k)


def build_ce_examples(rows, chunks_by_source, n_neg, max_claims, seed):
    rng = random.Random(seed)
    if max_claims and len(rows) > max_claims:
        rows = rng.sample(rows, max_claims)
    text_of = {}
    for chs in chunks_by_source.values():
        for c in chs:
            text_of[c["chunk_id"]] = c["text"]
    examples = []
    from sentence_transformers import InputExample

    for r in tqdm(rows, desc="reranker pairs"):
        chs = chunks_by_source.get(str(r["source"]), [])
        gold = r["gold_chunk_id"]
        gold_text = text_of.get(gold)
        if not gold_text:
            continue
        ranked = bm25_top(r, chs, 40)
        negs = [i for i in ranked if i != gold][:n_neg]
        examples.append(InputExample(texts=[r["claim"], gold_text], label=1.0))
        for nid in negs:
            examples.append(InputExample(texts=[r["claim"], text_of[nid]], label=0.0))
    return examples


def train_ce(args, examples):
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    model = CrossEncoder(args.model, num_labels=1, max_length=512)
    ds = Dataset.from_dict({
        "sentence_1": [ex.texts[0] for ex in examples],
        "sentence_2": [ex.texts[1] for ex in examples],
        "label": [float(ex.label) for ex in examples],
    })
    targs = CrossEncoderTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=50,
        logging_strategy="no",
        save_strategy="no",
        fp16=__import__("torch").cuda.is_available(),
        use_cpu=not __import__("torch").cuda.is_available(),
        dataloader_num_workers=0,
        report_to=[],
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=targs,
        train_dataset=ds,
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train()
    model.save(str(args.output_dir))
    return model


def evaluate(ce, rows, chunks_by_source, dense_by_id, top_k=50, batch_size=16):
    results = []
    for r in tqdm(rows, desc="hybrid+rerank"):
        chs = chunks_by_source.get(str(r["source"]), [])
        text_of = {c["chunk_id"]: c["text"] for c in chs}
        bm25_ids = bm25_top(r, chs, top_k)
        dense_ids = (dense_by_id.get(r["final_id"]) or {}).get("top_ids") or []
        fused = rrf_merge([bm25_ids, dense_ids], top_n=top_k) if dense_ids else bm25_ids
        pairs = [[r["claim"], text_of[cid]] for cid in fused if cid in text_of]
        ids = [cid for cid in fused if cid in text_of]
        if not pairs:
            exact = relaxed = None
            ranked = []
        else:
            scores = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            order = sorted(range(len(ids)), key=lambda i: float(scores[i]), reverse=True)
            ranked = [ids[i] for i in order]
            exact, relaxed = ranks_from_ids(ranked, r["gold_chunk_id"])
        results.append({
            "final_id": r["final_id"],
            "source": r["source"],
            "gold_chunk_id": r["gold_chunk_id"],
            "rank": exact,
            "rank_relaxed": relaxed,
            "gold_match_type": r["gold_match_type"],
            "judgement": r["judgement"],
            "type": r.get("type"),
        })
    n = len(results)
    summary = {
        "total_evaluated": n,
        **{f"recall_at_{k}": round(sum(1 for x in results if x["rank"] and x["rank"] <= k) / n, 4)
           for k in K_VALUES},
        "mrr": round(sum((1.0 / x["rank"]) if x["rank"] else 0.0 for x in results) / n, 4),
    }
    return summary, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        help="CPU default is mMiniLM; use BAAI/bge-reranker-v2-m3 on GPU/Octopus.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/arafa/localization"))
    parser.add_argument("--chunks", type=Path, default=Path("data/arafa/wikipedia_chunks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/inarticle_reranker"))
    parser.add_argument("--dense-results", type=Path, default=Path("data/arafa/localization/biencoder_test.json"))
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-negatives", type=int, default=4)
    parser.add_argument("--max-train-claims", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-eval-claims", type=int, default=None)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder

    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    eval_rows = load_jsonl(args.data_dir / f"{args.eval_split}.jsonl")
    if args.max_eval_claims:
        eval_rows = eval_rows[: args.max_eval_claims]
    sources = {str(r["source"]) for r in eval_rows}
    if not args.eval_only:
        if args.max_train_claims:
            random.Random(args.seed).shuffle(train_rows)
            train_rows = train_rows[: args.max_train_claims]
        sources |= {str(r["source"]) for r in train_rows}

    print("Loading chunks ...")
    chunks_by_source = load_chunks_for_sources(args.chunks, sources)

    if args.eval_only:
        ce = CrossEncoder(str(args.checkpoint or args.model), max_length=512)
    else:
        examples = build_ce_examples(
            train_rows, chunks_by_source, args.n_negatives, None, args.seed
        )
        print(f"CE training pairs: {len(examples)}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        ce = train_ce(args, examples)

    dense = []
    if args.dense_results.exists():
        dense = json.loads(args.dense_results.read_text(encoding="utf-8")).get("results", [])
    dense_by_id = {r["final_id"]: r for r in dense}

    summary, results = evaluate(ce, eval_rows, chunks_by_source, dense_by_id)
    out_path = args.data_dir / f"reranker_{args.eval_split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "model": args.model, "results": results}, f, ensure_ascii=False)
    print(summary)
    print(f"Written {out_path}")


if __name__ == "__main__":
    main()
