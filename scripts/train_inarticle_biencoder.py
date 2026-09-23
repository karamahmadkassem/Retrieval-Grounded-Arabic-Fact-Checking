"""
train_inarticle_biencoder.py

Fine-tune a multilingual E5/BGE bi-encoder for in-article evidence
localization. Hard negatives: in-article BM25 near-misses.

Usage:
    py scripts/train_inarticle_biencoder.py --epochs 1 --max-train-claims 4000
    py scripts/train_inarticle_biencoder.py --eval-only --checkpoint models/inarticle_biencoder
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arabic_text_utils import normalize
from localization_metrics import K_VALUES, ranks_from_ids, summarize
from run_inarticle_bm25_eval import BM25Article, load_jsonl

E5_PREFIX_QUERY = "query: "
E5_PREFIX_PASSAGE = "passage: "


def is_e5(name: str) -> bool:
    return "e5" in name.lower()


def q_text(model_name: str, text: str) -> str:
    return (E5_PREFIX_QUERY + text) if is_e5(model_name) else text


def p_text(model_name: str, text: str) -> str:
    return (E5_PREFIX_PASSAGE + text) if is_e5(model_name) else text


def load_chunks_for_sources(chunks_path: Path, sources: set[str]) -> dict[str, list[dict]]:
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    return {s: data[s]["chunks"] for s in sources if s in data}


def bm25_negatives(row, chunks, n_neg, rng: random.Random) -> list[str]:
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    ranked = BM25Article(ids, texts).rank(row["claim"], top_k=max(40, n_neg * 5))
    negs = [t for t in ranked if t != row["gold_chunk_id"]]
    if len(negs) < n_neg:
        rest = [i for i in ids if i != row["gold_chunk_id"] and i not in negs]
        rng.shuffle(rest)
        negs.extend(rest)
    return negs[:n_neg]


def build_train_examples(rows, chunks_by_source, model_name, n_neg, max_claims, seed):
    rng = random.Random(seed)
    if max_claims and len(rows) > max_claims:
        rows = rng.sample(rows, max_claims)
    examples = []
    text_of = {}
    for sid, chs in chunks_by_source.items():
        for c in chs:
            text_of[c["chunk_id"]] = c["text"]
    for r in tqdm(rows, desc="hard negatives"):
        chs = chunks_by_source.get(str(r["source"]), [])
        if not chs:
            continue
        gold_text = text_of.get(r["gold_chunk_id"])
        if not gold_text:
            continue
        negs = bm25_negatives(r, chs, n_neg, rng)
        examples.append({
            "query": q_text(model_name, r["claim"]),
            "positive": p_text(model_name, gold_text),
            "negatives": [p_text(model_name, text_of[nid]) for nid in negs if nid in text_of],
        })
    return examples


def train_model(args, examples):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    model = SentenceTransformer(args.model)
    payload = {
        "anchor": [ex["query"] for ex in examples],
        "positive": [ex["positive"] for ex in examples],
    }
    if examples and examples[0]["negatives"]:
        payload["negative"] = [ex["negatives"][0] for ex in examples]
    ds = Dataset.from_dict(payload)
    loss = MultipleNegativesRankingLoss(model)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        warmup_steps=50,
        logging_strategy="no",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to=[],
    )
    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=ds, loss=loss,
    )
    trainer.train()
    model.save(str(args.output_dir))
    return model


@torch.no_grad()
def encode_passages(model, texts, batch_size):
    return model.encode(
        texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True
    )


def evaluate_split(model, model_name, rows, chunks_by_source, top_k, batch_size):
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    results = []
    for source, group in tqdm(by_source.items(), desc="eval articles"):
        chs = chunks_by_source.get(str(source), [])
        ids = [c["chunk_id"] for c in chs]
        texts = [p_text(model_name, c["text"]) for c in chs]
        if not ids:
            continue
        P = encode_passages(model, texts, batch_size)
        queries = [q_text(model_name, r["claim"]) for r in group]
        Q = encode_passages(model, queries, batch_size)
        scores = Q @ P.T
        for i, r in enumerate(group):
            order = torch.argsort(scores[i], descending=True)[:top_k].tolist()
            ranked = [ids[j] for j in order]
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
                "top_ids": ranked[:50],
            })
    n = len(results)
    summary = {
        "total_evaluated": n,
        **{f"recall_at_{k}": round(sum(1 for r in results if r["rank"] and r["rank"] <= k) / n, 4)
           for k in K_VALUES + [50]},
        "mrr": round(sum((1.0 / r["rank"]) if r["rank"] else 0.0 for r in results) / n, 4),
    }
    # recall@50 for stage-1 gate
    r50 = sum(1 for r in results if r["rank"] and r["rank"] <= 50) / n if n else 0
    summary["recall_at_50"] = round(r50, 4)
    return summary, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--data-dir", type=Path, default=Path("data/arafa/localization"))
    parser.add_argument("--chunks", type=Path, default=Path("data/arafa/wikipedia_chunks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/inarticle_biencoder"))
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-negatives", type=int, default=7)
    parser.add_argument("--max-train-claims", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-eval-claims", type=int, default=None)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    eval_rows = load_jsonl(args.data_dir / f"{args.eval_split}.jsonl")
    if args.max_eval_claims:
        eval_rows = eval_rows[: args.max_eval_claims]
    sources = {str(r["source"]) for r in train_rows + eval_rows}
    if args.max_train_claims:
        random.Random(args.seed).shuffle(train_rows)
        train_rows = train_rows[: args.max_train_claims]
        sources = {str(r["source"]) for r in train_rows + eval_rows}

    print("Loading article chunks ...")
    chunks_by_source = load_chunks_for_sources(args.chunks, sources)

    if args.eval_only:
        path = args.checkpoint or args.output_dir
        model = SentenceTransformer(str(path))
        model_name = args.model
    else:
        print(f"Building training triples (n_neg={args.n_negatives}) ...")
        examples = build_train_examples(
            train_rows, chunks_by_source, args.model, args.n_negatives,
            None, args.seed,
        )
        print(f"Train examples: {len(examples)}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        model = train_model(args, examples)
        model_name = args.model

    print(f"Evaluating {args.eval_split} ...")
    summary, results = evaluate_split(
        model, model_name, eval_rows, chunks_by_source, args.top_k, args.batch_size
    )
    out_path = args.data_dir / f"biencoder_{args.eval_split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "model": args.model, "results": results}, f, ensure_ascii=False)
    print(summary)
    print(f"Written {out_path}")


if __name__ == "__main__":
    main()
