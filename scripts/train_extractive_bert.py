"""
train_extractive_bert.py

SQuAD-style span extraction: claim + known article → one evidence span.

Gold comes from localization gold_chunk_id (sentence_start–sentence_end).
Long pages are split into overlapping 6-sentence crops (stride 3) so BERT-base (384 tokens) sees the gold span.

Usage:
    py scripts/train_extractive_bert.py --smoke
    py scripts/train_extractive_bert.py --epochs 1 --eval-split val
    py scripts/train_extractive_bert.py --eval-only --checkpoint models/extractive_bert
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localization_metrics import (
    exact_match,
    parse_span,
    span_iou,
    summarize_spans,
    token_f1,
)
from run_inarticle_bm25_eval import load_jsonl

DEFAULT_MODEL = "bert-base-multilingual-cased"


def sentences_from_chunks(chunks: list[dict]) -> list[str] | None:
    ones = [c for c in chunks if c.get("window_size") == 1]
    if not ones:
        return None
    ones.sort(key=lambda c: c["sentence_start"])
    n = ones[-1]["sentence_start"] + 1
    by_i = {c["sentence_start"]: c["text"] for c in ones}
    if any(i not in by_i for i in range(n)):
        return None
    return [by_i[i] for i in range(n)]


def join_sentences(sentences: list[str]) -> tuple[str, list[int]]:
    """Article text and start-char of each sentence; last entry is len(text)."""
    starts = []
    parts = []
    pos = 0
    for i, sent in enumerate(sentences):
        if i:
            parts.append(" ")
            pos += 1
        starts.append(pos)
        parts.append(sent)
        pos += len(sent)
    starts.append(pos)
    return "".join(parts), starts


def gold_char_span(sentences: list[str], starts: list[int], sent_s: int, sent_e: int) -> tuple[int, int]:
    return starts[sent_s], starts[sent_e] + len(sentences[sent_e])


def char_span_to_sentences(starts: list[int], char_s: int, char_e: int) -> tuple[int, int] | None:
    n = len(starts) - 1
    pred_s = pred_e = None
    for i in range(n):
        a, b = starts[i], starts[i + 1]
        if char_e > a and char_s < b:
            if pred_s is None:
                pred_s = i
            pred_e = i
    if pred_s is None:
        return None
    return pred_s, pred_e


def load_chunks_for_sources(chunks_path: Path, sources: set[str]) -> dict[str, list[dict]]:
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    out = {s: data[s]["chunks"] for s in sources if s in data}
    del data
    gc.collect()
    return out


def build_examples(rows, chunks_by_source):
    examples = []
    skipped = 0
    sent_cache: dict[str, list[str] | None] = {}
    for r in rows:
        sid = str(r["source"])
        if sid not in sent_cache:
            sent_cache[sid] = sentences_from_chunks(chunks_by_source.get(sid, []))
        sentences = sent_cache[sid]
        span = parse_span(r["gold_chunk_id"])
        if not sentences or span is None:
            skipped += 1
            continue
        sent_s, sent_e = span
        if sent_s < 0 or sent_e >= len(sentences) or sent_s > sent_e:
            skipped += 1
            continue
        gold_text = " ".join(sentences[sent_s : sent_e + 1]).strip()
        if not gold_text:
            skipped += 1
            continue
        examples.append({
            "final_id": r["final_id"],
            "source": r["source"],
            "claim": r["claim"] or "",
            "sentences": sentences,
            "gold_text": gold_text,
            "gold_chunk_id": r["gold_chunk_id"],
            "sent_start": sent_s,
            "sent_end": sent_e,
            "gold_match_type": r["gold_match_type"],
            "judgement": r["judgement"],
        })
    return examples, skipped


def iter_sentence_crops(n: int, window: int, stride: int):
    if n <= window:
        yield 0, n
        return
    i = 0
    while i < n:
        j = min(n, i + window)
        yield i, j
        if j == n:
            break
        i += stride


def clip_claim(tokenizer, claim: str, max_length: int, min_context: int = 64) -> str:
    """Keep the claim short enough that only_second truncation still has a context budget."""
    special = tokenizer.num_special_tokens_to_add(pair=True)
    q_max = max(8, max_length - special - min_context)
    ids = tokenizer.encode(
        claim, add_special_tokens=False, truncation=True, max_length=q_max,
    )
    return tokenizer.decode(ids, skip_special_tokens=True)


def encode_crop(tokenizer, claim, crop_text, max_length):
    if not (crop_text or "").strip():
        return None
    claim = clip_claim(tokenizer, claim or "", max_length)
    if not claim.strip():
        claim = "?"
    try:
        return tokenizer(
            claim,
            crop_text,
            truncation="only_second",
            max_length=max_length,
            return_offsets_mapping=True,
            padding="max_length",
        )
    except Exception:
        return None


def token_span_in_crop(enc, gold_start, gold_end):
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    seq = enc.sequence_ids()
    ctx_idx = [
        j for j, s in enumerate(seq)
        if s == 1 and offsets[j] != (0, 0)
    ]
    if not ctx_idx:
        return None
    c0, c1 = ctx_idx[0], ctx_idx[-1]
    if offsets[c0][0] > gold_start or offsets[c1][1] < gold_end:
        return None
    t0 = c0
    while t0 <= c1 and offsets[t0][0] <= gold_start:
        t0 += 1
    t0 -= 1
    t1 = c1
    while t1 >= c0 and offsets[t1][1] >= gold_end:
        t1 -= 1
    t1 += 1
    if t0 < c0 or t1 > c1 or t0 > t1:
        return None
    return t0, t1


def crop_with_gold(ex, crop_s, crop_e):
    if ex["sent_start"] < crop_s or ex["sent_end"] >= crop_e:
        return None
    local = ex["sentences"][crop_s:crop_e]
    crop_text, local_starts = join_sentences(local)
    loc_s = ex["sent_start"] - crop_s
    loc_e = ex["sent_end"] - crop_s
    if loc_s < 0 or loc_e >= len(local) or loc_s > loc_e:
        return None
    cs, ce = gold_char_span(local, local_starts, loc_s, loc_e)
    if ce <= cs:
        return None
    return crop_text, cs, ce, local_starts


def featurize_train(examples, tokenizer, max_length, crop_sents, crop_stride):
    all_ids, all_mask, all_start, all_end = [], [], [], []
    n_keep = 0
    for ex in tqdm(examples, desc="featurize"):
        n = len(ex["sentences"])
        for crop_s, crop_e in iter_sentence_crops(n, crop_sents, crop_stride):
            packed = crop_with_gold(ex, crop_s, crop_e)
            if packed is None:
                continue
            crop_text, cs, ce, _ = packed
            enc = encode_crop(tokenizer, ex["claim"], crop_text, max_length)
            if enc is None:
                continue
            span = token_span_in_crop(enc, cs, ce)
            if span is None:
                continue
            t0, t1 = span
            all_ids.append(enc["input_ids"])
            all_mask.append(enc["attention_mask"])
            all_start.append(t0)
            all_end.append(t1)
            n_keep += 1
    print(f"Train windows with gold span: {n_keep}", flush=True)
    return {
        "input_ids": all_ids,
        "attention_mask": all_mask,
        "start_positions": all_start,
        "end_positions": all_end,
    }, n_keep


@torch.no_grad()
def predict_example(model, tokenizer, ex, max_length, crop_sents, crop_stride, max_answer_len, device):
    best = None
    n = len(ex["sentences"])
    for crop_s, crop_e in iter_sentence_crops(n, crop_sents, crop_stride):
        local = ex["sentences"][crop_s:crop_e]
        crop_text, local_starts = join_sentences(local)
        enc = encode_crop(tokenizer, ex["claim"], crop_text, max_length)
        if enc is None:
            continue
        ids = torch.tensor([enc["input_ids"]], device=device)
        mask = torch.tensor([enc["attention_mask"]], device=device)
        out = model(input_ids=ids, attention_mask=mask)
        start_l = out.start_logits[0].cpu()
        end_l = out.end_logits[0].cpu()
        seq = enc.sequence_ids()
        offsets = enc["offset_mapping"]
        ctx = [j for j, s in enumerate(seq) if s == 1 and offsets[j] != (0, 0)]
        if not ctx:
            continue
        ctx_set = set(ctx)
        k = min(20, len(ctx))
        starts = torch.topk(start_l, k).indices.tolist()
        ends = torch.topk(end_l, k).indices.tolist()
        for s in starts:
            for e in ends:
                if e < s or (e - s + 1) > max_answer_len:
                    continue
                if s not in ctx_set or e not in ctx_set:
                    continue
                a, b = offsets[s][0], offsets[e][1]
                if b <= a:
                    continue
                score = float(start_l[s] + end_l[e])
                if best is None or score > best[0]:
                    best = (score, a, b, crop_text, local_starts, crop_s)
    if best is None:
        return "", None
    _, a, b, crop_text, local_starts, crop_s = best
    pred = crop_text[a:b]
    local_sent = char_span_to_sentences(local_starts, a, b)
    if local_sent is None:
        return pred, None
    return pred, (local_sent[0] + crop_s, local_sent[1] + crop_s)


def evaluate(model, tokenizer, examples, args, device):
    model.eval()
    rows = []
    for ex in tqdm(examples, desc="eval"):
        pred, sent = predict_example(
            model, tokenizer, ex, args.max_length, args.crop_sents,
            args.crop_stride, args.max_answer_len, device,
        )
        gold_s, gold_e = ex["sent_start"], ex["sent_end"]
        if sent is None:
            iou = 0.0
            pred_id = ""
        else:
            pred_id = f"{ex['source']}:{sent[0]}-{sent[1]}"
            iou = span_iou(pred_id, ex["gold_chunk_id"])
        rows.append({
            "final_id": ex["final_id"],
            "source": ex["source"],
            "gold_chunk_id": ex["gold_chunk_id"],
            "pred_chunk_id": pred_id,
            "pred_text": pred,
            "gold_text": ex["gold_text"],
            "exact_match": exact_match(pred, ex["gold_text"]),
            "token_f1": round(token_f1(pred, ex["gold_text"]), 4),
            "sentence_iou": round(iou, 4),
            "gold_match_type": ex["gold_match_type"],
            "judgement": ex["judgement"],
            "gold_sent": [gold_s, gold_e],
            "pred_sent": list(sent) if sent else None,
        })
    return summarize_spans(rows), rows


def train_model(args, train_examples, tokenizer):
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from transformers import AutoModelForQuestionAnswering, get_linear_schedule_with_warmup

    feats, n_keep = featurize_train(
        train_examples, tokenizer, args.max_length, args.crop_sents, args.crop_stride
    )
    del train_examples
    gc.collect()
    if n_keep == 0:
        raise SystemExit("No training windows contained the gold span.")
    print("Loading QA model ...", flush=True)

    ids = torch.tensor(feats["input_ids"], dtype=torch.long)
    mask = torch.tensor(feats["attention_mask"], dtype=torch.long)
    start_pos = torch.tensor(feats["start_positions"], dtype=torch.long)
    end_pos = torch.tensor(feats["end_positions"], dtype=torch.long)
    del feats
    gc.collect()

    class FeatDS(TorchDataset):
        def __len__(self):
            return n_keep

        def __getitem__(self, i):
            return {
                "input_ids": ids[i],
                "attention_mask": mask[i],
                "start_positions": start_pos[i],
                "end_positions": end_pos[i],
            }

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)
    model.to(device)
    loader = DataLoader(FeatDS(), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    n_steps = max(1, args.epochs * len(loader))
    sched = get_linear_schedule_with_warmup(opt, min(50, n_steps // 10), n_steps)
    model.train()
    print(f"Training {n_steps} steps on {device} ...", flush=True)
    step = 0
    for _ in range(args.epochs):
        for batch in loader:
            step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if step == 1 or step % 200 == 0 or step == n_steps:
                print(f"step {step}/{n_steps} loss={out.loss.item():.4f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved {args.output_dir}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=Path("data/arafa/localization"))
    parser.add_argument("--chunks", type=Path, default=Path("data/arafa/wikipedia_chunks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/extractive_bert"))
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--crop-sents", type=int, default=6, help="Sentences per BERT crop")
    parser.add_argument("--crop-stride", type=int, default=3)
    parser.add_argument("--max-answer-len", type=int, default=80)
    parser.add_argument("--max-train-claims", type=int, default=None)
    parser.add_argument("--max-eval-claims", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="2000 train / 400 val")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.smoke:
        args.max_train_claims = args.max_train_claims or 2000
        args.max_eval_claims = args.max_eval_claims or 400

    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    eval_rows = load_jsonl(args.data_dir / f"{args.eval_split}.jsonl")
    rng = random.Random(args.seed)
    if args.max_train_claims and len(train_rows) > args.max_train_claims:
        rng.shuffle(train_rows)
        train_rows = train_rows[: args.max_train_claims]
    if args.max_eval_claims is not None:
        eval_rows = eval_rows[: args.max_eval_claims]

    sources = {str(r["source"]) for r in (eval_rows if args.eval_only else train_rows + eval_rows)}
    print(f"Loading chunks for {len(sources)} articles ...")
    chunks_by_source = load_chunks_for_sources(args.chunks, sources)

    eval_examples, skip_e = build_examples(eval_rows, chunks_by_source)
    print(f"Eval examples: {len(eval_examples)} (skipped {skip_e})")

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint or args.model if args.eval_only else args.model
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_only:
        path = args.checkpoint or args.output_dir
        model = AutoModelForQuestionAnswering.from_pretrained(str(path))
    else:
        train_examples, skip_t = build_examples(train_rows, chunks_by_source)
        print(f"Train examples: {len(train_examples)} (skipped {skip_t})", flush=True)
        del chunks_by_source, train_rows
        gc.collect()
        model = train_model(args, train_examples, tokenizer)

    model.to(device)
    if not eval_examples:
        print("No eval examples; skipped eval.", flush=True)
        return
    summary, results = evaluate(model, tokenizer, eval_examples, args, device)
    out_path = args.data_dir / f"extractive_bert_{args.eval_split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "model": args.model, "results": results},
            f, ensure_ascii=False,
        )
    print(summary)
    print(f"Written {out_path}")


if __name__ == "__main__":
    main()
