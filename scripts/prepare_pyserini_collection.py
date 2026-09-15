"""
prepare_pyserini_collection.py

Flattens wikipedia_chunks.json (gold corpus, all window sizes) and
wiki_corpus_chunks.jsonl (distractor corpus, window_size==1 only) into
Pyserini's JsonCollection format: sharded JSONL files of
{"id": chunk_id, "contents": text}.

Usage:
    py scripts/prepare_pyserini_collection.py
    py scripts/prepare_pyserini_collection.py --skip-distractor   # gold-only
    py scripts/prepare_pyserini_collection.py --max-articles 5000  # smoke test
"""

import argparse
import json
from pathlib import Path

SHARD_SIZE = 500_000  # docs per output file


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, shard_size: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self.shard_idx = 0
        self.count_in_shard = 0
        self.total = 0
        self.f = None
        self._open_new_shard()

    def _open_new_shard(self):
        if self.f:
            self.f.close()
        path = self.out_dir / f"{self.prefix}_{self.shard_idx:05d}.jsonl"
        self.f = open(path, "w", encoding="utf-8")
        self.shard_idx += 1
        self.count_in_shard = 0

    def write(self, doc: dict):
        if self.count_in_shard >= self.shard_size:
            self._open_new_shard()
        self.f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self.count_in_shard += 1
        self.total += 1

    def close(self):
        if self.f:
            self.f.close()


def write_gold_corpus(gold_path: Path, writer: ShardWriter):
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    for source_id, article in data.items():
        for chunk in article["chunks"]:
            writer.write({"id": chunk["chunk_id"], "contents": chunk["text"]})
    print(f"Gold corpus: {writer.total} chunks written so far.")


def write_distractor_corpus(distractor_path: Path, writer: ShardWriter, max_articles: int = None):
    n_articles = 0
    n_chunks_before = writer.total
    with open(distractor_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            for chunk in article["chunks"]:
                if chunk["window_size"] != 1:
                    continue  # distractor corpus is single-sentence only
                writer.write({"id": chunk["chunk_id"], "contents": chunk["text"]})
            n_articles += 1
            if max_articles and n_articles >= max_articles:
                break
    print(f"Distractor corpus: {n_articles} articles, "
          f"{writer.total - n_chunks_before} chunks written.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("data/arafa/wikipedia_chunks.json"))
    parser.add_argument("--distractor", type=Path, default=Path("data/arafa/wiki_corpus_chunks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/arafa/pyserini_collection"))
    parser.add_argument("--max-articles", type=int, default=None,
                         help="Cap on distractor articles, for smoke tests.")
    parser.add_argument(
        "--skip-distractor",
        action="store_true",
        help="Index gold corpus only (skip distractor JSONL).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(args.output_dir, "docs", SHARD_SIZE)

    write_gold_corpus(args.gold, writer)

    if args.skip_distractor:
        print("Skipping distractor corpus (--skip-distractor).")
    elif not args.distractor.exists():
        print(f"Distractor file not found: {args.distractor} — skipping distractor corpus.")
    else:
        write_distractor_corpus(args.distractor, writer, args.max_articles)

    writer.close()
    print(f"\nTotal chunks written: {writer.total}")
    print(f"Shards: {writer.shard_idx} file(s) in {args.output_dir}")


if __name__ == "__main__":
    main()