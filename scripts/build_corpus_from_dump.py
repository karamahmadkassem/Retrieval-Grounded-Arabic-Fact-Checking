"""
build_corpus_from_dump.py

Build the evaluation retrieval corpus from a local Arabic Wikipedia dump
processed by WikiExtractor — offline, no API rate limits.

WikiExtractor emits one JSON object per line:
    {"id": "12345", "title": "...", "url": "https://...", "text": "..."}

This script applies the same chunking as build_gold_passages.py
(arabic_text_utils.segment_sentences + build_sentence_windows), excludes
ARAFA's 3,301 known source articles (already in wikipedia_chunks.json),
and writes crash-safe JSONL.

Distractor articles default to single-sentence chunks (--max-window 1).
The gold corpus (wikipedia_chunks.json) keeps 1-3 sentence windows for
label precision; tripling distractor window sizes only inflates storage
and indexing cost without improving evaluation validity.

    data/arafa/wiki_corpus_chunks.jsonl

Prerequisites:
    pip install wikiextractor
    # download arwiki-latest-pages-articles.xml.bz2 from dumps.wikimedia.org
    python -m wikiextractor.WikiExtractor arwiki-latest-pages-articles.xml.bz2 \\
        --json -o data/arwiki/extracted --processes 4

Usage:
    py scripts/build_corpus_from_dump.py
    py scripts/build_corpus_from_dump.py --extract-dir data/arwiki/extracted
    py scripts/build_corpus_from_dump.py --max-articles 5000   # smoke test
    py scripts/build_corpus_from_dump.py --max-window 1        # default
    py scripts/build_corpus_from_dump.py --resume
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arabic_text_utils import build_sentence_windows, segment_sentences

MIN_SENTENCES = 1
CHECKPOINT_EVERY = 5000


def chunk_article(source_id: int, text: str, max_window: int) -> tuple[list, list]:
    sentences = segment_sentences(text)
    windows = build_sentence_windows(sentences, max_window=max_window)
    chunks = [
        {
            "chunk_id": f"{source_id}:{w['sentence_start']}-{w['sentence_end']}",
            "sentence_start": w["sentence_start"],
            "sentence_end": w["sentence_end"],
            "window_size": w["window_size"],
            "text": w["text"],
        }
        for w in windows
    ]
    return sentences, chunks


def load_known_sources(verification_path: Path) -> set[int]:
    if not verification_path.exists():
        return set()
    data = json.loads(verification_path.read_text(encoding="utf-8"))
    return {int(k) for k, v in data.items() if v.get("found")}


def load_done_sources(output_path: Path) -> set[int]:
    done: set[int] = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["source"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def iter_extract_files(extract_dir: Path):
    """Yield all WikiExtractor output files (wiki_00, wiki_01, ...)."""
    yield from sorted(p for p in extract_dir.rglob("wiki_*") if p.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data/arwiki/extracted"),
        help="WikiExtractor output directory (--json mode).",
    )
    parser.add_argument(
        "--verification",
        type=Path,
        default=Path("data/arafa/verification/source_verification.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/arafa/wiki_corpus_chunks.jsonl"),
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Cap articles written (for smoke tests). Default: process all.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-window",
        type=int,
        default=1,
        help="Max sentence window size for distractor chunks (default: 1).",
    )
    args = parser.parse_args()

    if not args.extract_dir.exists():
        raise SystemExit(
            f"Extract dir not found: {args.extract_dir}\n"
            "Run WikiExtractor first (see script docstring)."
        )

    known = load_known_sources(args.verification)
    done = load_done_sources(args.output) if args.resume else set()
    if done:
        print(f"Resuming: {len(done)} articles already in {args.output}")

    extract_files = list(iter_extract_files(args.extract_dir))
    if not extract_files:
        raise SystemExit(f"No wiki_* files under {args.extract_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if args.resume and args.output.exists() else "w"

    written = 0
    skipped_known = 0
    skipped_empty = 0
    skipped_done = 0

    with open(args.output, open_mode, encoding="utf-8") as out_f:
        for extract_path in extract_files:
            if args.max_articles and written >= args.max_articles:
                break

            with open(extract_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if args.max_articles and written >= args.max_articles:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        article = json.loads(line)
                        source_id = int(article["id"])
                    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                        continue

                    if source_id in known:
                        skipped_known += 1
                        continue
                    if source_id in done:
                        skipped_done += 1
                        continue

                    text = article.get("text") or ""
                    sentences, chunks = chunk_article(source_id, text, args.max_window)
                    if len(sentences) < MIN_SENTENCES or not chunks:
                        skipped_empty += 1
                        continue

                    record = {
                        "source": source_id,
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "num_sentences": len(sentences),
                        "chunks": chunks,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    done.add(source_id)
                    written += 1

                    if written % CHECKPOINT_EVERY == 0:
                        out_f.flush()
                        print(
                            f"  {written} articles written "
                            f"(skipped known={skipped_known}, empty={skipped_empty})",
                            flush=True,
                        )

    print("\n=== Summary ===")
    print(f"Articles written this run: {written}")
    print(f"Total in output file:      {len(done)}")
    print(f"Skipped (ARAFA known):     {skipped_known}")
    print(f"Skipped (empty/stub):      {skipped_empty}")
    print(f"Skipped (already done):    {skipped_done}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
