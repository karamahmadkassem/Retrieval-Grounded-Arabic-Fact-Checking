"""
build_distractor_corpus.py

DEPRECATED for large-scale corpus building — use build_corpus_from_dump.py
instead.  The live API hard-caps exlimit to 1 for full-article extracts,
so batching multiple pageids in fetch_texts_batch() silently returns empty
text for all but the first page (~2%% false "stub" rate).  One-page-at-a-time
fetch works but is too slow at 100K+ scale.

Samples a small set of RANDOM Arabic Wikipedia articles (unrelated to
ARAFA's 3,301 known sources), chunks them with the same sentence-window
logic as build_gold_passages.py, and writes them as a distractor corpus.

This is what turns wikipedia_chunks.json from a "shortlist of correct
answers" into a real evaluation corpus with genuine haystack difficulty
-- the missing piece flagged before building BM25.

Output format: JSONL, one article per line (crash-safe -- unlike a
single large JSON file, losing power mid-run only costs the last
unflushed line, not the whole file):

    data/arafa/distractor_chunks.jsonl
        {"source": 12345, "title": "...", "url": "...",
         "num_sentences": 42, "chunks": [...]}
        {"source": 67890, ...}
        ...

Usage:
    python scripts/build_distractor_corpus.py --num-articles 100000
    python scripts/build_distractor_corpus.py --num-articles 100000 --resume
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arabic_text_utils import build_sentence_windows, segment_sentences

API_URL = "https://ar.wikipedia.org/w/api.php"
USER_AGENT = (
    "ARAFA-RetrievalThesis/0.1 "
    "(Research project: Retrieval-Grounded Arabic Fact-Checking; "
    "contact: kma88@mail.aub.edu)"
)
REQUEST_DELAY_SECONDS = 0.3
RANDOM_BATCH_SIZE = 500   # max allowed for list=random without bot rights
TEXT_BATCH_SIZE = 50      # max pageids per extracts request
MAX_WINDOW = 3
MIN_SENTENCES = 1         # skip near-empty extracts / disambiguation stubs


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def fetch_random_page_ids(session: requests.Session, n: int) -> set:
    """Fetch n random ARTICLE (namespace=0, non-redirect) page ids."""
    ids = set()
    while len(ids) < n:
        params = {
            "action": "query", "list": "random",
            "rnnamespace": 0, "rnfilterredir": "nonredirects",
            "rnlimit": min(RANDOM_BATCH_SIZE, n - len(ids)),
            "format": "json",
        }
        resp = session.get(API_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("query", {}).get("random", [])
        for page in batch:
            ids.add(page["id"])
        if not batch:
            break  # safety valve, shouldn't normally happen
        time.sleep(REQUEST_DELAY_SECONDS)
    return ids


def fetch_texts_batch(session: requests.Session, page_ids: list) -> dict:
    params = {
        "action": "query", "pageids": "|".join(str(p) for p in page_ids),
        "prop": "extracts|info", "explaintext": 1, "inprop": "url",
        "format": "json",
    }
    resp = session.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("query", {}).get("pages", {})


def chunk_article(text: str) -> tuple:
    sentences = segment_sentences(text)
    windows = build_sentence_windows(sentences, max_window=MAX_WINDOW)
    chunks = [
        {
            "chunk_id": f"{{source}}:{w['sentence_start']}-{w['sentence_end']}",
            "sentence_start": w["sentence_start"],
            "sentence_end": w["sentence_end"],
            "window_size": w["window_size"],
            "text": w["text"],
        }
        for w in windows
    ]
    return sentences, chunks


def load_known_sources(verification_path: Path) -> set:
    if not verification_path.exists():
        return set()
    data = json.loads(verification_path.read_text(encoding="utf-8"))
    return {int(k) for k, v in data.items() if v.get("found")}


def load_already_done(output_path: Path) -> set:
    done = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(obj["source"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a truncated last line from a prior crash
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-articles", type=int, default=100_000)
    parser.add_argument(
        "--verification", type=Path,
        default=Path("data/arafa/verification/source_verification.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/arafa/distractor_chunks.jsonl")
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    known_sources = load_known_sources(args.verification)
    print(f"Excluding {len(known_sources)} already-known ARAFA source articles.")

    already_done = load_already_done(args.output) if args.resume else set()
    if already_done:
        print(f"Resuming: {len(already_done)} distractor articles already written.")

    session = build_session()
    exclude = known_sources | already_done
    needed = args.num_articles - len(already_done)
    print(f"Need {needed} more distractor articles with usable text.", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if args.resume and args.output.exists() else "w"
    written = 0
    skipped_empty = 0
    pending_ids: list[int] = []

    with open(args.output, open_mode, encoding="utf-8") as out_f:
        while written < needed:
            while len(pending_ids) < TEXT_BATCH_SIZE and written + len(pending_ids) < needed + TEXT_BATCH_SIZE:
                oversample = max(RANDOM_BATCH_SIZE, (needed - written) // 2 + 50)
                batch = fetch_random_page_ids(session, oversample)
                batch -= exclude
                for pid in batch:
                    if pid not in exclude:
                        pending_ids.append(pid)
                        exclude.add(pid)

            batch_ids = pending_ids[:TEXT_BATCH_SIZE]
            pending_ids = pending_ids[TEXT_BATCH_SIZE:]
            if not batch_ids:
                print("  warning: no new ids sampled; retrying ...", flush=True)
                continue

            try:
                pages = fetch_texts_batch(session, batch_ids)
            except requests.RequestException as e:
                print(f"  batch fetch error: {e}", flush=True)
                time.sleep(2)
                continue

            for pid_str, page in pages.items():
                if written >= needed:
                    break
                if "missing" in page:
                    skipped_empty += 1
                    continue
                text = page.get("extract", "")
                source_id = int(pid_str)
                sentences, chunks = chunk_article(text)
                if len(sentences) < MIN_SENTENCES or not chunks:
                    skipped_empty += 1
                    continue
                for c in chunks:
                    c["chunk_id"] = c["chunk_id"].format(source=source_id)
                record = {
                    "source": source_id,
                    "title": page.get("title"),
                    "url": page.get("fullurl"),
                    "num_sentences": len(sentences),
                    "chunks": chunks,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            out_f.flush()
            if written % 100 == 0 or written == needed:
                print(
                    f"  {written}/{needed} articles written "
                    f"(skipped {skipped_empty} empty/stub pages so far) ...",
                    flush=True,
                )
            time.sleep(REQUEST_DELAY_SECONDS)

    total_now = len(already_done) + written
    print(f"\n=== Summary ===")
    print(f"Articles written this run: {written}")
    print(f"Total distractor articles in corpus: {total_now}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()