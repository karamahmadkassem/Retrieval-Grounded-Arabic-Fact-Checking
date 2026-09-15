"""
build_gold_passages.py

Chunks every verified ARAFA source article into sentence-level retrieval
units (overlapping 1-3 sentence windows -- see arabic_text_utils.py),
then locates, for every claim, the specific chunk that best matches its
evidence text.

Produces two outputs:

1. data/arafa/wikipedia_chunks.json
   Every verified source article, chunked into sentence windows. This is
   the retrieval corpus for these 3,318 known articles. (Indexing the
   REST of Arabic Wikipedia is a separate, later full-corpus step that
   reuses these same chunking functions from arabic_text_utils.py.)

2. data/arafa/gold_passages.json
   For every claim, the best-matching chunk_id -- the retrieval ground
   truth Phase 1 will score Recall@k / MRR against.

Usage:
    py scripts/rebuild_chunks_from_cache.py                   # once: full retrieval corpus
    py scripts/build_gold_passages.py --sample-size 200       # quick test (seed=42)
    py scripts/build_gold_passages.py                           # full gold-label run
    py scripts/build_gold_passages.py --resume                  # continue interrupted run
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests
from rapidfuzz.fuzz import ratio as fuzz_ratio
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arabic_text_utils import (
    build_sentence_windows,
    build_windows_of_size,
    normalize,
    segment_sentences,
)

API_URL = "https://ar.wikipedia.org/w/api.php"
USER_AGENT = (
    "ARAFA-RetrievalThesis/0.1 "
    "(Research project: Retrieval-Grounded Arabic Fact-Checking; "
    "contact: kma88@mail.aub.edu)"
)
REQUEST_DELAY_SECONDS = 0.3
FUZZY_THRESHOLD = 0.75
MAX_WINDOW = 3
WORD_OVERLAP_PREFILTER = 0.3  # skip clearly-irrelevant chunks before SequenceMatcher


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


def load_article_cache(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def fetch_full_text(session: requests.Session, page_id: int) -> str:
    params = {
        "action": "query", "pageids": page_id,
        "prop": "extracts", "explaintext": 1, "format": "json",
    }
    resp = session.get(API_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    page = data.get("query", {}).get("pages", {}).get(str(page_id), {})
    return page.get("extract", "")


def best_fuzzy_ratio(evidence: str, text: str) -> float:
    """Slide an evidence-length window across text (same approach as evidence_match_pilot)."""
    ev_len = len(evidence)
    if ev_len == 0 or len(text) < ev_len:
        return fuzz_ratio(evidence, text) / 100.0

    best = 0.0
    step = max(1, ev_len // 4)
    for start in range(0, len(text) - ev_len + 1, step):
        window = text[start : start + ev_len]
        score = fuzz_ratio(evidence, window) / 100.0
        if score > best:
            best = score
        if best > 0.97:
            break
    return best


def _prepped_from_raw(raw_chunk: dict) -> dict:
    norm = normalize(raw_chunk["text"])
    return {**raw_chunk, "_norm_text": norm, "_tokens": set(norm.split())}


def _raw_from_window(source_id: int, window: dict) -> dict:
    return {
        "chunk_id": f"{source_id}:{window['sentence_start']}-{window['sentence_end']}",
        "sentence_start": window["sentence_start"],
        "sentence_end": window["sentence_end"],
        "window_size": window["window_size"],
        "text": window["text"],
    }


def _search_prepped(prepped_chunks: list, evidence_norm: str, evidence_tokens: set,
                    size_range: range) -> dict:
    for size in size_range:
        for c in prepped_chunks:
            if c["window_size"] != size:
                continue
            if evidence_norm and evidence_norm in c["_norm_text"]:
                return {"chunk_id": c["chunk_id"], "match_type": "exact", "similarity": 1.0}

    best_ratio, best_chunk_id = 0.0, None
    for c in prepped_chunks:
        if c["window_size"] not in size_range:
            continue
        if not evidence_tokens or not c["_tokens"]:
            continue
        overlap = len(evidence_tokens & c["_tokens"]) / len(evidence_tokens)
        if overlap < WORD_OVERLAP_PREFILTER:
            continue
        ratio = best_fuzzy_ratio(evidence_norm, c["_norm_text"])
        if ratio > best_ratio:
            best_ratio, best_chunk_id = ratio, c["chunk_id"]

    if best_chunk_id and best_ratio >= FUZZY_THRESHOLD:
        return {"chunk_id": best_chunk_id, "match_type": "fuzzy", "similarity": round(best_ratio, 3)}
    return {"chunk_id": best_chunk_id, "match_type": "none", "similarity": round(best_ratio, 3)}


def find_best_chunk(
    evidence_norm: str,
    prepped_chunks: list,
    *,
    sentences: list | None = None,
    source_id: int | None = None,
    evidence_sentence_count: int = 1,
) -> dict:
    """
    Prefers the SMALLEST window size with an exact (substring) match --
    keeping gold evidence spans as tight as possible, echoing FEVER's own
    "minimal and complete" evidence annotation principle. Falls back to
    fuzzy matching (with a cheap word-overlap prefilter) if no exact
    match exists at any window size.

    For the rare evidence spans longer than MAX_WINDOW sentences (~1% of
    claims), builds on-demand larger windows so gold labels stay aligned
    with the retrieval corpus.
    """
    evidence_tokens = set(evidence_norm.split())
    result = _search_prepped(prepped_chunks, evidence_norm, evidence_tokens, range(1, MAX_WINDOW + 1))
    if result["match_type"] != "none" or not sentences or not source_id:
        return result
    if evidence_sentence_count <= MAX_WINDOW:
        return result

    extended_raw = []
    for size in range(MAX_WINDOW + 1, min(evidence_sentence_count, len(sentences)) + 1):
        for window in build_windows_of_size(sentences, size):
            extended_raw.append(_raw_from_window(source_id, window))

    extended_prepped = [_prepped_from_raw(c) for c in extended_raw]
    extended = _search_prepped(
        extended_prepped, evidence_norm, evidence_tokens,
        range(MAX_WINDOW + 1, min(evidence_sentence_count, len(sentences)) + 1),
    )
    if extended["match_type"] != "none":
        matched = next(c for c in extended_raw if c["chunk_id"] == extended["chunk_id"])
        extended["new_chunk"] = matched
    return extended


def merge_chunks(existing: dict, current: dict) -> dict:
    """Keep all sources seen so far; in-memory entries win on key collision."""
    merged = dict(existing)
    merged.update(current)
    return merged


def append_extended_chunks(chunks_path: Path, patches: dict[str, list]) -> None:
    """Append rare >3-sentence window chunks into an existing corpus file."""
    if not patches:
        return
    on_disk = json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.exists() else {}
    for source_key, new_chunks in patches.items():
        if source_key not in on_disk:
            continue
        existing_ids = {c["chunk_id"] for c in on_disk[source_key]["chunks"]}
        for chunk in new_chunks:
            if chunk["chunk_id"] not in existing_ids:
                on_disk[source_key]["chunks"].append(chunk)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(on_disk, f, ensure_ascii=False, separators=(",", ":"))


def save_gold_passages(gold_passages, gold_path):
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gold_path, "w", encoding="utf-8") as f:
        json.dump(gold_passages, f, ensure_ascii=False, separators=(",", ":"))


def save_outputs(chunks_by_source, gold_passages, chunks_path, gold_path):
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = chunks_by_source
    if chunks_path.exists():
        existing = json.loads(chunks_path.read_text(encoding="utf-8"))
        to_save = merge_chunks(existing, chunks_by_source)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, ensure_ascii=False, separators=(",", ":"))
    save_gold_passages(gold_passages, gold_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arafa", type=Path, default=Path("data/arafa/ARAFA.json"))
    parser.add_argument("--verification", type=Path, default=Path("data/arafa/verification/source_verification.json"))
    parser.add_argument("--chunks-output", type=Path, default=Path("data/arafa/wikipedia_chunks.json"))
    parser.add_argument("--gold-output", type=Path, default=Path("data/arafa/gold_passages.json"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed when --sample-size is set (matches pilot).")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--article-cache",
        type=Path,
        default=Path("data/arafa/evidence_match_full_article_cache.json"),
        help="Pre-fetched article texts from Phase 0 (falls back to live API on miss).",
    )
    args = parser.parse_args()

    with open(args.arafa, "r", encoding="utf-8") as f:
        claims = json.load(f)
    with open(args.verification, "r", encoding="utf-8") as f:
        verification = json.load(f)

    eligible = [c for c in claims if verification.get(str(c["source"]), {}).get("found")]
    if args.sample_size:
        rng = random.Random(args.seed)
        eligible = rng.sample(eligible, min(args.sample_size, len(eligible)))
    print(f"Processing {len(eligible)} claims across "
          f"{len(set(c['source'] for c in eligible))} unique sources.")

    gold_passages = {}
    sources_built: dict[str, dict] = {}
    extended_patches: dict[str, list] = {}
    if args.resume and args.gold_output.exists():
        gold_passages = json.loads(args.gold_output.read_text(encoding="utf-8"))
        print(f"Resuming: {len(gold_passages)} claims already resolved.")

    session = build_session()
    article_cache = load_article_cache(args.article_cache)
    if article_cache:
        print(f"Loaded {len(article_cache)} articles from cache ({args.article_cache}).")
    prepped_cache = {}
    sentences_cache = {}

    def get_article_text(source_id: int) -> str:
        cached = article_cache.get(source_id)
        if cached:
            return cached
        text = fetch_full_text(session, source_id)
        time.sleep(REQUEST_DELAY_SECONDS)
        return text

    def prep_chunks(source_id: int) -> tuple[list, list]:
        if source_id in prepped_cache:
            return prepped_cache[source_id], sentences_cache[source_id]

        source_key = str(source_id)
        sentences = segment_sentences(get_article_text(source_id))
        windows = build_sentence_windows(sentences, max_window=MAX_WINDOW)
        raw_chunks = [_raw_from_window(source_id, w) for w in windows]
        if args.sample_size:
            info = verification.get(source_key, {})
            sources_built[source_key] = {
                "title": info.get("title"),
                "url": info.get("url"),
                "num_sentences": len(sentences),
                "chunks": raw_chunks,
            }

        prepped = [_prepped_from_raw(c) for c in raw_chunks]
        prepped_cache[source_id] = prepped
        sentences_cache[source_id] = sentences
        return prepped, sentences

    total = len(eligible)
    for i, claim in enumerate(eligible, start=1):
        final_id = claim["final_id"]
        if args.resume and str(final_id) in gold_passages:
            continue

        source_id = claim["source"]
        print(f"[{i}/{total}] final_id={final_id} source={source_id} ...", end=" ")

        try:
            chunks, sentences = prep_chunks(source_id)
        except requests.RequestException as e:
            print(f"fetch error: {e}")
            gold_passages[str(final_id)] = {
                "final_id": final_id, "source": source_id,
                "chunk_id": None, "match_type": "error", "similarity": 0.0,
            }
            continue

        evidence_norm = normalize(claim["evidence"])
        result = find_best_chunk(
            evidence_norm,
            chunks,
            sentences=sentences,
            source_id=source_id,
            evidence_sentence_count=len(segment_sentences(claim["evidence"])),
        )
        new_chunk = result.pop("new_chunk", None)
        if new_chunk:
            source_key = str(source_id)
            if args.sample_size and source_key in sources_built:
                sources_built[source_key]["chunks"].append(new_chunk)
            else:
                extended_patches.setdefault(source_key, []).append(new_chunk)
            prepped_cache[source_id] = chunks + [_prepped_from_raw(new_chunk)]
        print(f"{result['match_type']} (sim={result['similarity']})")
        gold_passages[str(final_id)] = {"final_id": final_id, "source": source_id, **result}

        if i % 200 == 0:
            save_gold_passages(gold_passages, args.gold_output)

    save_gold_passages(gold_passages, args.gold_output)
    if args.sample_size:
        save_outputs(sources_built, gold_passages, args.chunks_output, args.gold_output)
    elif extended_patches:
        append_extended_chunks(args.chunks_output, extended_patches)
        print(f"Appended extended-window chunks for {len(extended_patches)} sources.")

    total_done = len(gold_passages)
    counts = {k: sum(1 for v in gold_passages.values() if v["match_type"] == k)
              for k in ("exact", "fuzzy", "none", "error")}
    print("\n=== Summary ===")
    for k, v in counts.items():
        print(f"{k.capitalize():7s}: {v} ({v/total_done:.1%})")
    if args.sample_size:
        print(f"\nChunk corpus written to {args.chunks_output}")
    print(f"Gold passages written to {args.gold_output}")


if __name__ == "__main__":
    main()