"""
evidence_match_pilot.py

For every claim whose source was verified as found, fetch the FULL
article text and check whether the claim's evidence string is actually
locatable inside it -- exact match after normalization, or fuzzy match
via a sliding-window similarity score.

This is the key pilot before committing to full-corpus Wikipedia
indexing: it tells us the real, current match rate and whether we can
rely on live Wikipedia or need a period-matched historical dump instead.

Output (data/arafa/evidence_match_pilot.json or evidence_match_full.json):
    {
      "summary": {
        "total_checked": ...,
        "exact_match": ...,
        "fuzzy_match": ...,
        "no_match": ...,
        "exact_rate": ...,
        "fuzzy_or_better_rate": ...
      },
      "results": [
        {
          "final_id": 2,
          "source": 10135,
          "match_type": "fuzzy",   # "exact" | "fuzzy" | "none"
          "similarity": 0.71,
          "evidence_preview": "...",
        },
        ...
      ]
    }

Usage:
    py scripts/evidence_match_pilot.py --sample-size 200
    py scripts/evidence_match_pilot.py --sample-size 200 --fuzzy-threshold 0.85
    py scripts/evidence_match_pilot.py --all --output data/arafa/evidence_match_full.json
    py scripts/evidence_match_pilot.py --all --resume   # continue interrupted full run
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import requests
from rapidfuzz.fuzz import ratio as fuzz_ratio
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

API_URL = "https://ar.wikipedia.org/w/api.php"
USER_AGENT = (
    "ARAFA-RetrievalThesis/0.1 "
    "(Research project: Retrieval-Grounded Arabic Fact-Checking; "
    "contact: kma88@mail.aub.edu)"
)
REQUEST_DELAY_SECONDS = 0.3
DEFAULT_CHECKPOINT_INTERVAL = 1000
# Full-article extracts cannot be batched: MediaWiki lowers exlimit to 1
# for whole-page extract requests, so we fetch one page at a time and cache.
CACHE_SAVE_INTERVAL = 25

# Arabic diacritics (tashkeel) -- strip these before comparing, since this
# was the single biggest source of "near but not exact" mismatch we found
# manually (ARAFA's evidence vs. today's live article wording).
TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")
WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    if not text:
        return ""
    text = TASHKEEL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def save_json(data: Any, path: Path) -> None:
    """Atomic-ish JSON write with Windows fallback."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    for attempt in range(5):
        try:
            tmp_path.replace(path)
            return
        except OSError:
            if attempt == 4:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                tmp_path.unlink(missing_ok=True)
                return
            time.sleep(0.5)


def fetch_full_text(session: requests.Session, page_id: int) -> str:
    params = {
        "action": "query",
        "pageids": page_id,
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
    }
    resp = session.get(API_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    page = pages.get(str(page_id), {})
    return page.get("extract", "")


def load_article_cache(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def save_article_cache(cache: Dict[int, str], path: Path) -> None:
    if not cache:
        return
    payload = {str(k): v for k, v in cache.items()}
    save_json(payload, path)


def warm_article_cache(
    session: requests.Session,
    source_ids: Set[int],
    cache: Dict[int, str],
    cache_path: Path,
) -> None:
    missing = sorted(
        sid for sid in source_ids if sid not in cache or not cache.get(sid)
    )
    if not missing:
        return

    since_save = 0
    for pid in tqdm(missing, desc="Fetching articles", unit="article"):
        try:
            text = fetch_full_text(session, pid)
            cache[pid] = text
        except requests.RequestException as e:
            cache[pid] = ""
            tqdm.write(f"fetch error for source={pid}: {e}")
        since_save += 1
        if since_save >= CACHE_SAVE_INTERVAL:
            save_article_cache(cache, cache_path)
            since_save = 0
        time.sleep(REQUEST_DELAY_SECONDS)

    save_article_cache(cache, cache_path)


def best_fuzzy_ratio(evidence: str, article: str) -> float:
    """
    Slide a window the size of the evidence string across the article and
    return the best SequenceMatcher ratio found. This handles evidence
    that has drifted/reworded slightly but is still recognizably the
    same passage.
    """
    ev_len = len(evidence)
    if ev_len == 0 or len(article) < ev_len:
        return fuzz_ratio(evidence, article) / 100.0

    best = 0.0
    step = max(1, ev_len // 4)  # coarse stride to keep this fast
    for start in range(0, len(article) - ev_len + 1, step):
        window = article[start : start + ev_len]
        score = fuzz_ratio(evidence, window) / 100.0
        if score > best:
            best = score
        if best > 0.97:  # good enough, stop early
            break
    return best


def build_summary(
    counts: Dict[str, int], total: int, fuzzy_threshold: float
) -> Dict[str, Any]:
    return {
        "total_checked": total,
        "exact_match": counts["exact"],
        "fuzzy_match": counts["fuzzy"],
        "no_match": counts["none"],
        "exact_rate": round(counts["exact"] / total, 4) if total else 0,
        "fuzzy_or_better_rate": round((counts["exact"] + counts["fuzzy"]) / total, 4)
        if total
        else 0,
        "fuzzy_threshold_used": fuzzy_threshold,
    }


def load_existing_output(path: Path) -> tuple[List[Dict[str, Any]], Set[int], Dict[str, int]]:
    if not path.exists():
        return [], set(), {"exact": 0, "fuzzy": 0, "none": 0}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    processed = {r["final_id"] for r in results}
    counts = {"exact": 0, "fuzzy": 0, "none": 0}
    for r in results:
        counts[r["match_type"]] += 1
    return results, processed, counts


def match_claim(
    claim: Dict[str, Any],
    article_cache: Dict[int, str],
    normalized_article_cache: Dict[int, str],
    fuzzy_threshold: float,
) -> Dict[str, Any]:
    source_id = claim["source"]
    if source_id not in normalized_article_cache:
        normalized_article_cache[source_id] = normalize(
            article_cache.get(source_id, "")
        )
    article_norm = normalized_article_cache[source_id]
    evidence_norm = normalize(claim["evidence"])

    if evidence_norm and evidence_norm in article_norm:
        match_type = "exact"
        similarity = 1.0
    else:
        similarity = best_fuzzy_ratio(evidence_norm, article_norm)
        match_type = "fuzzy" if similarity >= fuzzy_threshold else "none"

    return {
        "final_id": claim["final_id"],
        "source": source_id,
        "match_type": match_type,
        "similarity": round(similarity, 3),
        "evidence_preview": claim["evidence"][:80],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arafa", type=Path, default=Path("data/arafa/ARAFA.json"))
    parser.add_argument(
        "--verification",
        type=Path,
        default=Path("data/arafa/verification/source_verification.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/arafa/evidence_match_pilot.json")
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=None,
        help="Persisted article text cache (default: <output-stem>_article_cache.json).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of claims to sample for the pilot. Use 0 with --all for full run.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process ALL eligible claims (ignores --sample-size).",
    )
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Save progress every N claims (full run).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing --output file if present.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh even if output/cache already exist.",
    )
    parser.add_argument(
        "--match-only",
        action="store_true",
        help="Skip Wikipedia fetching; use existing --cache-file only.",
    )
    args = parser.parse_args()

    cache_path = args.cache_file or args.output.with_name(
        f"{args.output.stem}_article_cache.json"
    )

    with open(args.arafa, "r", encoding="utf-8") as f:
        claims = json.load(f)
    with open(args.verification, "r", encoding="utf-8") as f:
        verification = json.load(f)

    eligible = [
        c for c in claims if verification.get(str(c["source"]), {}).get("found")
    ]
    print(f"{len(eligible)} claims have a verified source out of {len(claims)} total.")

    run_all = args.all or args.sample_size == 0
    if run_all:
        work = sorted(eligible, key=lambda c: c["final_id"])
        print(f"Full run: processing all {len(work)} eligible claims.")
    else:
        import random

        random.seed(args.seed)
        work = random.sample(eligible, min(args.sample_size, len(eligible)))
        print(f"Sampling {len(work)} claims for the pilot.")

    resume = args.resume or (run_all and not args.no_resume)
    results: List[Dict[str, Any]] = []
    processed_ids: Set[int] = set()
    counts = {"exact": 0, "fuzzy": 0, "none": 0}

    if resume:
        results, processed_ids, counts = load_existing_output(args.output)
        if processed_ids:
            print(
                f"Resuming: {len(processed_ids)} claims already done, "
                f"{len(work) - len(processed_ids)} remaining."
            )

    remaining = [c for c in work if c["final_id"] not in processed_ids]
    if not remaining:
        print("Nothing left to process.")
        summary = build_summary(counts, len(results), args.fuzzy_threshold)
        save_json({"summary": summary, "results": results}, args.output)
        print("\n=== Summary ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        return

    session = build_session()
    article_cache = load_article_cache(cache_path)

    needed_sources = {c["source"] for c in remaining}
    nonempty_cached = sum(1 for sid in needed_sources if article_cache.get(sid))
    print(
        f"Article cache: {len(article_cache)} entries, "
        f"{nonempty_cached}/{len(needed_sources)} needed sources have text."
    )
    if args.match_only:
        missing_text = sorted(sid for sid in needed_sources if not article_cache.get(sid))
        if missing_text:
            print(
                f"Warning: {len(missing_text)} sources still have no cached text; "
                "those claims will likely be no_match."
            )
    else:
        warm_article_cache(session, needed_sources, article_cache, cache_path)

    checkpoint_interval = args.checkpoint_interval if run_all else len(remaining)
    normalized_article_cache: Dict[int, str] = {}

    with tqdm(total=len(remaining), desc="Matching evidence", unit="claim") as pbar:
        since_checkpoint = 0
        for claim in remaining:
            result = match_claim(
                claim, article_cache, normalized_article_cache, args.fuzzy_threshold
            )
            counts[result["match_type"]] += 1
            results.append(result)
            processed_ids.add(claim["final_id"])
            since_checkpoint += 1
            pbar.update(1)
            pbar.set_postfix(
                exact=counts["exact"],
                fuzzy=counts["fuzzy"],
                none=counts["none"],
            )

            if since_checkpoint >= checkpoint_interval:
                summary = build_summary(counts, len(results), args.fuzzy_threshold)
                save_json({"summary": summary, "results": results}, args.output)
                since_checkpoint = 0

    summary = build_summary(counts, len(results), args.fuzzy_threshold)
    save_json({"summary": summary, "results": results}, args.output)

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nReport written to {args.output}")
    print(f"Article cache written to {cache_path}")


if __name__ == "__main__":
    main()
