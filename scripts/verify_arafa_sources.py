"""
verify_arafa_sources.py

Checks whether each unique `source` field in the ARAFA dataset (an Arabic
Wikipedia page ID) actually resolves to a real, existing Wikipedia article,
using the MediaWiki API. Produces two JSON files:

1. <output-dir>/source_verification.json
   One record per UNIQUE source ID (deduplicated — there are ~3,318 unique
   sources across ~182K claims, so this avoids re-querying the same page
   dozens of times):
     {
       "10135": {
         "source": 10135,
         "found": true,
         "title": "هارون الرشيد",
         "url": "https://ar.wikipedia.org/wiki/...",
         "extract_chars": 8421,
         "error": null
       },
       ...
     }

2. <output-dir>/claims_verification.json
   One record per CLAIM (final_id), joined back to its source's result —
   this is the file you'll actually use downstream, since it's indexed the
   same way as ARAFA.json itself:
     [
       {"final_id": 1, "source": 10135, "found": true,
        "title": "هارون الرشيد", "url": "https://ar.wikipedia.org/wiki/..."},
       ...
     ]

3. <output-dir>/missing_sources.json
   Just the list of source IDs that did NOT resolve, for quick manual review.

Usage:
    pip install requests tqdm
    python verify_arafa_sources.py \
        --input data/arafa/ARAFA.json \
        --output-dir data/arafa/verification

Resumable: if source_verification.json already exists, sources already
checked are skipped on the next run — safe to stop (Ctrl+C) and restart.

Notes on the API:
- We query the MediaWiki API in batches of pageids (default 20 per request,
  the safe limit for the `extracts` prop on non-bot accounts; page existence
  info itself has a higher limit, but 20 keeps extract_chars accurate for
  every item in the batch).
- `redirects=1` is passed so that if a source ID redirects to another page
  (e.g. a merged/renamed article), the API follows it and returns the final
  page — the "found" result reflects the *resolved* page.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from tqdm import tqdm

WIKI_API_URL = "https://ar.wikipedia.org/w/api.php"
USER_AGENT = (
    "ARAFA-Retrieval-Research/1.0 "
    "(Retrieval-Grounded-Arabic-Fact-Checking thesis project; "
    "https://github.com/karamahmadkassem/Retrieval-Grounded-Arabic-Fact-Checking)"
)


def load_claims(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_json(data: Any, path: Path) -> None:
    """Write JSON to disk. Uses a temp file + rename when possible; falls back
    to an in-place write if the rename fails (common on Windows when the
    target is locked). Progress is still safe to resume via batch checkpoints."""
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
            time.sleep(0.5 * (attempt + 1))


def chunked(seq: List[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def query_batch(session: requests.Session, pageids: List[int]) -> Dict[int, Dict[str, Any]]:
    """Query the MediaWiki API for a batch of pageids.
    Returns {pageid: {source, found, title, url, extract_chars, error}}.
    """
    params = {
        "action": "query",
        "pageids": "|".join(str(pid) for pid in pageids),
        "prop": "info|extracts",
        "inprop": "url",
        "explaintext": 1,
        "exlimit": "max",
        "redirects": 1,
        "format": "json",
    }

    results: Dict[int, Dict[str, Any]] = {}

    try:
        resp = session.get(WIKI_API_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # network error, timeout, bad JSON, etc.
        for pid in pageids:
            results[pid] = {
                "source": pid,
                "found": False,
                "title": None,
                "url": None,
                "extract_chars": 0,
                "error": f"request_failed: {e}",
            }
        return results

    query = data.get("query", {})
    pages = query.get("pages", {})
    pageid_set = set(pageids)

    accounted_for = set()
    for page_id_str, page in pages.items():
        try:
            resolved_id = int(page_id_str)
        except ValueError:
            continue

        # IMPORTANT MediaWiki quirk: when a requested pageid does NOT exist,
        # the API does not echo back that original id at all — instead it
        # invents a synthetic negative id ("-1", "-2", ...) for the "missing"
        # placeholder, with no way to tell which of several bad ids in the
        # same batch it corresponds to. We deliberately ignore those
        # synthetic entries here (resolved_id not in our real pageid_set)
        # and instead let the catch-all loop below mark the real requested
        # ids that never got a proper response as "not found" — this keeps
        # source_verification.json free of bogus "-1"/"-2" keys.
        if resolved_id not in pageid_set:
            continue

        is_missing = "missing" in page
        extract = page.get("extract", "") or ""

        results[resolved_id] = {
            "source": resolved_id,
            "found": not is_missing,
            "title": page.get("title"),
            "url": page.get("fullurl"),
            "extract_chars": len(extract),
            "error": None,
        }
        accounted_for.add(resolved_id)

    # Any requested pageid that never appeared under its own id above is
    # either missing (per the quirk above) or genuinely not returned —
    # either way it did not resolve to a real article.
    for pid in pageids:
        if pid not in accounted_for:
            results[pid] = {
                "source": pid,
                "found": False,
                "title": None,
                "url": None,
                "extract_chars": 0,
                "error": "missing_or_invalid_pageid",
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ARAFA source Wikipedia page IDs.")
    parser.add_argument("--input", default="data/arafa/ARAFA.json", help="Path to ARAFA.json")
    parser.add_argument(
        "--output-dir", default="data/arafa/verification", help="Directory to write output JSON files"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Pageids per API request (keep <=20 for full extract data)"
    )
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds to sleep between batches")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only check the first N unique sources (useful for a quick test run)"
    )
    parser.add_argument("--retries", type=int, default=3, help="Retries per failed batch")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    source_out_path = output_dir / "source_verification.json"
    claims_out_path = output_dir / "claims_verification.json"
    missing_out_path = output_dir / "missing_sources.json"

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading claims from {input_path} ...")
    claims = load_claims(input_path)
    print(f"Loaded {len(claims)} claim records.")

    # Collect unique sources, preserving first-seen order.
    seen = set()
    unique_sources: List[int] = []
    for c in claims:
        src = c.get("source")
        if src is not None and src not in seen:
            seen.add(src)
            unique_sources.append(src)
    print(f"Found {len(unique_sources)} unique source IDs across all claims.")

    if args.limit:
        unique_sources = unique_sources[: args.limit]
        print(f"--limit set: only checking the first {len(unique_sources)} unique sources.")

    # Resume support: skip sources already checked in a previous run.
    existing_results: Dict[str, Any] = load_existing(source_out_path)
    already_checked = {int(k) for k in existing_results.keys()}
    to_check = [s for s in unique_sources if s not in already_checked]
    print(f"{len(already_checked)} already checked previously, {len(to_check)} remaining.")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_results: Dict[str, Any] = dict(existing_results)

    batches = list(chunked(to_check, args.batch_size))
    for batch in tqdm(batches, desc="Verifying sources", unit="batch"):
        batch_results = None
        for attempt in range(args.retries):
            batch_results = query_batch(session, batch)
            all_failed = all(
                v.get("error") and "request_failed" in v["error"] for v in batch_results.values()
            )
            if not all_failed:
                break
            time.sleep(2 * (attempt + 1))  # back off before retrying

        for pid, record in batch_results.items():
            all_results[str(pid)] = record

        # Save after every batch so progress is never lost if interrupted.
        save_json(all_results, source_out_path)
        time.sleep(args.sleep)

    # --- Summary stats over unique sources ---
    total = len(all_results)
    found_count = sum(1 for v in all_results.values() if v.get("found"))
    pct = (found_count / total * 100) if total else 0.0
    print(f"\nUnique sources: {found_count}/{total} resolved to a real article ({pct:.1f}%).")

    missing_ids = sorted(int(k) for k, v in all_results.items() if not v.get("found"))
    save_json(missing_ids, missing_out_path)
    print(f"Missing source IDs written to {missing_out_path} ({len(missing_ids)} ids).")

    # --- Build per-claim output, joined to each claim's source result ---
    print("Building per-claim verification file ...")
    claims_output = []
    for c in claims:
        src = c.get("source")
        src_record = all_results.get(
            str(src),
            {"found": False, "title": None, "url": None, "error": "source_not_checked"},
        )
        claims_output.append(
            {
                "final_id": c.get("final_id"),
                "source": src,
                "found": src_record.get("found"),
                "title": src_record.get("title"),
                "url": src_record.get("url"),
            }
        )

    save_json(claims_output, claims_out_path)

    found_claims = sum(1 for c in claims_output if c["found"])
    claim_pct = (found_claims / len(claims_output) * 100) if claims_output else 0.0
    print(f"Per-claim file written to {claims_out_path}")
    print(f"Claims with a resolved source: {found_claims}/{len(claims_output)} ({claim_pct:.1f}%).")
    print(f"Per-source file written to {source_out_path}")


if __name__ == "__main__":
    main()
