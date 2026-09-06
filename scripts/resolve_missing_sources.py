"""
resolve_missing_sources.py

For source (page) IDs that the live Wikipedia API reports as "missing"
(deleted, merged, renamed, or otherwise gone), attempt to recover the
current article using several strategies:

1. Wikipedia API with redirects=1 (rename / merge redirect target)
2. Log events (delete / move / merge) on titles recovered from Wayback
3. Title search on Wikipedia after recovering a historical title from Wayback
4. Wayback Machine archived permalink (ar.wikipedia.org/?curid=<id>)

Output (data/arafa/verification/resolved_missing_sources.json):
    {
      "122891": {
        "source": 122891,
        "resolved": true,
        "resolution_method": "redirect",
        "title": "...",
        "url": "https://ar.wikipedia.org/wiki/...",
        "resolved_pageid": 10485859,
        "wayback_found": false,
        "wayback_url": null,
        "snapshot_timestamp": null,
        "notes": null
      },
      ...
    }

Usage:
    py scripts/resolve_missing_sources.py
    py scripts/resolve_missing_sources.py --ids 122891 801976 ...
    py scripts/resolve_missing_sources.py --ids-file data/arafa/verification/missing_sources.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WIKI_API_URL = "https://ar.wikipedia.org/w/api.php"
CDX_API = "https://web.archive.org/cdx/search/cdx"
USER_AGENT = (
    "ARAFA-Retrieval-Research/1.0 "
    "(Retrieval-Grounded-Arabic-Fact-Checking thesis project; "
    "https://github.com/karamahmadkassem/Retrieval-Grounded-Arabic-Fact-Checking)"
)
WIKI_DELAY_SECONDS = 0.5
WAYBACK_DELAY_SECONDS = 1.5

URL_VARIANTS = [
    "ar.wikipedia.org/wiki/index.php?curid={id}",
    "ar.wikipedia.org/w/index.php?curid={id}",
    "ar.wikipedia.org/?curid={id}",
]

TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
DEFAULT_IDS_FILE = Path("data/arafa/verification/missing_sources.json")
DEFAULT_OUTPUT = Path("data/arafa/verification/resolved_missing_sources.json")


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(
        total=5,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def save_json(data: Any, path: Path) -> None:
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


def wiki_query(session: requests.Session, params: Dict[str, Any]) -> Dict[str, Any]:
    query_params = {"format": "json", **params}
    resp = session.get(WIKI_API_URL, params=query_params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def try_wikipedia_redirect(session: requests.Session, source_id: int) -> Optional[Dict[str, Any]]:
    """Query pageid with redirects=1; return target page if resolved."""
    data = wiki_query(
        session,
        {
            "action": "query",
            "pageids": str(source_id),
            "redirects": 1,
            "prop": "info",
            "inprop": "url",
        },
    )
    query = data.get("query", {})
    pages = query.get("pages", {})
    requested = pages.get(str(source_id))

    if requested and "missing" not in requested:
        return {
            "resolution_method": "live_page",
            "title": requested.get("title"),
            "url": requested.get("fullurl"),
            "resolved_pageid": requested.get("pageid"),
            "notes": "Page exists at original ID.",
        }

    redirects = query.get("redirects", [])
    for page in pages.values():
        pageid = page.get("pageid")
        if pageid and pageid != source_id and "missing" not in page:
            method = "redirect" if redirects else "alternate_pageid"
            notes = None
            if redirects:
                notes = "; ".join(f"{r.get('from')} -> {r.get('to')}" for r in redirects)
            return {
                "resolution_method": method,
                "title": page.get("title"),
                "url": page.get("fullurl"),
                "resolved_pageid": pageid,
                "notes": notes,
            }

    return None


def find_wayback_snapshot(session: requests.Session, source_id: int) -> Dict[str, Any]:
    for variant in URL_VARIANTS:
        target_url = variant.format(id=source_id)
        params = {
            "url": target_url,
            "output": "json",
            "limit": 1,
            "filter": "statuscode:200",
        }
        try:
            resp = session.get(CDX_API, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError):
            time.sleep(WAYBACK_DELAY_SECONDS)
            continue

        if len(data) > 1:
            row = data[1]
            timestamp = row[1]
            original = row[2]
            archived_url = f"https://web.archive.org/web/{timestamp}/{original}"
            return {
                "found": True,
                "archived_url": archived_url,
                "snapshot_timestamp": timestamp,
                "matched_variant": target_url,
            }
        time.sleep(WAYBACK_DELAY_SECONDS)

    return {"found": False}


def fetch_title_from_archive(session: requests.Session, archived_url: str) -> Optional[str]:
    try:
        resp = session.get(archived_url, timeout=60)
        resp.raise_for_status()
        match = TITLE_RE.search(resp.text)
        if match:
            title = match.group(1)
            return title.split(" - ويكيبيديا")[0].strip()
    except requests.RequestException:
        pass
    return None


def resolve_title_on_wikipedia(
    session: requests.Session, title: str
) -> Optional[Dict[str, Any]]:
    """Resolve a historical title to the current live article."""
    data = wiki_query(
        session,
        {
            "action": "query",
            "titles": title,
            "redirects": 1,
            "prop": "info",
            "inprop": "url",
        },
    )
    pages = data.get("query", {}).get("pages", {})
    redirects = data.get("query", {}).get("redirects", [])

    for page in pages.values():
        if "missing" not in page:
            method = "redirect" if redirects else "title_lookup"
            notes = None
            if redirects:
                notes = "; ".join(f"{r.get('from')} -> {r.get('to')}" for r in redirects)
            return {
                "resolution_method": method,
                "title": page.get("title"),
                "url": page.get("fullurl"),
                "resolved_pageid": page.get("pageid"),
                "notes": notes,
            }
    return None


def check_log_events(session: requests.Session, title: str) -> Optional[Dict[str, Any]]:
    """Follow move/merge/delete log events to find a surviving title."""
    data = wiki_query(
        session,
        {
            "action": "query",
            "list": "logevents",
            "letype": "move|merge|delete",
            "letitle": title,
            "lelimit": 10,
        },
    )
    events = data.get("query", {}).get("logevents", [])

    candidate_titles: List[str] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "move":
            target = event.get("params", {}).get("4::target") or event.get("params", {}).get(
                "target_title"
            )
            if target:
                candidate_titles.append(target)
        elif event_type == "merge":
            dest = event.get("params", {}).get("4::dest") or event.get("params", {}).get(
                "dest_title"
            )
            if dest:
                candidate_titles.append(dest)

    for candidate in candidate_titles:
        resolved = resolve_title_on_wikipedia(session, candidate)
        if resolved:
            resolved["resolution_method"] = "merge" if any(
                e.get("type") == "merge" for e in events
            ) else "move"
            resolved["notes"] = f"Recovered via log events from historical title {title!r}."
            return resolved

    delete_only = events and all(e.get("type") == "delete" for e in events)
    if delete_only:
        return {
            "resolution_method": "deleted",
            "title": title,
            "url": None,
            "resolved_pageid": None,
            "notes": "Log shows page was deleted; no merge/move target found.",
        }

    return None


def resolve_source(session: requests.Session, source_id: int) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "source": source_id,
        "resolved": False,
        "resolution_method": None,
        "title": None,
        "url": None,
        "resolved_pageid": None,
        "wayback_found": False,
        "wayback_url": None,
        "snapshot_timestamp": None,
        "notes": None,
    }

    wiki_result = try_wikipedia_redirect(session, source_id)
    time.sleep(WIKI_DELAY_SECONDS)
    if wiki_result:
        base.update(wiki_result)
        base["resolved"] = True
        return base

    snap = find_wayback_snapshot(session, source_id)
    if snap["found"]:
        base["wayback_found"] = True
        base["wayback_url"] = snap["archived_url"]
        base["snapshot_timestamp"] = snap["snapshot_timestamp"]

        historical_title = fetch_title_from_archive(session, snap["archived_url"])
        time.sleep(WAYBACK_DELAY_SECONDS)

        if historical_title:
            base["title"] = historical_title

            title_result = resolve_title_on_wikipedia(session, historical_title)
            time.sleep(WIKI_DELAY_SECONDS)
            if title_result:
                base.update(title_result)
                base["resolved"] = True
                if base["resolution_method"] == "title_lookup":
                    base["resolution_method"] = "wayback_title_lookup"
                return base

            log_result = check_log_events(session, historical_title)
            time.sleep(WIKI_DELAY_SECONDS)
            if log_result and log_result.get("resolution_method") != "deleted":
                base.update(log_result)
                base["resolved"] = True
                return base
            if log_result and log_result.get("resolution_method") == "deleted":
                base.update(log_result)
                base["resolved"] = False
                return base

            base["resolution_method"] = "wayback_only"
            base["notes"] = (
                "Wayback snapshot found but no live Wikipedia article matches the archived title."
            )
            return base

        base["resolution_method"] = "wayback_only"
        base["notes"] = "Wayback snapshot found but title could not be extracted."
        return base

    base["resolution_method"] = "unresolved"
    base["notes"] = "No live redirect and no Wayback snapshot found."
    return base


def safe_print(message: str) -> None:
    """Print UTF-8 text without crashing on Windows cp1252 consoles."""
    try:
        print(message)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((message + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids", type=int, nargs="+", help="Missing source IDs, space-separated."
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=DEFAULT_IDS_FILE,
        help=f"JSON file containing a list of missing IDs (default: {DEFAULT_IDS_FILE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args()

    if args.ids:
        ids = args.ids
    else:
        if not args.ids_file.exists():
            parser.error(f"IDs file not found: {args.ids_file}")
        with open(args.ids_file, "r", encoding="utf-8") as f:
            ids = json.load(f)

    session = build_session()
    results: Dict[str, Any] = {}

    for i, source_id in enumerate(ids, start=1):
        safe_print(f"[{i}/{len(ids)}] Resolving source {source_id} ...")
        record = resolve_source(session, source_id)
        results[str(source_id)] = record

        if record["resolved"]:
            safe_print(
                f"  RESOLVED via {record['resolution_method']}: "
                f"{record['title']} ({record['url']})"
            )
        elif record.get("resolution_method") == "deleted":
            safe_print(f"  DELETED: {record.get('title')}")
        elif record.get("wayback_found"):
            safe_print(
                f"  PARTIAL (Wayback only): {record.get('title') or 'unknown title'} "
                f"-> {record['wayback_url']}"
            )
        else:
            safe_print(f"  UNRESOLVED: {record.get('notes')}")

        save_json(results, args.output)

    resolved = sum(1 for v in results.values() if v["resolved"])
    wayback_only = sum(
        1 for v in results.values() if v.get("resolution_method") == "wayback_only"
    )
    deleted = sum(1 for v in results.values() if v.get("resolution_method") == "deleted")
    unresolved = len(ids) - resolved - wayback_only - deleted

    print("\n=== Summary ===")
    print(f"Resolved to live Wikipedia: {resolved}/{len(ids)}")
    print(f"Wayback only (no live page): {wayback_only}/{len(ids)}")
    print(f"Confirmed deleted:          {deleted}/{len(ids)}")
    print(f"Still unresolved:           {unresolved}/{len(ids)}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
