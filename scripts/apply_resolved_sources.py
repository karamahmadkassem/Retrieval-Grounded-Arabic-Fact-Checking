"""
apply_resolved_sources.py

Patch ARAFA verification files using resolved_missing_sources.json:
- missing_sources.json  -> only truly unresolved IDs
- source_verification.json -> mark resolved sources as found
- claims_verification.json -> propagate resolved source metadata to claims
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

VERIFICATION_DIR = Path("data/arafa/verification")
RESOLVED_PATH = VERIFICATION_DIR / "resolved_missing_sources.json"
MISSING_PATH = VERIFICATION_DIR / "missing_sources.json"
SOURCE_PATH = VERIFICATION_DIR / "source_verification.json"
CLAIMS_PATH = VERIFICATION_DIR / "claims_verification.json"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def count_missing_sources(source_verification: Dict[str, Any]) -> int:
    return sum(1 for v in source_verification.values() if not v.get("found"))


def count_unresolved_claims(claims: List[Dict[str, Any]]) -> int:
    return sum(1 for c in claims if not c.get("found"))


def build_source_patch(resolution: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "found": True,
        "title": resolution["title"],
        "url": resolution["url"],
        "extract_chars": 0,
        "error": None,
        "resolution_method": resolution.get("resolution_method"),
        "resolved_pageid": resolution.get("resolved_pageid"),
        "resolution_notes": resolution.get("notes"),
    }


def main() -> None:
    resolved_all: Dict[str, Dict[str, Any]] = load_json(RESOLVED_PATH)
    source_verification: Dict[str, Any] = load_json(SOURCE_PATH)
    claims: List[Dict[str, Any]] = load_json(CLAIMS_PATH)
    old_missing: List[int] = load_json(MISSING_PATH)

    before = {
        "missing_sources": len(old_missing),
        "sources_not_found": count_missing_sources(source_verification),
        "claims_not_found": count_unresolved_claims(claims),
    }

    resolved_ids: Set[int] = {
        int(k) for k, v in resolved_all.items() if v.get("resolved")
    }
    unresolved_ids: Set[int] = {
        int(k) for k, v in resolved_all.items() if not v.get("resolved")
    }

    expected_resolved = {int(k) for k in old_missing} - unresolved_ids
    if resolved_ids != expected_resolved:
        print(
            f"WARNING: resolved ID set mismatch. "
            f"resolved={sorted(resolved_ids)}, expected={sorted(expected_resolved)}",
            file=sys.stderr,
        )

    # Patch source_verification for resolved IDs
    sources_updated = 0
    for sid in sorted(resolved_ids):
        key = str(sid)
        if key not in source_verification:
            print(f"WARNING: source {sid} not in source_verification.json", file=sys.stderr)
            continue
        entry = source_verification[key]
        patch = build_source_patch(resolved_all[key])
        entry.update(patch)
        sources_updated += 1

    # Update missing_sources.json
    new_missing = sorted(unresolved_ids)
    save_json(new_missing, MISSING_PATH)
    save_json(source_verification, SOURCE_PATH)

    # Patch claims for resolved source IDs
    resolved_lookup = {sid: resolved_all[str(sid)] for sid in resolved_ids}
    claims_updated = 0
    for claim in claims:
        src = claim.get("source")
        if src not in resolved_lookup:
            continue
        resolution = resolved_lookup[src]
        claim["found"] = True
        claim["title"] = resolution["title"]
        claim["url"] = resolution["url"]
        claims_updated += 1

    save_json(claims, CLAIMS_PATH)

    after = {
        "missing_sources": len(new_missing),
        "sources_not_found": count_missing_sources(source_verification),
        "claims_not_found": count_unresolved_claims(claims),
    }

    print("=== apply_resolved_sources summary ===")
    print(f"Resolved source IDs patched: {sources_updated}")
    print(f"Claims updated: {claims_updated}")
    print()
    print("Before:")
    for k, v in before.items():
        print(f"  {k}: {v}")
    print()
    print("After:")
    for k, v in after.items():
        print(f"  {k}: {v}")
    print()
    print(f"missing_sources.json now: {new_missing}")
    print(f"Resolved IDs: {sorted(resolved_ids)}")


if __name__ == "__main__":
    main()
