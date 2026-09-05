"""Verify ARAFA.json and its formatted copy are semantically equivalent."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import ijson


def canonical_bytes(obj: Any) -> bytes:
    """Minified UTF-8 JSON bytes preserving key order."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_json(path: Path) -> tuple[bool, str | None]:
    try:
        with path.open("rb") as f:
            parser = ijson.parse(f)
            for _ in parser:
                pass
        return True, None
    except Exception as exc:
        return False, str(exc)


def stream_compare(
    original_path: Path,
    formatted_path: Path,
    max_diffs: int = 5,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "original_valid": False,
        "formatted_valid": False,
        "original_count": 0,
        "formatted_count": 0,
        "equal": False,
        "original_hash": "",
        "formatted_hash": "",
        "differences": [],
    }

    original_valid, original_err = validate_json(original_path)
    formatted_valid, formatted_err = validate_json(formatted_path)
    result["original_valid"] = original_valid
    result["formatted_valid"] = formatted_valid
    if not original_valid:
        result["differences"].append(f"Original JSON invalid: {original_err}")
        return result
    if not formatted_valid:
        result["differences"].append(f"Formatted JSON invalid: {formatted_err}")
        return result

    original_hasher = hashlib.sha256()
    formatted_hasher = hashlib.sha256()

    with original_path.open("rb") as original_file, formatted_path.open("rb") as formatted_file:
        original_items = ijson.items(original_file, "item")
        formatted_items = ijson.items(formatted_file, "item")

        index = 0
        while True:
            try:
                original_record = next(original_items)
            except StopIteration:
                original_record = None

            try:
                formatted_record = next(formatted_items)
            except StopIteration:
                formatted_record = None

            if original_record is None and formatted_record is None:
                break

            if original_record is None:
                result["differences"].append(
                    f"Formatted has extra record at index {index} (original ended earlier)"
                )
                result["formatted_count"] = index + 1
                break

            if formatted_record is None:
                result["differences"].append(
                    f"Original has extra record at index {index} (formatted ended earlier)"
                )
                result["original_count"] = index + 1
                break

            original_bytes = canonical_bytes(original_record)
            formatted_bytes = canonical_bytes(formatted_record)
            original_hasher.update(original_bytes)
            original_hasher.update(b"\n")
            formatted_hasher.update(formatted_bytes)
            formatted_hasher.update(b"\n")

            if original_record != formatted_record and len(result["differences"]) < max_diffs:
                if original_record.keys() != formatted_record.keys():
                    result["differences"].append(
                        f"Record {index}: field set mismatch "
                        f"{set(original_record.keys()) ^ set(formatted_record.keys())}"
                    )
                else:
                    for key in original_record:
                        if original_record[key] != formatted_record[key]:
                            result["differences"].append(
                                f"Record {index}: field '{key}' differs"
                            )
                            break
            elif original_record != formatted_record and len(result["differences"]) == max_diffs:
                result["differences"].append("... additional differences omitted")

            index += 1
            if index % 25000 == 0:
                print(f"  compared {index:,} records...")

        result["original_count"] = index
        result["formatted_count"] = index

    if result["original_count"] != result["formatted_count"]:
        result["differences"].append(
            f"Record count mismatch: {result['original_count']:,} vs {result['formatted_count']:,}"
        )

    result["original_hash"] = original_hasher.hexdigest()
    result["formatted_hash"] = formatted_hasher.hexdigest()
    result["equal"] = (
        result["original_valid"]
        and result["formatted_valid"]
        and result["original_count"] == result["formatted_count"]
        and not result["differences"]
        and result["original_hash"] == result["formatted_hash"]
    )
    return result


def print_report(result: dict[str, Any]) -> None:
    print("Validation report")
    print("-------------------")
    print(f"Original valid JSON:  {result['original_valid']}")
    print(f"Formatted valid JSON: {result['formatted_valid']}")
    print(f"Original records:     {result['original_count']:,}")
    print(f"Formatted records:    {result['formatted_count']:,}")
    print(f"Content hash (orig):  {result['original_hash']}")
    print(f"Content hash (fmt):   {result['formatted_hash']}")
    print(f"Semantically equal:   {result['equal']}")
    if result["differences"]:
        print("\nDifferences:")
        for diff in result["differences"]:
            print(f"  - {diff}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("data/arafa/ARAFA.json"),
        help="Minified source JSON array",
    )
    parser.add_argument(
        "--formatted",
        type=Path,
        default=Path("data/arafa/ARAFA.formatted.json"),
        help="Pretty-printed JSON array",
    )
    parser.add_argument(
        "--max-diffs",
        type=int,
        default=5,
        help="Maximum per-record differences to report",
    )
    args = parser.parse_args()

    for path in (args.original, args.formatted):
        if not path.is_file():
            raise SystemExit(f"File not found: {path}")

    print(f"Comparing {args.original} and {args.formatted}")
    result = stream_compare(args.original, args.formatted, max_diffs=args.max_diffs)
    print_report(result)
    sys.exit(0 if result["equal"] else 1)


if __name__ == "__main__":
    main()
