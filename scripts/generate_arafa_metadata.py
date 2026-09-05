"""Generate comprehensive metadata for the ARAFA dataset using streaming JSON parsing."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ijson

SCHEMA_FIELDS = [
    "final_id",
    "source",
    "claim",
    "evidence",
    "judgement",
    "reasoning",
    "Entity_in_Claim",
    "Co_referenced_in_Evidence",
    "Co_referenced_in_Text",
    "type",
]

TEXT_FIELDS = ["claim", "evidence", "reasoning", "Entity_in_Claim", "Co_referenced_in_Evidence", "Co_referenced_in_Text"]


def is_null_or_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def update_length_stats(stats: dict[str, Any], length: int) -> None:
    stats["count"] += 1
    stats["total_chars"] += length
    stats["min_chars"] = min(stats["min_chars"], length)
    stats["max_chars"] = max(stats["max_chars"], length)


def finalize_length_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = stats["count"]
    if count == 0:
        return {"count": 0, "min_chars": None, "max_chars": None, "mean_chars": None, "total_chars": 0}
    return {
        "count": count,
        "min_chars": stats["min_chars"],
        "max_chars": stats["max_chars"],
        "mean_chars": round(stats["total_chars"] / count, 2),
        "total_chars": stats["total_chars"],
    }


def new_length_stats() -> dict[str, Any]:
    return {"count": 0, "total_chars": 0, "min_chars": float("inf"), "max_chars": 0}


def compute_records_per_source_stats(source_counts: Counter[int]) -> dict[str, Any]:
    if not source_counts:
        return {"min": None, "max": None, "mean": None, "median": None}

    values = sorted(source_counts.values())
    n = len(values)
    mid = n // 2
    median = values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2
    return {
        "min": values[0],
        "max": values[-1],
        "mean": round(sum(values) / n, 2),
        "median": median,
    }


def sorted_counter(counter: Counter) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], str(item[0]))))


def collect_metadata(input_path: Path) -> dict[str, Any]:
    total_records = 0
    judgement_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    source_counts: Counter[int] = Counter()
    field_null_counts: Counter[str] = Counter()
    field_missing_counts: Counter[str] = Counter()
    unexpected_fields: Counter[str] = Counter()

    final_id_min: int | None = None
    final_id_max: int | None = None
    final_ids_seen: set[int] = set()

    text_length_stats = {field: new_length_stats() for field in TEXT_FIELDS}
    judgement_by_type: dict[str, Counter[str]] = {}

    with input_path.open("rb") as infile:
        for record in ijson.items(infile, "item"):
            total_records += 1

            record_keys = set(record.keys())
            for field in SCHEMA_FIELDS:
                if field not in record_keys:
                    field_missing_counts[field] += 1
                elif is_null_or_empty(record[field]):
                    field_null_counts[field] += 1

            for key in record_keys:
                if key not in SCHEMA_FIELDS:
                    unexpected_fields[key] += 1

            judgement = record.get("judgement")
            if judgement is not None:
                judgement_counts[str(judgement)] += 1

            claim_type = record.get("type")
            type_key = claim_type if claim_type is not None else "(null)"
            type_counts[type_key] += 1

            if judgement is not None:
                if type_key not in judgement_by_type:
                    judgement_by_type[type_key] = Counter()
                judgement_by_type[type_key][str(judgement)] += 1

            source = record.get("source")
            if source is not None:
                source_counts[int(source)] += 1

            final_id = record.get("final_id")
            if final_id is not None:
                final_id = int(final_id)
                final_ids_seen.add(final_id)
                final_id_min = final_id if final_id_min is None else min(final_id_min, final_id)
                final_id_max = final_id if final_id_max is None else max(final_id_max, final_id)

            for field in TEXT_FIELDS:
                value = record.get(field)
                if isinstance(value, str):
                    update_length_stats(text_length_stats[field], len(value))

            if total_records % 25000 == 0:
                print(f"  processed {total_records:,} records...")

    field_completeness: dict[str, dict[str, int | float]] = {}
    for field in SCHEMA_FIELDS:
        null_count = field_null_counts[field]
        missing_count = field_missing_counts[field]
        populated = total_records - null_count - missing_count
        field_completeness[field] = {
            "populated": populated,
            "null_or_empty": null_count,
            "missing": missing_count,
            "populated_pct": round(100 * populated / total_records, 4) if total_records else 0.0,
        }

    typed_records = sum(count for key, count in type_counts.items() if key != "(null)")
    null_type_records = type_counts.get("(null)", 0)

    metadata: dict[str, Any] = {
        "dataset": "ARAFA",
        "description": "Arabic fact-checking dataset of claim–evidence pairs",
        "source_file": str(input_path.as_posix()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_size_bytes": input_path.stat().st_size,
        "records": {
            "total": total_records,
            "unique_final_ids": len(final_ids_seen),
            "final_id_range": {"min": final_id_min, "max": final_id_max},
            "duplicate_final_ids": total_records - len(final_ids_seen),
        },
        "sources": {
            "unique_count": len(source_counts),
            "records_per_source": compute_records_per_source_stats(source_counts),
            "top_10_by_record_count": dict(
                sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
            ),
        },
        "judgement": {
            "counts": sorted_counter(judgement_counts),
            "percentages": {
                label: round(100 * count / total_records, 4)
                for label, count in sorted_counter(judgement_counts).items()
            },
        },
        "type": {
            "counts": sorted_counter(type_counts),
            "percentages": {
                label: round(100 * count / total_records, 4)
                for label, count in sorted_counter(type_counts).items()
            },
            "typed_records": typed_records,
            "null_type_records": null_type_records,
            "unique_non_null_types": sum(1 for key in type_counts if key != "(null)"),
        },
        "judgement_by_type": {
            claim_type: dict(sorted(counter.items()))
            for claim_type, counter in sorted(judgement_by_type.items(), key=lambda item: str(item[0]))
        },
        "field_completeness": field_completeness,
        "text_length_stats": {
            field: finalize_length_stats(text_length_stats[field]) for field in TEXT_FIELDS
        },
        "schema": {
            "expected_fields": SCHEMA_FIELDS,
            "unexpected_fields": dict(sorted(unexpected_fields.items())),
        },
    }

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/arafa/ARAFA.json"),
        help="Minified source JSON array",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/arafa/ARAFA.metadata.json"),
        help="Metadata output JSON file",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("Output path must differ from input path")

    print(f"Generating metadata from {args.input}")
    metadata = collect_metadata(args.input)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as outfile:
        json.dump(metadata, outfile, ensure_ascii=False, indent=2)
        outfile.write("\n")

    print(f"Done. Wrote metadata to {args.output}")
    print(f"  Total records: {metadata['records']['total']:,}")
    print(f"  Judgements: {metadata['judgement']['counts']}")
    print(f"  Unique sources: {metadata['sources']['unique_count']:,}")
    print(f"  Unique types (non-null): {metadata['type']['unique_non_null_types']}")

    sys.exit(0)


if __name__ == "__main__":
    main()
