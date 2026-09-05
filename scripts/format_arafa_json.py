"""Create a pretty-printed copy of ARAFA.json without modifying the original."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ijson


def format_array(input_path: Path, output_path: Path, indent: int = 2) -> int:
    """Stream-parse input JSON array and write a formatted copy. Returns record count."""
    count = 0
    with input_path.open("rb") as infile, output_path.open("w", encoding="utf-8") as outfile:
        outfile.write("[\n")
        first = True
        for record in ijson.items(infile, "item"):
            if not first:
                outfile.write(",\n")
            first = False
            outfile.write(json.dumps(record, ensure_ascii=False, indent=indent))
            count += 1
            if count % 25000 == 0:
                print(f"  formatted {count:,} records...")
        outfile.write("\n]\n")
    return count


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
        default=Path("data/arafa/ARAFA.formatted.json"),
        help="Pretty-printed destination file",
    )
    parser.add_argument("--indent", type=int, default=2, help="Indent width (default: 2)")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("Output path must differ from input path")

    print(f"Formatting {args.input} -> {args.output}")
    count = format_array(args.input, args.output, indent=args.indent)
    print(f"Done. Wrote {count:,} records to {args.output}")


if __name__ == "__main__":
    main()
