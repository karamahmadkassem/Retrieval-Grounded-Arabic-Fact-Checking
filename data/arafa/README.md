# ARAFA Dataset

**ARAFA** (Arabic fact-checking dataset) contains **181,976** claim–evidence pairs for Arabic fact verification research.

## Files

| File | Size | Description |
|------|------|-------------|
| `ARAFA.json` | ~180 MB | Full dataset (minified JSON array, single line) |
| `ARAFA.formatted.json` | ~189 MB | Same data, pretty-printed with 2-space indent for readability |
| `ARAFA.metadata.json` | ~7 KB | Dataset statistics and schema summary (see [Metadata](#metadata)) |

Both JSON data files contain the same **181,976** records in the same order. Use `ARAFA.json` for loading in code; use `ARAFA.formatted.json` for manual inspection or diff-friendly viewing.

> **GitHub note:** This file exceeds GitHub's 100 MB limit. It is excluded from regular git by default (see root `.gitignore`). To version it on GitHub, use [Git LFS](#git-lfs-setup) as configured in `.gitattributes`.

## Record Schema

Each record is a JSON object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `final_id` | int | Unique record identifier |
| `source` | int | Source document identifier |
| `claim` | string | Claim to verify (Arabic) |
| `evidence` | string | Supporting evidence text (Arabic) |
| `judgement` | string | Verdict: `supported`, `refuted`, or `nei` (not enough information) |
| `reasoning` | string | Explanation for the judgement (Arabic) |
| `Entity_in_Claim` | string | Primary entity mentioned in the claim |
| `Co_referenced_in_Evidence` | string | Co-referenced entity in evidence |
| `Co_referenced_in_Text` | string | Co-referenced entity in claim text |
| `type` | string \| null | Claim transformation type (e.g. `Temporal Nuance`, `Scope Refinement`) |

## Metadata

`ARAFA.metadata.json` is the authoritative summary of dataset statistics. Regenerate it after any change to `ARAFA.json`:

```bash
py scripts/generate_arafa_metadata.py
```

Optional paths:

```bash
py scripts/generate_arafa_metadata.py --input data/arafa/ARAFA.json --output data/arafa/ARAFA.metadata.json
```

The metadata file includes:

| Section | Contents |
|---------|----------|
| `records` | Total count, unique `final_id` values, ID range, duplicates |
| `sources` | Unique source count, records-per-source distribution, top sources |
| `judgement` | Counts and percentages for `supported`, `refuted`, `nei` |
| `type` | Claim transformation type counts (10 non-null types + null) |
| `judgement_by_type` | Cross-tabulation of judgement vs. transformation type |
| `field_completeness` | Populated / null / missing counts per schema field |
| `text_length_stats` | Character length min/max/mean for text fields |
| `schema` | Expected field list and any unexpected fields found |

### Key metrics (from `ARAFA.metadata.json`)

| Metric | Value |
|--------|-------|
| Total records | 181,976 |
| Unique sources | 3,318 |
| Unique `final_id` values | 181,976 (range 1–183,312; 1,336 gaps) |
| Records per source (mean / median) | 54.9 / 51 |

**Judgement distribution**

| Judgement | Count | % |
|-----------|-------|---|
| `supported` | 111,303 | 61.2% |
| `refuted` | 42,109 | 23.1% |
| `nei` | 28,564 | 15.7% |

**Claim transformation types** (`type` field; 27.5% of records have a non-null type)

| Type | Count | % |
|------|-------|---|
| *(null)* | 131,976 | 72.5% |
| Temporal Nuance | 15,249 | 8.4% |
| Scope Refinement | 10,566 | 5.8% |
| Qualitative Shift | 9,050 | 5.0% |
| Quantitative Precision | 8,274 | 4.5% |
| Contextual Reframing | 3,946 | 2.2% |
| Relationship Reconfiguration | 2,797 | 1.5% |
| Causation vs. Correlation Logic | 108 | 0.06% |
| Conditional Logic | 5 | — |
| Range Logic | 4 | — |
| Duration Logic | 1 | — |

All `refuted` records have a non-null `type`; records with null `type` are only `supported` or `nei`.

## Validation

To confirm the formatted copy matches the original (valid JSON, same record count, deep equality):

```bash
py scripts/validate_arafa_json.py
```

Optional paths:

```bash
py scripts/validate_arafa_json.py --original data/arafa/ARAFA.json --formatted data/arafa/ARAFA.formatted.json
```

To regenerate the formatted file from the minified original:

```bash
py scripts/format_arafa_json.py
```

Requires `ijson` (`py -m pip install ijson`). Both scripts stream records and do not modify `ARAFA.json`.

## Loading Example

```python
import json

with open("data/arafa/ARAFA.json", encoding="utf-8") as f:
    records = json.load(f)

print(f"Loaded {len(records):,} records")
print(records[0]["judgement"])
```

## Git LFS Setup

To push the dataset to GitHub:

```bash
git lfs install
git lfs track "data/arafa/ARAFA.json"
git add .gitattributes
git add data/arafa/ARAFA.json
git commit -m "Add ARAFA dataset via Git LFS"
```

Remove the `data/arafa/ARAFA.json` line from `.gitignore` before adding the file.
