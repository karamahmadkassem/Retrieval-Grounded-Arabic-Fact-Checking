"""Checks whether none-match claims correlate with long evidence spans."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arabic_text_utils import segment_sentences

ROOT = Path(__file__).resolve().parents[1]
ARAFA = json.loads((ROOT / "data/arafa/ARAFA.json").read_text(encoding="utf-8"))
GOLD = json.loads((ROOT / "data/arafa/gold_passages.json").read_text(encoding="utf-8"))

claims_by_id = {c["final_id"]: c for c in ARAFA}
buckets = {"exact": [], "fuzzy": [], "none": []}
for final_id_str, result in GOLD.items():
    claim = claims_by_id.get(int(final_id_str))
    if not claim:
        continue
    n_sentences = len(segment_sentences(claim["evidence"]))
    match_type = result["match_type"]
    if match_type in buckets:
        buckets[match_type].append(n_sentences)

lines = []
for label, counts in buckets.items():
    if not counts:
        continue
    avg = sum(counts) / len(counts)
    over_3 = sum(1 for c in counts if c > 3)
    lines.append(
        f"{label:6s}: n={len(counts):6d}  avg_sentences={avg:.2f}  "
        f"evidence>3_sentences={over_3} ({over_3/len(counts):.1%})"
    )

out = ROOT / "results" / "none_case_diagnosis.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nWrote {out}")
