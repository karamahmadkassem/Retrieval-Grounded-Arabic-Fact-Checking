"""Document claim/source count gaps for methodology notes."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arabic_text_utils import segment_sentences

ROOT = Path(__file__).resolve().parents[1]
claims = json.loads((ROOT / "data/arafa/ARAFA.json").read_text(encoding="utf-8"))
ver = json.loads((ROOT / "data/arafa/verification/source_verification.json").read_text(encoding="utf-8"))
gold = json.loads((ROOT / "data/arafa/gold_passages.json").read_text(encoding="utf-8"))
cache = json.loads((ROOT / "data/arafa/evidence_match_full_article_cache.json").read_text(encoding="utf-8"))
chunks = json.loads((ROOT / "data/arafa/wikipedia_chunks.json").read_text(encoding="utf-8"))

found = [c for c in claims if ver.get(str(c["source"]), {}).get("found")]
not_found = [c for c in claims if not ver.get(str(c["source"]), {}).get("found")]
found_ids = {int(k) for k, v in ver.items() if v.get("found")}
missing_cache = sorted(found_ids - {int(k) for k in cache})
missing_chunks = sorted(found_ids - {int(k) for k in chunks})

print("=== Claim counts ===")
print(f"total claims:           {len(claims)}")
print(f"verified found:         {len(found)}")
print(f"not found:              {len(not_found)}")
print(f"gold entries:           {len(gold)}")
print(f"gap (total - found):    {len(claims) - len(found)}")

print("\n=== Not-found breakdown ===")
print(f"unique missing sources: {len({c['source'] for c in not_found})}")
for sid, n in Counter(c["source"] for c in not_found).most_common(10):
    print(f"  source {sid}: {n} claims  title={ver.get(str(sid), {}).get('title')}")

print("\n=== Source coverage ===")
print(f"verified found sources: {len(found_ids)}")
print(f"in article cache:       {len(found_ids) - len(missing_cache)}")
print(f"in wikipedia_chunks:    {len(found_ids) - len(missing_chunks)}")
print(f"found but no cache:     {len(missing_cache)}")
if missing_cache:
    print(f"  ids: {missing_cache}")

print("\n=== Found sources with empty/unusable text (no chunks) ===")
for sid in missing_chunks:
    info = ver[str(sid)]
    text = cache.get(str(sid), "")
    print(
        f"  source {sid}: title={info.get('title')!r}  "
        f"cache_chars={len(text)}  sentences={len(segment_sentences(text))}"
    )
