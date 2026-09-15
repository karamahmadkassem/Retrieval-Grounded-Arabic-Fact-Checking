"""Rebuild wikipedia_chunks.json from Phase 0 article cache (no API calls)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arabic_text_utils import build_sentence_windows, segment_sentences
from build_gold_passages import MAX_WINDOW, _raw_from_window

ROOT = Path(__file__).resolve().parents[1]
verification = json.loads((ROOT / "data/arafa/verification/source_verification.json").read_text(encoding="utf-8"))
cache = json.loads((ROOT / "data/arafa/evidence_match_full_article_cache.json").read_text(encoding="utf-8"))
out_path = ROOT / "data/arafa/wikipedia_chunks.json"

chunks_by_source = {}
for key, info in verification.items():
    if not info.get("found"):
        continue
    source_id = int(key)
    text = cache.get(key, "")
    if not text:
        continue
    sentences = segment_sentences(text)
    windows = build_sentence_windows(sentences, max_window=MAX_WINDOW)
    chunks_by_source[key] = {
        "title": info.get("title"),
        "url": info.get("url"),
        "num_sentences": len(sentences),
        "chunks": [_raw_from_window(source_id, w) for w in windows],
    }

tmp_path = out_path.with_suffix(".json.tmp")
with open(tmp_path, "w", encoding="utf-8") as f:
    json.dump(chunks_by_source, f, ensure_ascii=False, separators=(",", ":"))
tmp_path.replace(out_path)
print(f"Wrote {len(chunks_by_source)} sources to {out_path}")
