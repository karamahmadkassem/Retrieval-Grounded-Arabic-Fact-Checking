"""
arabic_text_utils.py

Shared text utilities for Arabic Wikipedia processing: normalization and
sentence segmentation/windowing. Used by both the gold-passage builder
(this phase) and, later, the full-corpus indexing script (Phase 1
proper), so the chunking definition stays identical across both.
"""

import re
from typing import List

TASHKEEL_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")
WHITESPACE_RE = re.compile(r"\s+")

# Sentence-ending punctuation, Arabic and Latin. We split on runs of these
# followed by whitespace or end-of-string.
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?؟]+(?=\s|$)")


def normalize(text: str) -> str:
    """Strip Arabic diacritics (tashkeel) and collapse whitespace, so
    ARAFA's evidence text and live Wikipedia's current wording compare
    fairly despite diacritic/formatting drift."""
    if not text:
        return ""
    text = TASHKEEL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def segment_sentences(text: str) -> List[str]:
    """
    Heuristic Arabic-aware sentence splitter.

    Splits on '.', '!', '?', '؟' EXCEPT when the punctuation sits between
    two digits (a decimal number, e.g. "3.5 مليون"), which is not treated
    as a sentence boundary.

    This is a regex heuristic, not a full NLP sentence tokenizer. Arabic's
    heavy use of coordinating conjunctions (و) can produce very long
    compound sentences that a human would perceive as multiple ideas but
    that this splitter will keep as one sentence -- that's expected and
    fine, since our windowing below groups 1-3 sentences anyway. Spot-check
    a sample before trusting this at full scale; if segmentation quality
    is poor, a dedicated tool (e.g. CAMeL Tools' sentence splitter) is the
    upgrade path.
    """
    text = text.strip()
    if not text:
        return []

    sentences = []
    start = 0
    for m in SENTENCE_BOUNDARY_RE.finditer(text):
        end = m.end()
        pre_char = text[m.start() - 1] if m.start() > 0 else ""
        post_char = text[end:end + 1]
        if pre_char.isdigit() and post_char.isdigit():
            continue  # decimal number, e.g. "3.5" -- not a sentence break
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end

    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)

    return sentences


def build_sentence_windows(sentences: List[str], max_window: int = 3) -> List[dict]:
    """
    Builds overlapping sliding windows of 1..max_window consecutive
    sentences. Each window is a candidate retrieval chunk.

    A 3-sentence article produces windows of size 1 (x3), size 2 (x2),
    and size 3 (x1) -- 6 total, covering both single-sentence evidence
    (~83% of FEVER-style cases) and short multi-sentence evidence (~17%)
    without ever cutting a sentence in half.
    """
    n = len(sentences)
    windows = []
    for size in range(1, max_window + 1):
        for start in range(0, n - size + 1):
            end = start + size - 1
            windows.append({
                "sentence_start": start,
                "sentence_end": end,
                "window_size": size,
                "text": " ".join(sentences[start:end + 1]),
            })
    return windows


def build_windows_of_size(sentences: List[str], size: int) -> List[dict]:
    """Build sliding windows of exactly one size (used for long evidence spans)."""
    n = len(sentences)
    if size < 1 or size > n:
        return []
    return [
        {
            "sentence_start": start,
            "sentence_end": start + size - 1,
            "window_size": size,
            "text": " ".join(sentences[start : start + size]),
        }
        for start in range(0, n - size + 1)
    ]
