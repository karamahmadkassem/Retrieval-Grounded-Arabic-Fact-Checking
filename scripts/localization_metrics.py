"""Shared metrics for in-article evidence localization."""

from __future__ import annotations

K_VALUES = [1, 5, 10]


def parse_span(chunk_id: str) -> tuple[int, int] | None:
    """Parse 'source:start-end' into (start, end) inclusive sentence indices."""
    if not chunk_id or ":" not in chunk_id:
        return None
    span = chunk_id.rsplit(":", 1)[-1]
    if "-" not in span:
        return None
    try:
        a, b = span.split("-", 1)
        return int(a), int(b)
    except ValueError:
        return None


def span_iou(pred_id: str, gold_id: str) -> float:
    p, g = parse_span(pred_id), parse_span(gold_id)
    if p is None or g is None:
        return 0.0
    p0, p1 = p
    g0, g1 = g
    inter = max(0, min(p1, g1) - max(p0, g0) + 1)
    union = (p1 - p0 + 1) + (g1 - g0 + 1) - inter
    return inter / union if union else 0.0


def ranks_from_ids(retrieved_ids: list[str], gold_id: str, iou_threshold: float = 0.5):
    exact = None
    if gold_id in retrieved_ids:
        exact = retrieved_ids.index(gold_id) + 1
    relaxed = exact
    if relaxed is None:
        for i, cid in enumerate(retrieved_ids, start=1):
            if span_iou(cid, gold_id) >= iou_threshold:
                relaxed = i
                break
    return exact, relaxed


def summarize(rows: list[dict], rank_key: str = "rank") -> dict | None:
    n = len(rows)
    if n == 0:
        return None
    hits = {k: 0 for k in K_VALUES}
    rr = 0.0
    for r in rows:
        rank = r.get(rank_key)
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits[k] += 1
        rr += (1.0 / rank) if rank else 0.0
    return {
        "n": n,
        **{f"recall@{k}": round(hits[k] / n, 4) for k in K_VALUES},
        "mrr": round(rr / n, 4),
    }
