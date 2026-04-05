from __future__ import annotations

from collections import Counter

from kb.models import ScoredResult


def check(results: list[ScoredResult], threshold: float = 0.6) -> tuple[bool, float]:
    """
    Returns (converged, ratio).
    Converged when >= threshold of results share the same document_id.
    """
    if not results:
        return False, 0.0
    doc_ids = [r.document_id for r in results]
    top_count = Counter(doc_ids).most_common(1)[0][1]
    ratio = top_count / len(doc_ids)
    return ratio >= threshold, ratio
