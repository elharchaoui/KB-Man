from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


class SparseEncoder:
    """
    Maintains a BM25 corpus in memory.
    Both collections (parent_summaries, child_chunks) share the same encoder
    since they are searched independently — each collection gets its own instance.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def bootstrap(self, texts: dict[str, str]) -> None:
        """Load existing corpus from Qdrant on startup."""
        self._ids = list(texts.keys())
        self._tokenized = [_tokenize(t) for t in texts.values()]
        self._rebuild()

    def add(self, id: str, text: str) -> None:
        if id in self._ids:
            idx = self._ids.index(id)
            self._tokenized[idx] = _tokenize(text)
        else:
            self._ids.append(id)
            self._tokenized.append(_tokenize(text))
        self._rebuild()

    def remove(self, id: str) -> None:
        if id not in self._ids:
            return
        idx = self._ids.index(id)
        self._ids.pop(idx)
        self._tokenized.pop(idx)
        self._dirty = True

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> SparseVector:
        if self._dirty:
            self._rebuild()
        if self._bm25 is None or not self._ids:
            return SparseVector(indices=[], values=[])

        tokens = _tokenize(text)
        scores = self._bm25.get_scores(tokens)

        indices = [i for i, s in enumerate(scores) if s > 0.0]
        values = [float(scores[i]) for i in indices]
        return SparseVector(indices=indices, values=values)

    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None
        self._dirty = False
