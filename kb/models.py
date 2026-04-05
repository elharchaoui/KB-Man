from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class Document:
    document_id: str
    title: str
    source: str                              # URL or "user"
    type: Literal["url", "text", "code"]
    tags: list[str] = field(default_factory=list)
    added_at: datetime = field(default_factory=datetime.utcnow)
    fetched_at: datetime | None = None       # URL only


@dataclass
class ScoredResult:
    point_id: str
    document_id: str
    level: Literal["summary", "chunk"]
    score: float
    text: str
    payload: dict


@dataclass
class IterationRecord:
    iteration: int
    query: str
    results: list[ScoredResult]
    converged: bool
    convergence_ratio: float


@dataclass
class RetrievalOutput:
    iterations: list[IterationRecord]
    final_results: list[ScoredResult]        # deduplicated across iterations


@dataclass
class AddResult:
    document_id: str
    title: str
    type: Literal["url", "text", "code"]
    parent_stored: int
    child_chunks_stored: int


@dataclass
class DeleteResult:
    document_id: str
    deleted_summaries: int
    deleted_chunks: int
