from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Storage
    qdrant_path: str = "~/.kb/qdrant"

    # Chunking
    child_chunk_size: int = 128
    normalization_skip_threshold: int = 300

    # Retrieval
    top_k: int = 5
    candidate_multiplier: int = 3
    similarity_threshold: float = 0.75
    high_confidence_threshold: float = 0.92
    convergence_ratio: float = 0.6
    rrf_k: int = 60
    max_iterations: int = 3

    # Injection
    max_injected_tokens: int = 4000

    # Encoding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # Ingest
    summarize_urls: bool = True

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None:
            path = Path.home() / ".kb" / "config.json"
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open() as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def save(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".kb" / "config.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.__dict__, f, indent=2)
