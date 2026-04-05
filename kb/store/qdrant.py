from __future__ import annotations

from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FusionQuery,
    Fusion,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from kb.config import Config
from kb.models import ScoredResult

COLLECTION_SUMMARIES = "parent_summaries"
COLLECTION_CHUNKS = "child_chunks"


class QdrantStore:
    def __init__(self, config: Config) -> None:
        path = str(Path(config.qdrant_path).expanduser())
        self._client = AsyncQdrantClient(path=path)
        self._config = config

    async def setup(self) -> None:
        """Create collections if they don't exist."""
        existing = {c.name for c in await self._client.get_collections().then(lambda r: r.collections)}
        for name in (COLLECTION_SUMMARIES, COLLECTION_CHUNKS):
            if name not in existing:
                await self._client.create_collection(
                    collection_name=name,
                    vectors_config={"dense": VectorParams(
                        size=self._config.embedding_dimensions,
                        distance=Distance.COSINE,
                    )},
                    sparse_vectors_config={"sparse": SparseVectorParams()},
                )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def upsert_summary(
        self,
        document_id: str,
        dense: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        payload: dict,
    ) -> None:
        await self._client.upsert(
            collection_name=COLLECTION_SUMMARIES,
            points=[PointStruct(
                id=document_id,
                vector={
                    "dense": dense,
                    "sparse": SparseVector(indices=sparse_indices, values=sparse_values),
                },
                payload=payload,
            )],
        )

    async def upsert_chunks(self, points: list[dict]) -> None:
        """points: list of {id, dense, sparse_indices, sparse_values, payload}"""
        await self._client.upsert(
            collection_name=COLLECTION_CHUNKS,
            points=[
                PointStruct(
                    id=p["id"],
                    vector={
                        "dense": p["dense"],
                        "sparse": SparseVector(
                            indices=p["sparse_indices"],
                            values=p["sparse_values"],
                        ),
                    },
                    payload=p["payload"],
                )
                for p in points
            ],
        )

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_by_document_id(self, document_id: str) -> tuple[int, int]:
        """Delete all points for a document. Returns (summaries_deleted, chunks_deleted)."""
        doc_filter = Filter(must=[
            FieldCondition(key="document_id", match=MatchValue(value=document_id))
        ])

        before_s = await self._client.count(COLLECTION_SUMMARIES, count_filter=doc_filter)
        before_c = await self._client.count(COLLECTION_CHUNKS, count_filter=doc_filter)

        await self._client.delete(COLLECTION_SUMMARIES, points_selector=doc_filter)
        await self._client.delete(COLLECTION_CHUNKS, points_selector=doc_filter)

        return before_s.count, before_c.count

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def hybrid_query(
        self,
        collection: str,
        dense: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        limit: int,
        candidate_limit: int,
    ) -> list[ScoredResult]:
        results = await self._client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dense, using="dense", limit=candidate_limit),
                Prefetch(
                    query=SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=candidate_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        level = "summary" if collection == COLLECTION_SUMMARIES else "chunk"
        return [
            ScoredResult(
                point_id=str(r.id),
                document_id=r.payload.get("document_id", str(r.id)),
                level=level,
                score=r.score,
                text=r.payload.get("text", ""),
                payload=r.payload,
            )
            for r in results.points
        ]

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    async def fetch_all_texts(self) -> dict[str, dict[str, str]]:
        """
        Returns texts for BM25 corpus bootstrap.
        {
          "summaries": { point_id: text, ... },
          "chunks":    { point_id: text, ... },
        }
        """
        summaries: dict[str, str] = {}
        chunks: dict[str, str] = {}

        offset = None
        while True:
            result, offset = await self._client.scroll(
                COLLECTION_SUMMARIES,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for p in result:
                summaries[str(p.id)] = p.payload.get("text", "")
            if offset is None:
                break

        offset = None
        while True:
            result, offset = await self._client.scroll(
                COLLECTION_CHUNKS,
                limit=100,
                offset=offset,
                with_payload=True,
            )
            for p in result:
                chunks[str(p.id)] = p.payload.get("text", "")
            if offset is None:
                break

        return {"summaries": summaries, "chunks": chunks}

    async def collection_stats(self) -> dict:
        s = await self._client.get_collection(COLLECTION_SUMMARIES)
        c = await self._client.get_collection(COLLECTION_CHUNKS)
        return {
            "summaries": s.points_count,
            "chunks": c.points_count,
        }
