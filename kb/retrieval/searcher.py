from __future__ import annotations

from kb.config import Config
from kb.encoding.dense import DenseEncoder
from kb.encoding.sparse import SparseEncoder
from kb.models import ScoredResult
from kb.store.qdrant import QdrantStore, COLLECTION_SUMMARIES, COLLECTION_CHUNKS


class Searcher:
    def __init__(
        self,
        store: QdrantStore,
        dense: DenseEncoder,
        sparse_summaries: SparseEncoder,
        sparse_chunks: SparseEncoder,
        config: Config,
    ) -> None:
        self._store = store
        self._dense = dense
        self._sparse_summaries = sparse_summaries
        self._sparse_chunks = sparse_chunks
        self._config = config

    async def search(self, query: str) -> list[ScoredResult]:
        k = self._config.top_k
        candidate_limit = k * self._config.candidate_multiplier

        dense_vec = await self._dense.encode(query)

        sparse_s = self._sparse_summaries.encode(query)
        sparse_c = self._sparse_chunks.encode(query)

        summary_results, chunk_results = await _gather(
            self._store.hybrid_query(
                COLLECTION_SUMMARIES, dense_vec,
                sparse_s.indices, sparse_s.values,
                limit=k, candidate_limit=candidate_limit,
            ),
            self._store.hybrid_query(
                COLLECTION_CHUNKS, dense_vec,
                sparse_c.indices, sparse_c.values,
                limit=k, candidate_limit=candidate_limit,
            ),
        )

        combined = summary_results + chunk_results
        combined.sort(key=lambda r: r.score, reverse=True)
        return combined[:k * 2]


async def _gather(*coros):
    import asyncio
    return await asyncio.gather(*coros)
