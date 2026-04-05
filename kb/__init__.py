from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
from openai import AsyncOpenAI

from kb.config import Config
from kb.encoding.dense import DenseEncoder
from kb.encoding.sparse import SparseEncoder
from kb.ingest import chunker, detector, normalizer, summarizer, tagger
from kb.ingest.fetcher import fetch_url
from kb.injection import injector
from kb.models import AddResult, DeleteResult, Document, RetrievalOutput
from kb.retrieval.iterator import IterativeRetriever
from kb.retrieval.searcher import Searcher
from kb.store.qdrant import QdrantStore

_enc = tiktoken.get_encoding("cl100k_base")

_INTENT_SKIP_TOKENS = 5


class KnowledgeBase:
    def __init__(
        self,
        llm_client: AsyncOpenAI,
        embedding_client: AsyncOpenAI,
        llm_model: str,
        config: Config | None = None,
    ) -> None:
        self._config = config or Config()
        self._llm = llm_client
        self._llm_model = llm_model

        self._store = QdrantStore(self._config)
        self._dense = DenseEncoder(embedding_client, self._config)
        self._sparse_summaries = SparseEncoder()
        self._sparse_chunks = SparseEncoder()

        searcher = Searcher(
            self._store,
            self._dense,
            self._sparse_summaries,
            self._sparse_chunks,
            self._config,
        )
        self._retriever = IterativeRetriever(
            searcher, llm_client, llm_model, self._config
        )

        self._documents: dict[str, Document] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        await self._store.setup()
        texts = await self._store.fetch_all_texts()
        self._sparse_summaries.bootstrap(texts["summaries"])
        self._sparse_chunks.bootstrap(texts["chunks"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add(self, input: str, tags: list[str] | None = None) -> AddResult:
        tags = tags or []
        input_type = detector.detect(input)
        raw_text = input
        title = "user note"
        fetched_at = None

        # Fetch URL
        if input_type == "url":
            raw_text, title, fetched_at = await fetch_url(input)

        # RAG readiness — normalize
        if normalizer.should_normalize(raw_text, input_type, self._config):
            normalized = await normalizer.normalize(raw_text, input_type, self._llm, self._llm_model)
        else:
            normalized = raw_text

        # Summary + auto-tags run concurrently (one fewer serial LLM round-trip)
        import asyncio as _asyncio
        summary_text, auto_tags = await _asyncio.gather(
            summarizer.summarize(normalized, input_type, self._llm, self._llm_model),
            tagger.suggest_tags(normalized, input_type, self._llm, self._llm_model),
        )

        # Merge: user-provided tags first, then auto-tags (no duplicates)
        all_tags = list(dict.fromkeys(tags + [t for t in auto_tags if t not in tags]))

        chunks = chunker.chunk(normalized, input_type, self._config)  # list[(text, header)]

        document_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Encode summary
        summary_dense = await self._dense.encode(summary_text)
        summary_sparse = self._sparse_summaries.encode(summary_text)
        self._sparse_summaries.add(document_id, summary_text)

        summary_payload = {
            "document_id": document_id,
            "title": title,
            "source": input if input_type == "url" else "user",
            "type": input_type,
            "tags": all_tags,
            "added_at": now,
            "text": summary_text,
        }
        if fetched_at:
            summary_payload["fetched_at"] = fetched_at.isoformat()

        await self._store.upsert_summary(
            document_id=document_id,
            dense=summary_dense,
            sparse_indices=summary_sparse.indices,
            sparse_values=summary_sparse.values,
            payload=summary_payload,
        )

        # Encode and store chunks
        chunk_texts = [text for text, _ in chunks]
        chunk_dense_vecs = await self._dense.encode_batch(chunk_texts)
        chunk_points = []
        for idx, ((chunk_text, chunk_header), dense_vec) in enumerate(zip(chunks, chunk_dense_vecs)):
            chunk_id = str(uuid.uuid4())
            sparse_vec = self._sparse_chunks.encode(chunk_text)
            self._sparse_chunks.add(chunk_id, chunk_text)
            payload = {
                "document_id": document_id,
                "chunk_index": idx,
                "source": input if input_type == "url" else "user",
                "type": input_type,
                "added_at": now,
                "text": chunk_text,
            }
            if chunk_header is not None:
                payload["chunk_header"] = chunk_header
            chunk_points.append({
                "id": chunk_id,
                "dense": dense_vec,
                "sparse_indices": sparse_vec.indices,
                "sparse_values": sparse_vec.values,
                "payload": payload,
            })

        await self._store.upsert_chunks(chunk_points)

        doc = Document(
            document_id=document_id,
            title=title,
            source=input if input_type == "url" else "user",
            type=input_type,
            tags=all_tags,
            added_at=datetime.fromisoformat(now),
            fetched_at=fetched_at,
        )
        self._documents[document_id] = doc

        return AddResult(
            document_id=document_id,
            title=title,
            type=input_type,
            parent_stored=1,
            child_chunks_stored=len(chunks),
            auto_tags=auto_tags,
        )

    async def search(self, query: str, k: int | None = None) -> RetrievalOutput:
        if k is not None:
            original_top_k = self._config.top_k
            self._config.top_k = k
        output = await self._retriever.retrieve(query)
        if k is not None:
            self._config.top_k = original_top_k
        return output

    async def delete(self, document_id: str) -> DeleteResult:
        deleted_s, deleted_c = await self._store.delete_by_document_id(document_id)
        self._sparse_summaries.remove(document_id)
        self._documents.pop(document_id, None)
        return DeleteResult(
            document_id=document_id,
            deleted_summaries=deleted_s,
            deleted_chunks=deleted_c,
        )

    def list(self) -> list[Document]:
        return list(self._documents.values())

    async def retrieve_for_turn(self, message: str) -> str | None:
        """Single-pass retrieval for auto-injection. Returns formatted context or None."""
        if len(_enc.encode(message)) <= _INTENT_SKIP_TOKENS:
            return None
        output = await self._retriever.retrieve(message)
        return injector.build(output, self._config)
