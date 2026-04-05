from __future__ import annotations

from openai import AsyncOpenAI

from kb.config import Config
from kb.models import IterationRecord, RetrievalOutput, ScoredResult
from kb.retrieval import convergence as conv
from kb.retrieval import reformulator
from kb.retrieval.searcher import Searcher


class IterativeRetriever:
    def __init__(
        self,
        searcher: Searcher,
        llm_client: AsyncOpenAI,
        llm_model: str,
        config: Config,
    ) -> None:
        self._searcher = searcher
        self._llm_client = llm_client
        self._llm_model = llm_model
        self._config = config

    async def retrieve(self, query: str) -> RetrievalOutput:
        accumulated: list[ScoredResult] = []
        records: list[IterationRecord] = []
        current_query = query

        for i in range(1, self._config.max_iterations + 1):
            results = await self._searcher.search(current_query)

            # High-confidence early exit
            if results and results[0].score >= self._config.high_confidence_threshold:
                records.append(IterationRecord(
                    iteration=i,
                    query=current_query,
                    results=results,
                    converged=True,
                    convergence_ratio=1.0,
                ))
                accumulated.extend(results)
                break

            converged, ratio = conv.check(results, self._config.convergence_ratio)
            records.append(IterationRecord(
                iteration=i,
                query=current_query,
                results=results,
                converged=converged,
                convergence_ratio=ratio,
            ))
            accumulated.extend(results)

            if converged or i == self._config.max_iterations:
                break

            current_query = await reformulator.reformulate(
                original_query=query,
                accumulated=accumulated,
                client=self._llm_client,
                model=self._llm_model,
            )

        return RetrievalOutput(
            iterations=records,
            final_results=_deduplicate(accumulated, self._config.similarity_threshold),
        )


def _deduplicate(results: list[ScoredResult], threshold: float) -> list[ScoredResult]:
    seen_chunks: set[str] = set()
    seen_summaries: set[str] = set()
    deduped: list[ScoredResult] = []

    for r in sorted(results, key=lambda x: x.score, reverse=True):
        if r.score < threshold:
            continue
        if r.level == "chunk":
            if r.point_id not in seen_chunks:
                seen_chunks.add(r.point_id)
                deduped.append(r)
        else:
            if r.document_id not in seen_summaries:
                seen_summaries.add(r.document_id)
                deduped.append(r)

    return deduped
