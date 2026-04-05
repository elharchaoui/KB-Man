from __future__ import annotations

import json

from openai import AsyncOpenAI

from kb.models import ScoredResult

_PROMPT = """\
You are helping refine a knowledge base search query.

Original query: {original_query}

The previous search returned scattered results across multiple documents:
{results_summary}

Produce a more targeted search query that will retrieve more focused results.
Return ONLY the new query string, nothing else."""


def _summarize_results(results: list[ScoredResult]) -> str:
    seen: dict[str, dict] = {}
    for r in results:
        if r.document_id not in seen:
            seen[r.document_id] = {
                "title": r.payload.get("title", "untitled"),
                "type": r.payload.get("type", "unknown"),
                "best_score": r.score,
            }
    lines = [
        f"- [{v['type']}] {v['title']} (score: {v['best_score']:.2f})"
        for v in seen.values()
    ]
    return "\n".join(lines)


async def reformulate(
    original_query: str,
    accumulated: list[ScoredResult],
    client: AsyncOpenAI,
    model: str,
) -> str:
    prompt = _PROMPT.format(
        original_query=original_query,
        results_summary=_summarize_results(accumulated),
    )
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=128,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Soft failure — fall back to original query
        return original_query
