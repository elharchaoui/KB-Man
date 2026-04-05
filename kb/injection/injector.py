from __future__ import annotations

from collections import defaultdict

import tiktoken

from kb.config import Config
from kb.models import RetrievalOutput, ScoredResult

_enc = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def build(output: RetrievalOutput, config: Config) -> str | None:
    if not output.final_results:
        return None

    # Group results by document_id
    by_doc: dict[str, dict] = defaultdict(lambda: {"chunks": [], "summary": None})
    for r in output.final_results:
        if r.level == "chunk":
            by_doc[r.document_id]["chunks"].append(r)
        else:
            by_doc[r.document_id]["summary"] = r

    # Build iteration history header
    header_lines = [f"[KB RETRIEVAL — {len(output.iterations)} iteration(s)]", ""]
    for rec in output.iterations:
        status = "converged: yes" if rec.converged else f"converged: no (ratio={rec.convergence_ratio:.2f})"
        top_docs = _top_doc_titles(rec.results)
        header_lines.append(f"Iteration {rec.iteration} | query: \"{rec.query}\"")
        header_lines.append(f"  → {top_docs} — {status}")
    header_lines.append("")

    # Build content blocks — chunks preferred, summary as fallback
    content_blocks: list[tuple[float, str]] = []
    for doc_id, data in by_doc.items():
        chunks: list[ScoredResult] = data["chunks"]
        summary: ScoredResult | None = data["summary"]

        if chunks:
            chunks.sort(key=lambda r: r.score, reverse=True)
            best = chunks[0]
            meta = _meta_line(best)
            block = f"Source: {meta}\n---\n" + "\n\n".join(c.text for c in chunks)
            content_blocks.append((best.score, block))
        elif summary:
            meta = _meta_line(summary)
            block = f"Source: {meta}\n---\n{summary.text}"
            content_blocks.append((summary.score, block))

    # Sort by score, enforce token budget
    content_blocks.sort(key=lambda x: x[0], reverse=True)
    budget = config.max_injected_tokens
    selected: list[str] = []
    used = 0
    for _, block in content_blocks:
        cost = _token_count(block)
        if used + cost > budget:
            break
        selected.append(block)
        used += cost

    if not selected:
        return None

    header = "\n".join(header_lines)
    body = "\n\n".join(selected)
    return f"{header}[RETRIEVED KNOWLEDGE — deduplicated]\n\n{body}\n\n[END KB RETRIEVAL]"


def _meta_line(r: ScoredResult) -> str:
    source = r.payload.get("source", "user")
    title = r.payload.get("title", "untitled")
    added = r.payload.get("added_at", "")[:10]
    fetched = r.payload.get("fetched_at", "")[:10]
    date = fetched or added
    return f"{source} | \"{title}\" | {date}"


def _top_doc_titles(results: list[ScoredResult]) -> str:
    seen: dict[str, float] = {}
    best_result: dict[str, ScoredResult] = {}
    for r in results:
        doc_id = r.document_id
        if doc_id not in seen or r.score > seen[doc_id]:
            seen[doc_id] = r.score
            best_result[doc_id] = r
    titles = sorted(seen.keys(), key=lambda d: seen[d], reverse=True)[:3]
    parts = []
    for t in titles:
        r = best_result[t]
        title = r.payload.get("title", t[:8])
        parts.append(f'"{title}" ({seen[t]:.2f})')
    return ", ".join(parts)
