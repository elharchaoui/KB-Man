# Knowledge Base (KB) — Design Document

**Status:** Draft  
**Date:** 2026-04-04  
**Project:** nanobot

---

## Overview

A personal knowledge base layer for nanobot. The user explicitly curates knowledge — URLs, documents, notes, code snippets — and the agent queries it on their behalf. The KB is user-driven: nothing is stored without an explicit instruction.

---

## Goals

- Give the user a queryable knowledge base that survives session restarts
- Support adding knowledge from URLs, raw text, and code snippets
- Surface relevant knowledge automatically or on demand during conversations
- Keep the system local-first and inspectable

---

## Non-Goals

- Automatic ingestion without explicit user instruction
- Multi-user isolation (single-user system)
- Real-time web crawling or RSS feeds
- Cloud sync (local-first; sync is out of scope)
- Semantic deduplication (handled by the user)

---

## Architecture

```
User instruction: "remember this URL / note / snippet"
    │
    ▼
┌─────────────────────┐
│  Ingest Layer        │  Smart detect → fetch URLs
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  RAG Readiness       │  LLM → clean structured Markdown
│  (Normalizer)        │  Strip noise, impose ## headers,
│                      │  fence code blocks, preserve all facts
└────────┬────────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌─────────────────┐    ┌─────────────────────┐
│  Summarizer      │    │  Structure-aware     │
│  (parent)        │    │  Chunker (children)  │
│  LLM summary of  │    │  Splits normalized   │
│  normalized doc  │    │  doc into ~128-token │
│                  │    │  boundary-aligned    │
│                  │    │  chunks              │
└────────┬────────┘    └──────────┬──────────┘
         └──────────┬─────────────┘
                    ▼
┌─────────────────────┐
│  Embedder            │  Embed both parent (summary) and child (chunks)
│                      │  Error → raise immediately, store nothing
└────────┬────────────┘
         │
         │
         ▼
┌─────────────────────┐
│  Sparse Encoder      │  BM25 sparse vectors via rank_bm25
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Qdrant              │  Two collections, each with dense + sparse vectors
│  (local, embedded)   │  · parent_summaries
│                      │  · child_chunks
└────────┬────────────┘
                        │ (at query time)
                        ▼
┌─────────────────────────────────────┐
│  Hybrid Retriever                    │
│  Vector + BM25 → RRF fusion          │
│  Iterative loop (max 3 iterations)   │
│  LLM convergence check per iteration │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Context Injector    │  Iteration history + deduplicated chunks
│                      │  injected into agent context
└─────────────────────┘
```

---

## Ingest Layer — Smart Detect

Detection logic:

```
input
 ├── matches URL pattern?     → fetch → normalize → summarize + chunk
 ├── looks like a code block? → normalize → summarize + chunk at boundaries
 └── otherwise (text/note)    → normalize (if long) → summarize + chunk
```

| Scenario | Normalize | Summary (parent) | Chunks (child) |
|---|---|---|---|
| URL | Always — raw HTML is always noisy | LLM summary of normalized doc | Normalized doc, chunked |
| Text / note (> 300 tokens) | Yes — impose structure | LLM summary of normalized doc | Normalized doc, chunked |
| Text / note (≤ 300 tokens) | Skip — already readable | LLM summary or full text if very short | Full text, chunked |
| Code snippet | Yes — add section headers, normalize style | LLM description of what the code does | Split at function/class boundaries |

### URL fetching

- Use `WebFetch` to retrieve the page
- Pass raw content through the RAG Readiness normalizer before any other step
- Metadata on both parent and children: `source_url`, `fetched_at`

### RAG Readiness — Normalizer

Sits between the fetcher and the summarizer/chunker. Takes raw content (HTML, messy text, code) and produces a clean, consistently structured Markdown document.

**LLM prompt:**
```
Convert the following content into a clean, well-structured Markdown document.

Rules:
- Use ## for major sections, ### for subsections
- Separate paragraphs with blank lines
- Wrap all code in fenced blocks with language hints (```python, ```bash, etc.)
- Preserve all technical details, facts, names, and numbers exactly — do not summarize
- Remove navigation, ads, footers, sidebars, and any non-content elements

Content:
{raw_content}
```

**Why this matters:** The summarizer and chunker both operate on the normalized document, not the raw input. Summarization quality improves because the LLM works from clean structured text. Chunking reliability improves because the splitter can trust that `##` headers and blank lines are meaningful structural boundaries, not HTML artifacts.

**Skip condition:** If input is ≤ 300 tokens and contains no URL, skip normalization — the content is already human-readable and the LLM call is not worth it.

### Short inputs

If the full normalized content is under the child chunk size threshold (~128 tokens): one parent (summary = full text) and one child (same text). No separate summarization call needed.

---

## Chunking — Parent (Summary) / Child (Chunks)

```
Document
  │
  ├── Parent: Summary
  │     One per document (or per major section for long documents)
  │     Embedded → stored in parent_summaries collection
  │     Used as document-level retrieval signal
  │
  └── Children: Chunks
        ~128 tokens each, boundary-aligned
        Embedded → stored in child_chunks collection
        Used as passage-level retrieval signal
```

Both levels are embedded. Retrieval queries both simultaneously via hybrid search. There is no "injection-only" level — the agent decides what to use after seeing the full iteration history.

### Splitting rules for child chunks (applied in order of priority)

1. **Markdown headers** (`##`, `###`) — hard boundary, never split across a section
2. **Blank lines** (paragraph breaks) — preferred split point
3. **Sentence boundaries** (`.`, `?`, `!` followed by space + capital letter)
4. **Token limit fallback** — split mid-sentence only if a paragraph exceeds the child size limit

### Per content type

| Type | Child split strategy |
|---|---|
| `url` | Paragraph-aware, then sentence |
| `text` | Paragraph-aware, then sentence |
| `code` | Never split mid-block. Split only at top-level function/class boundaries. Fallback: token limit for unsupported languages. |

### Size parameters

| Parameter | Value |
|---|---|
| Parent summary length | ~200–300 tokens (LLM-generated) |
| Child chunk size | ~128 tokens |

---

## Embedding & Sparse Encoding

### Dense embeddings
- **Model:** `text-embedding-3-small` (OpenAI-compatible, via OpenRouter)
- **Dimensions:** 1536
- Both parent summaries and child chunks are embedded

### Sparse vectors (BM25)
- **Library:** `rank_bm25` (local, no API call)
- A corpus-level BM25 index is maintained per collection
- On each `kb_add`, the corpus is updated and sparse vectors are recomputed for the new document
- Sparse vectors are stored alongside dense vectors in Qdrant

### Error policy
If the dense embedding API call fails, raise immediately. Nothing is stored. No fallback, no partial state, no retry queue. Sparse encoding is local and cannot fail.

---

## Storage — Qdrant (local embedded)

Single store for both vector and keyword search. No server process required.

- **Library:** `qdrant-client`
- **Persistence path:** `~/.nanobot/kb/qdrant/`
- **Two collections:** `parent_summaries`, `child_chunks`
- Each collection holds both dense and sparse vectors per point

### Collection config

```python
from qdrant_client.models import VectorParams, Distance, SparseVectorParams

client.create_collection(
    collection_name="parent_summaries",
    vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()}
)

client.create_collection(
    collection_name="child_chunks",
    vectors_config={"dense": VectorParams(size=1536, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()}
)
```

### `parent_summaries` point payload

```jsonc
{
  "id": "doc-uuid",
  "vectors": {
    "dense": [0.1, ...],
    "sparse": { "indices": [...], "values": [...] }
  },
  "payload": {
    "document_id": "doc-uuid",
    "title": "<inferred or URL hostname>",
    "source": "https://..." | "user",
    "type": "url" | "text" | "code",
    "tags": [],
    "added_at": "2026-04-04T12:00:00Z",
    "fetched_at": "2026-04-04T12:00:00Z",  // URL only
    "text": "<summary text>"
  }
}
```

### `child_chunks` point payload

```jsonc
{
  "id": "chunk-uuid",
  "vectors": {
    "dense": [0.1, ...],
    "sparse": { "indices": [...], "values": [...] }
  },
  "payload": {
    "document_id": "doc-uuid",
    "chunk_index": 0,
    "source": "https://..." | "user",
    "type": "url" | "text" | "code",
    "added_at": "2026-04-04T12:00:00Z",
    "text": "<chunk text>"
  }
}
```

Deletion cascades: `kb_delete(document_id)` filters and deletes all points with matching `document_id` from both collections in a single Qdrant filter call.

---

## Retrieval — Hybrid Iterative Search

### Hybrid search + RRF fusion (single iteration)

For a given query string:

1. Compute dense embedding via `text-embedding-3-small`
2. Compute sparse vector via `rank_bm25`
3. Issue a single hybrid query per collection using Qdrant's native `prefetch` + RRF fusion:

```python
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector

results = client.query_points(
    collection_name="child_chunks",
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=top_k * 3),
        Prefetch(query=SparseVector(indices=..., values=...), using="sparse", limit=top_k * 3),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k
)
```

4. Run the same query against `parent_summaries`
5. Merge both result sets, re-rank by RRF score, return top-K items with their payload

### Iterative loop (max 3 iterations)

```
iteration = 1
accumulated_results = []

while iteration ≤ 3:
    results = hybrid_search(current_query)
    accumulated_results += results

    convergence = llm_check_convergence(results)
    //  LLM inspects metadata: document_id, title, type, source
    //  Converged = most results cluster around the same document_id(s)

    if convergence.converged:
        break
    else:
        current_query = llm_reformulate(original_query, accumulated_results)
        iteration += 1

final_set = deduplicate(accumulated_results)  // by chunk_id / document_id
```

### Convergence check

Convergence is determined by a metadata heuristic — no LLM call:

```python
def is_converged(results, threshold=0.6):
    doc_ids = [r.payload["document_id"] for r in results]
    top_count = Counter(doc_ids).most_common(1)[0][1]
    return top_count / len(doc_ids) >= threshold
```

If ≥ 60% of results share the same `document_id` → converged, stop. If scattered → LLM reformulates and continues.

A high-confidence early exit is applied before the convergence check: if the top result score ≥ 0.92, inject immediately without iterating.

### Query reformulation

The LLM uses the original query + all accumulated results so far as signal to produce a more targeted query for the next iteration. The reformulated query may focus on a specific document, refine terminology, or shift angle based on what was found.

### Parameters

| Parameter | Default |
|---|---|
| Top-K (per collection per search) | 5 |
| Candidate multiplier | ×3 |
| Similarity threshold (vector) | 0.75 |
| RRF constant k | 60 |
| Max iterations | 3 |

---

## Context Injection

After the iterative loop completes, the agent receives the full iteration history plus the deduplicated final set. Structure:

```
[KB RETRIEVAL — N iteration(s)]

Iteration 1 | query: "<original query>"
  Retrieved: <doc title> (document_id: abc, score: 0.84) — converged: no
  Signal: results scattered across 3 documents

Iteration 2 | query: "<reformulated query>"
  Retrieved: <doc title> (document_id: abc, score: 0.91) — converged: yes

[RETRIEVED KNOWLEDGE — deduplicated]

Source: https://example.com | "Page Title" | fetched 2026-04-04
---
<summary or chunk text>

Source: user note | "Note Title" | added 2026-04-04
---
<summary or chunk text>

[END KB RETRIEVAL]
```

### What gets injected

For each document in the final deduplicated set:
- If one or more child chunks scored above threshold → inject those chunks (precise passage)
- If no child chunk scored above threshold for a document, but its section summary did → inject the summary instead

This avoids injecting both a summary and its chunks (redundant), while ensuring the agent always gets the most specific content available.

### Parameters

- **Max injected tokens:** 4000 (highest-scoring first if truncation needed)
- **Deduplication:** by `chunk_id` for chunks, by `document_id` for summaries — no duplicate content injected across iterations
- **Injection skipped** if nothing meets the similarity threshold after all iterations

The agent has the full trace — queries used, convergence decisions, and all retrieved content — and reasons over everything to form its final response.

---

## Tool Interface

### `kb_add`

Add a URL, text note, or code snippet to the knowledge base.

```
kb_add(input: string, tags?: string[]) → {
    document_id: string,
    title: string,
    type: "url" | "text" | "code",
    parent_stored: 1,
    child_chunks_stored: number
}
```

- Runs smart detect on `input`
- Raises on embedding failure — nothing stored

### `kb_search`

Manually query the knowledge base (explicit recall or debugging). Runs the full hybrid iterative search.

```
kb_search(query: string, k?: number) → RetrievalResult[]
```

### `kb_delete`

Remove a document (summary + all its chunks) by document ID. Cascades across both Qdrant collections via payload filter.

```
kb_delete(document_id: string) → { deleted_summaries: number, deleted_chunks: number }
```

### `kb_list`

List all stored documents with their metadata.

```
kb_list() → { document_id, title, source, type, tags, added_at }[]
```

---

## Automatic vs. Manual Retrieval

- **Automatic (default):** Every user message triggers the hybrid iterative retrieval. Retrieved content is injected silently if above threshold.
- **Manual override:** User can say "check the knowledge base for X" to trigger an explicit `kb_search` call.
- **Opt-out:** User can say "ignore the knowledge base" to suppress injection for that turn.
- **Latency note:** Auto-retrieval adds 1–3 embedding + search round-trips per turn. Revisit if latency becomes an issue; a fast first-pass heuristic (skip on very short messages) may be added.

---

## File Layout

```
~/.nanobot/
└── kb/
    ├── qdrant/           # Qdrant local persistence directory
    └── config.json       # tunable parameters
```

---

## Configuration (`config.json`)

```jsonc
{
  "child_chunk_size": 128,
  "top_k": 5,
  "candidate_multiplier": 3,
  "similarity_threshold": 0.75,
  "rrf_k": 60,
  "max_iterations": 3,
  "max_injected_tokens": 4000,
  "summarize_urls": true,
  "normalization_skip_threshold": 300,
  "embedding_model": "text-embedding-3-small"
}
```

---

## Open Questions

| # | Question | Status |
|---|---|---|
| 1 | Re-fetching a URL: update (replace all chunks for that document_id) or append? | Leaning toward update — delete by document_id, re-ingest |
| 2 | PDF / local file path inputs? | Deferred to v2 |
| 3 | Should auto-retrieval be skipped for short/trivial messages (greetings, yes/no)? | Default skip under 5 tokens; revisit |

---

## Implementation Phases

### Phase 1 — Core storage
- [ ] Set up Qdrant (local embedded) with `parent_summaries` and `child_chunks` collections (dense + sparse vectors)
- [ ] Implement structure-aware chunker (summary + child split)
- [ ] LLM summarization call for parent generation
- [ ] Wire up dense embedding API (`text-embedding-3-small`), raise on error
- [ ] Wire up sparse encoding (`rank_bm25`), maintain per-collection corpus
- [ ] Implement hybrid search using Qdrant native prefetch + RRF fusion
- [ ] Expose `kb_add`, `kb_search`, `kb_delete`, `kb_list` tools

### Phase 2 — Ingest polish
- [ ] Smart detect (URL vs. text vs. code)
- [ ] RAG Readiness normalizer (LLM → clean structured Markdown)
- [ ] Skip condition: normalization bypassed for inputs ≤ 300 tokens with no URL
- [ ] URL fetch → normalize → summarize + chunk pipeline
- [ ] Tag support and metadata filtering in `kb_search`

### Phase 3 — Iterative retrieval
- [ ] Metadata-based convergence heuristic (document_id clustering, no LLM)
- [ ] High-confidence early exit (score ≥ 0.92)
- [ ] LLM query reformulation on non-convergence
- [ ] Iteration loop (max 3), accumulation, deduplication

### Phase 4 — Auto-retrieval & injection
- [ ] Hook retrieval into every conversation turn (single-pass for auto, full loop for explicit kb_search)
- [ ] Injection rule: chunks preferred, summary fallback per document
- [ ] Structured iteration history + deduplicated content injection
- [ ] Token budget enforcement (highest-scoring first)
- [ ] Short-message skip heuristic

### Phase 5 — Maintenance
- [ ] `kb_delete` cascades across both Qdrant collections via payload filter
- [ ] `kb_list` with filtering by type/tags
- [ ] Config hot-reload
- [ ] Qdrant collection stats / health check command
