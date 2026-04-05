from __future__ import annotations

from kb import KnowledgeBase


def make_tools(kb: KnowledgeBase) -> list[dict]:
    """
    Returns a list of tool definitions (OpenAI-compatible) backed by the KB instance.
    Returns a list of tool definitions (OpenAI-compatible function calling format).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "kb_add",
                "description": (
                    "Add a URL, text note, or code snippet to the knowledge base. "
                    "The content will be fetched (if URL), normalized, summarized, "
                    "and indexed for future retrieval."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The URL, text, or code snippet to store.",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for organization.",
                        },
                    },
                    "required": ["input"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "kb_search",
                "description": (
                    "Search the knowledge base with a natural language query. "
                    "Runs full hybrid iterative search (up to 3 iterations)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for.",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "kb_delete",
                "description": "Remove a document and all its chunks from the knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "The document_id returned by kb_add or kb_list.",
                        },
                    },
                    "required": ["document_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "kb_list",
                "description": "List all documents stored in the knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    ]


async def dispatch(name: str, args: dict, kb: KnowledgeBase) -> str:
    """Route a tool call to the KB and return a string result."""
    if name == "kb_add":
        result = await kb.add(args["input"], args.get("tags", []))
        return (
            f"Stored: {result.title!r} [{result.type}]\n"
            f"document_id: {result.document_id}\n"
            f"chunks: {result.child_chunks_stored}"
        )

    if name == "kb_search":
        output = await kb.search(args["query"], args.get("k"))
        from kb.injection import injector
        from kb.config import Config
        context = injector.build(output, kb._config)
        return context or "No relevant results found."

    if name == "kb_delete":
        result = await kb.delete(args["document_id"])
        return (
            f"Deleted document {result.document_id}: "
            f"{result.deleted_summaries} summary, {result.deleted_chunks} chunks."
        )

    if name == "kb_list":
        docs = kb.list()
        if not docs:
            return "Knowledge base is empty."
        lines = [
            f"- [{d.type}] {d.title!r} | id: {d.document_id} | added: {d.added_at.date()}"
            + (f" | tags: {', '.join(d.tags)}" if d.tags else "")
            for d in docs
        ]
        return "\n".join(lines)

    raise ValueError(f"Unknown tool: {name}")
