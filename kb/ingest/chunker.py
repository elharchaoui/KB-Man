from __future__ import annotations

import re

import tiktoken

from kb.config import Config
from kb.ingest.detector import InputType

_enc = tiktoken.get_encoding("cl100k_base")

_HEADER_RE = re.compile(r"^#{1,3} .+", re.MULTILINE)
_FENCE_RE = re.compile(r"```[\w]*\n.*?```", re.DOTALL)
_SENTENCE_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")

# Top-level Python/JS/TS function and class boundaries
_CODE_BOUNDARY_RE = re.compile(
    r"^(?:def |class |async def |function |const \w+ = |export (?:default )?(?:function|class))",
    re.MULTILINE,
)


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def chunk(text: str, input_type: InputType, config: Config) -> list[tuple[str, str | None]]:
    """
    Return a list of (chunk_text, section_header) pairs.
    section_header is the nearest Markdown ## / ### above the chunk, or None.
    """
    if _token_count(text) <= config.child_chunk_size:
        return [(text, None)]
    if input_type == "code":
        return _chunk_code(text, config.child_chunk_size)
    return _chunk_text(text, config.child_chunk_size)


def _chunk_text(text: str, max_tokens: int) -> list[tuple[str, str | None]]:
    # Split on markdown headers first (hard boundary)
    sections = _split_on_headers(text)
    chunks: list[tuple[str, str | None]] = []
    for section_text, header in sections:
        for piece in _split_section(section_text, max_tokens):
            piece = piece.strip()
            if piece:
                chunks.append((piece, header))
    return chunks


def _split_on_headers(text: str) -> list[tuple[str, str | None]]:
    """Return (section_text, header_or_None) pairs split at Markdown headers."""
    parts = _HEADER_RE.split(text)
    headers = _HEADER_RE.findall(text)
    sections: list[tuple[str, str | None]] = []
    for i, part in enumerate(parts):
        header: str | None = headers[i - 1] if i > 0 else None
        # Keep the header line in the text for retrieval context
        body = (header + "\n" + part).strip() if header else part.strip()
        if body:
            sections.append((body, header))
    return sections


def _split_section(text: str, max_tokens: int) -> list[str]:
    if _token_count(text) <= max_tokens:
        return [text]

    # Split on blank lines (paragraph breaks)
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        candidate = (buffer + "\n\n" + para).strip() if buffer else para
        if _token_count(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            # para itself may be too long — split on sentences
            if _token_count(para) <= max_tokens:
                buffer = para
            else:
                chunks.extend(_split_by_sentences(para, max_tokens))
                buffer = ""

    if buffer:
        chunks.append(buffer)
    return chunks


def _split_by_sentences(text: str, max_tokens: int) -> list[str]:
    sentences = _SENTENCE_RE.split(text)
    chunks: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = (buffer + " " + sentence).strip() if buffer else sentence
        if _token_count(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            # sentence itself exceeds limit — hard split by token count
            if _token_count(sentence) <= max_tokens:
                buffer = sentence
            else:
                chunks.extend(_hard_split(sentence, max_tokens))
                buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def _hard_split(text: str, max_tokens: int) -> list[str]:
    tokens = _enc.encode(text)
    return [
        _enc.decode(tokens[i: i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def _chunk_code(text: str, max_tokens: int) -> list[tuple[str, str | None]]:
    if _token_count(text) <= max_tokens:
        return [(text, None)]

    boundaries = list(_CODE_BOUNDARY_RE.finditer(text))
    if not boundaries:
        return [(piece, None) for piece in _hard_split(text, max_tokens)]

    # Each section's header is its first (boundary) line, e.g. "def foo(x):"
    sections: list[tuple[str, str | None]] = []
    for i, match in enumerate(boundaries):
        start = match.start()
        end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()
        # First line is the function/class signature
        header = section_text.splitlines()[0] if section_text else None
        sections.append((section_text, header))

    chunks: list[tuple[str, str | None]] = []
    buffer = ""
    buffer_header: str | None = None
    for section_text, header in sections:
        candidate = (buffer + "\n\n" + section_text).strip() if buffer else section_text
        if _token_count(candidate) <= max_tokens:
            buffer = candidate
            if buffer_header is None:
                buffer_header = header
        else:
            if buffer:
                chunks.append((buffer, buffer_header))
            if _token_count(section_text) <= max_tokens:
                buffer = section_text
                buffer_header = header
            else:
                chunks.extend((piece, header) for piece in _hard_split(section_text, max_tokens))
                buffer = ""
                buffer_header = None
    if buffer:
        chunks.append((buffer, buffer_header))
    return chunks
