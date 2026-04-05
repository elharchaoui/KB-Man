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


def chunk(text: str, input_type: InputType, config: Config) -> list[str]:
    if _token_count(text) <= config.child_chunk_size:
        return [text]
    if input_type == "code":
        return _chunk_code(text, config.child_chunk_size)
    return _chunk_text(text, config.child_chunk_size)


def _chunk_text(text: str, max_tokens: int) -> list[str]:
    # Split on markdown headers first (hard boundary)
    sections = _split_on_headers(text)
    chunks: list[str] = []
    for section in sections:
        chunks.extend(_split_section(section, max_tokens))
    return [c.strip() for c in chunks if c.strip()]


def _split_on_headers(text: str) -> list[str]:
    parts = _HEADER_RE.split(text)
    headers = _HEADER_RE.findall(text)
    sections: list[str] = []
    for i, part in enumerate(parts):
        header = headers[i - 1] if i > 0 else ""
        sections.append((header + "\n" + part).strip() if header else part.strip())
    return [s for s in sections if s]


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


def _chunk_code(text: str, max_tokens: int) -> list[str]:
    if _token_count(text) <= max_tokens:
        return [text]

    # Protect fenced blocks from splitting
    boundaries = list(_CODE_BOUNDARY_RE.finditer(text))
    if not boundaries:
        return _hard_split(text, max_tokens)

    sections: list[str] = []
    for i, match in enumerate(boundaries):
        start = match.start()
        end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
        sections.append(text[start:end].strip())

    chunks: list[str] = []
    buffer = ""
    for section in sections:
        candidate = (buffer + "\n\n" + section).strip() if buffer else section
        if _token_count(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            # section itself too long — hard split as last resort
            if _token_count(section) <= max_tokens:
                buffer = section
            else:
                chunks.extend(_hard_split(section, max_tokens))
                buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks
