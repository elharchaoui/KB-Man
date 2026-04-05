from __future__ import annotations

import tiktoken
from openai import AsyncOpenAI

from kb.config import Config
from kb.ingest.detector import InputType

_enc = tiktoken.get_encoding("cl100k_base")

_PROMPT = """\
Convert the following content into a clean, well-structured Markdown document.

Rules:
- Use ## for major sections, ### for subsections
- Separate paragraphs with blank lines
- Wrap all code in fenced blocks with language hints (```python, ```bash, etc.)
- Preserve all technical details, facts, names, and numbers exactly — do not summarize
- Remove navigation, ads, footers, sidebars, and any non-content elements
- For code inputs: add a ## header per top-level function or class, normalize style

Content:
{content}"""


def _token_count(text: str) -> int:
    return len(_enc.encode(text))


def should_normalize(text: str, input_type: InputType, config: Config) -> bool:
    if input_type == "url":
        return True
    if input_type == "code":
        return True
    return _token_count(text) > config.normalization_skip_threshold


async def normalize(
    text: str,
    input_type: InputType,
    client: AsyncOpenAI,
    model: str,
) -> str:
    prompt = _PROMPT.format(content=text)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
