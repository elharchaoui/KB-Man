from __future__ import annotations

from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from kb.ingest.detector import InputType

_PROMPT = """\
Analyze the content below and return 3–7 concise tags.

Rules:
- Lowercase kebab-case only (e.g. "mixture-of-experts", "speculative-decoding")
- Prefer specific technical terms over generic ones
- No duplicates, no punctuation other than hyphens

Content (first 2000 chars):
{content}"""


class _TagList(BaseModel):
    tags: list[str] = Field(min_length=1, max_length=10)


async def suggest_tags(
    text: str,
    input_type: InputType,
    client: AsyncOpenAI,
    model: str,
) -> list[str]:
    """Return LLM-suggested tags for the normalised content.

    Uses structured output — raises if the model returns an invalid response.
    """
    prompt = _PROMPT.format(content=text[:2000])
    response = await client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=_TagList,
        temperature=0,
        max_tokens=120,
    )
    return [t.lower().strip() for t in response.choices[0].message.parsed.tags]
