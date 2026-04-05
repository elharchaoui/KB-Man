from __future__ import annotations

from openai import AsyncOpenAI

from kb.ingest.detector import InputType

_PROMPTS: dict[InputType, str] = {
    "url": (
        "Summarize the following document in 5–8 sentences. "
        "Preserve key facts, names, technical details, and numbers exactly. "
        "Be precise and information-dense.\n\nDocument:\n{content}"
    ),
    "text": (
        "Summarize the following text in 3–6 sentences. "
        "Preserve key facts and details exactly.\n\nText:\n{content}"
    ),
    "code": (
        "Describe what the following code does in 2–4 sentences. "
        "Mention the language, main functions or classes, and purpose.\n\nCode:\n{content}"
    ),
}


async def summarize(
    text: str,
    input_type: InputType,
    client: AsyncOpenAI,
    model: str,
) -> str:
    prompt = _PROMPTS[input_type].format(content=text)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
