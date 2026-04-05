from __future__ import annotations

import os
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

_JINA_BASE = "https://r.jina.ai/"
_TIMEOUT = 30


class FetchError(Exception):
    pass


async def fetch_url(url: str) -> tuple[str, str, datetime]:
    """
    Fetch a URL via Jina Reader and return (markdown_text, title, fetched_at).

    Jina handles JavaScript-heavy sites, anti-bot pages, and PDFs.
    Set JINA_API_KEY in the environment for higher rate limits (optional).
    """
    headers = {
        "Accept": "application/json",
        "X-Respond-With": "markdown",
        "X-No-Cache": "true",
    }
    api_key = os.environ.get("JINA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=_TIMEOUT) as client:
            response = await client.get(_JINA_BASE + url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as e:
        raise FetchError(f"Jina Reader failed for {url}: {e}") from e
    except Exception as e:
        raise FetchError(f"Unexpected error fetching {url}: {e}") from e

    data = payload.get("data", payload)
    title = data.get("title", "").strip() or urlparse(url).netloc
    content = data.get("content", data.get("text", "")).strip()

    if not content:
        raise FetchError(f"Jina returned empty content for {url}")

    return content, title, datetime.now(timezone.utc)
