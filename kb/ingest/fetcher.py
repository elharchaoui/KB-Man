from __future__ import annotations

import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx


class FetchError(Exception):
    pass


async def fetch_url(url: str) -> tuple[str, str, datetime]:
    """
    Fetch a URL and return (raw_text, title, fetched_at).
    Strips HTML tags to get readable text.
    Raises FetchError on network or HTTP errors.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            response = await client.get(url, headers={"User-Agent": "kb/1.0"})
            response.raise_for_status()
            html = response.text
    except httpx.HTTPError as e:
        raise FetchError(f"Failed to fetch {url}: {e}") from e

    title = _extract_title(html) or urlparse(url).netloc
    text = _strip_html(html)
    return text, title, datetime.now(timezone.utc)


def _extract_title(html: str) -> str | None:
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return None


def _strip_html(html: str) -> str:
    # Remove script and style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
