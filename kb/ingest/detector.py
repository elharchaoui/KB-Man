from __future__ import annotations

import re
from typing import Literal

InputType = Literal["url", "code", "text"]

_URL_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)

_CODE_SIGNALS = [
    r"^```",                        # fenced block
    r"def \w+\s*\(",                # Python function
    r"(class|function|const|let|var|import|from|#include)\s+\w+",
    r"[{};]\s*$",                   # C-style braces
]
_CODE_RE = re.compile("|".join(_CODE_SIGNALS), re.MULTILINE)


def detect(input: str) -> InputType:
    stripped = input.strip()
    if _URL_RE.match(stripped):
        return "url"
    if _CODE_RE.search(stripped):
        return "code"
    return "text"
