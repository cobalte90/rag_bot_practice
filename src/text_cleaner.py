"""Text cleaning helpers for technical documentation."""

from __future__ import annotations

import re


_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_MULTI_SPACES_INSIDE_LINE = re.compile(r"(?<=\S)[ \t]{2,}(?=\S)")


def clean_text(text: str) -> str:
    """Normalize whitespace while keeping technical content readable."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    normalized = normalized.replace("\u00a0", " ")

    cleaned_lines: list[str] = []
    for line in normalized.split("\n"):
        line = line.rstrip()
        line = _MULTI_SPACES_INSIDE_LINE.sub(" ", line)
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = _MULTI_BLANK_LINES.sub("\n\n", cleaned)
    return cleaned.strip()
