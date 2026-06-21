"""General-purpose helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable


def setup_logging(level: str) -> None:
    """Configure application-wide logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write JSON Lines file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read JSON Lines file into memory."""

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def split_for_telegram(text: str, max_length: int = 4096) -> list[str]:
    """Split long messages into Telegram-safe chunks."""

    if len(text) <= max_length:
        return [text]

    parts: list[str] = []
    remaining = text
    while len(remaining) > max_length:
        split_index = remaining.rfind("\n", 0, max_length)
        if split_index == -1:
            split_index = remaining.rfind(" ", 0, max_length)
        if split_index == -1:
            split_index = max_length
        parts.append(remaining[:split_index].strip())
        remaining = remaining[split_index:].strip()
    if remaining:
        parts.append(remaining)
    return parts
