"""Ensure the retrieval index exists before starting the bot."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.retriever import HybridRetriever
from src.utils import setup_logging
from scripts.ingest import build_index


def main() -> None:
    settings = load_settings()
    setup_logging(settings.log_level)
    settings.ensure_runtime_directories()

    if HybridRetriever.index_exists(settings.index_dir):
        print("Index already exists")
        return

    print("Index not found. Building index...")
    build_index(settings)


if __name__ == "__main__":
    main()
