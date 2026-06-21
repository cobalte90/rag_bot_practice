"""Quick retrieval smoke script for reports and debugging."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_settings
from src.retriever import HybridRetriever, IndexNotFoundError
from src.utils import setup_logging


TEST_QUESTIONS = [
    "Что такое индекс в PostgreSQL?",
    "Как создать таблицу?",
    "Как работает EXPLAIN?",
    "Чем DELETE отличается от TRUNCATE?",
    "Что такое транзакция?",
    "Как сделать backup базы?",
    "Что такое VACUUM?",
    "Как создать внешний ключ?",
    "Как посмотреть план выполнения запроса?",
    "Какие типы данных есть в PostgreSQL?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test hybrid retrieval without Telegram or Mistral API.")
    parser.add_argument("--debug", action="store_true", help="Print raw score diagnostics.")
    parser.add_argument("--question", action="append", help="Custom question. Can be used multiple times.")
    args = parser.parse_args()

    settings = load_settings()
    setup_logging(settings.log_level)

    try:
        retriever = HybridRetriever(settings=settings)
    except IndexNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    questions = args.question or TEST_QUESTIONS
    for question in questions:
        response = retriever.search(question, top_k=3)
        print("=" * 100)
        print(f"Question: {question}")
        print(f"Insufficient context: {response.insufficient_context}")
        print(f"Max score: {response.max_score:.4f}")
        print("-" * 100)
        for rank, chunk in enumerate(response.chunks, start=1):
            print(f"[{rank}] score={chunk.score:.4f} vector={chunk.vector_score:.4f} keyword={chunk.keyword_score:.4f}")
            print(f"    source={chunk.source_file} | pages={chunk.page_start}-{chunk.page_end} | chunk_id={chunk.chunk_id}")
            excerpt = chunk.text[:500].replace("\n", " ")
            print(f"    text={excerpt}")
        if args.debug:
            print("-" * 100)
            print("Debug scores:")
            print(response.debug)
        print()


if __name__ == "__main__":
    main()