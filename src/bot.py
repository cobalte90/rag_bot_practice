"""Telegram bot entrypoint."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import Message

from src.config import ConfigError, Settings
from src.llm_client import LLMClient, LLMError
from src.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retriever import HybridRetriever
from src.utils import split_for_telegram


LOGGER = logging.getLogger(__name__)


START_TEXT = (
    "Я PostgresHelperBot. Я отвечаю на вопросы по документации PostgreSQL 18. "
    "Документация на английском, но ответы я даю на русском и показываю короткую выдержку из источника."
)

HELP_TEXT = """Примеры вопросов:

* Что такое индекс в PostgreSQL?
* Как работает `EXPLAIN`?
* Чем `DELETE` отличается от `TRUNCATE`?
* Как создать внешний ключ?
* Что такое транзакция?
* Как сделать backup?"""


class TelegramBotApp:
    """Encapsulates bot lifecycle and command handlers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.retriever: HybridRetriever | None = None
        self.retriever_error: str | None = None
        self.llm_client: LLMClient | None = None

        self.router = Router()
        self.router.message.register(self.handle_start, Command("start"))
        self.router.message.register(self.handle_help, Command("help"))
        self.router.message.register(self.handle_sources, Command("sources"))
        self.router.message.register(self.handle_status, Command("status"))
        self.router.message.register(self.handle_question, F.text)

    async def run(self) -> None:
        token = self.settings.require_telegram_token()
        self._initialize_runtime()

        bot = Bot(token=token)
        dispatcher = Dispatcher()
        dispatcher.include_router(self.router)

        try:
            await dispatcher.start_polling(bot)
        finally:
            await bot.session.close()

    def _initialize_runtime(self) -> None:
        try:
            self.retriever = HybridRetriever(settings=self.settings)
            self.retriever_error = None
        except Exception as exc:
            self.retriever = None
            self.retriever_error = str(exc)
            LOGGER.warning("Retriever is unavailable at startup: %s", exc)

        if self.settings.mistral_api_key:
            self.llm_client = LLMClient(settings=self.settings)
        else:
            self.llm_client = None

    async def handle_start(self, message: Message) -> None:
        await message.answer(START_TEXT)

    async def handle_help(self, message: Message) -> None:
        await message.answer(HELP_TEXT)

    async def handle_sources(self, message: Message) -> None:
        if not self.settings.index_files["meta"].exists():
            await message.answer("Индекс пока не построен. Сначала запустите: python scripts/ingest.py")
            return

        with self.settings.index_files["meta"].open("r", encoding="utf-8") as file_obj:
            meta = json.load(file_obj)

        source_files = "\n".join(f"* {item}" for item in meta.get("source_files", [])) or "* не найдено"
        created_at = meta.get("created_at", "неизвестно")
        chunk_count = meta.get("chunk_count", 0)
        embedding_model = meta.get("embedding_model", self.settings.embedding_model)
        text = (
            "Документы в базе:\n"
            f"{source_files}\n\n"
            f"Количество чанков: {chunk_count}\n"
            f"Embedding model: {embedding_model}\n"
            f"Индекс построен: {created_at}"
        )
        await message.answer(text)

    async def handle_status(self, message: Message) -> None:
        index_exists = HybridRetriever.index_exists(self.settings.index_dir)
        retriever_loaded = self.retriever is not None
        text = "\n".join(
            [
                f"Индекс: {'есть' if index_exists else 'нет'}",
                f"Retriever: {'загружен' if retriever_loaded else 'не загружен'}",
                f"Mistral API key: {'есть' if bool(self.settings.mistral_api_key) else 'нет'}",
                f"Telegram token: {'есть' if bool(self.settings.telegram_bot_token) else 'нет'}",
            ]
        )
        if self.retriever_error:
            text += f"\nОшибка retriever: {self.retriever_error}"
        await message.answer(text)

    async def handle_question(self, message: Message) -> None:
        question = (message.text or "").strip()
        if not question:
            await message.answer("Вопрос пустой. Напишите, что именно вы хотите найти в документации PostgreSQL.")
            return

        if not self.retriever:
            await message.answer("Индекс не найден. Сначала запустите: python scripts/ingest.py")
            return

        placeholder = await message.answer("Ищу ответ в документации PostgreSQL...")
        retrieval_started = time.perf_counter()
        try:
            retrieval_response = self.retriever.search(question)
        except ValueError as exc:
            await placeholder.edit_text(str(exc))
            return

        retrieval_elapsed = time.perf_counter() - retrieval_started
        LOGGER.info(
            "User query=%r | chunks=%s | max_score=%.4f | retrieval_time=%.3fs",
            question,
            len(retrieval_response.chunks),
            retrieval_response.max_score,
            retrieval_elapsed,
        )

        if retrieval_response.insufficient_context:
            await placeholder.edit_text(
                "В найденных фрагментах документации недостаточно информации для ответа на этот вопрос."
            )
            return

        if not self.llm_client:
            await placeholder.edit_text(
                "Не удалось получить ответ от LLM. Попробуйте позже или проверьте MISTRAL_API_KEY."
            )
            return

        llm_started = time.perf_counter()
        try:
            answer = await asyncio.to_thread(
                self.llm_client.generate_answer,
                SYSTEM_PROMPT,
                build_user_prompt(question, retrieval_response.chunks),
            )
        except (ConfigError, LLMError) as exc:
            LOGGER.error("LLM request failed for query=%r: %s", question, exc)
            await placeholder.edit_text(
                "Не удалось получить ответ от LLM. Попробуйте позже или проверьте MISTRAL_API_KEY."
            )
            return

        llm_elapsed = time.perf_counter() - llm_started
        LOGGER.info("LLM completed for query=%r in %.3fs", question, llm_elapsed)

        parts = split_for_telegram(answer)
        await placeholder.edit_text(parts[0])
        for part in parts[1:]:
            await message.answer(part)


async def run_bot(settings: Settings) -> None:
    """Run the Telegram bot."""

    app = TelegramBotApp(settings)
    await app.run()