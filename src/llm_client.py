"""Mistral API client."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.config import ConfigError, Settings


LOGGER = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM client cannot obtain a valid answer."""


@dataclass(slots=True)
class LLMClient:
    """A minimal Mistral chat-completions client with retries."""

    settings: Settings

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an answer using Mistral's chat API."""

        api_key = self.settings.mistral_api_key
        if not api_key:
            raise ConfigError(
                "MISTRAL_API_KEY is missing. Add it to `.env` before asking the bot to generate answers."
            )

        payload = {
            "model": self.settings.mistral_model,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        last_error: Exception | None = None
        url = f"{self.settings.mistral_base_url.rstrip('/')}/chat/completions"
        for attempt in range(self.settings.llm_max_retries + 1):
            try:
                return self._perform_request(url=url, api_key=api_key, payload=payload)
            except LLMError as exc:
                last_error = exc
                if attempt >= self.settings.llm_max_retries:
                    break
                sleep_seconds = min(2 ** attempt, 8)
                LOGGER.warning("LLM request failed (attempt %s). Retrying in %ss: %s", attempt + 1, sleep_seconds, exc)
                time.sleep(sleep_seconds)

        raise LLMError(str(last_error) if last_error else "Mistral API request failed.")

    def _perform_request(self, url: str, api_key: str, payload: dict[str, Any]) -> str:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url=url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=self.settings.llm_timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 401:
                raise LLMError("Mistral API rejected the key (401 Unauthorized).") from exc
            if exc.code == 429:
                raise LLMError("Mistral API rate limit exceeded (429).") from exc
            raise LLMError(f"Mistral API returned HTTP {exc.code}: {raw_body}") from exc
        except urllib_error.URLError as exc:
            raise LLMError(f"Network error while calling Mistral API: {exc}") from exc
        except TimeoutError as exc:  # pragma: no cover - depends on socket layer
            raise LLMError("Timed out while waiting for Mistral API.") from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise LLMError("Mistral API returned invalid JSON.") from exc

        choices = payload.get("choices") or []
        if not choices:
            raise LLMError("Mistral API returned an empty choices list.")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not content or not str(content).strip():
            raise LLMError("Mistral API returned an empty response.")

        return str(content).strip()
