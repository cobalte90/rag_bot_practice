"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class ConfigError(RuntimeError):
    """Raised when application configuration is invalid."""


def _resolve_path(raw_value: str | None, default: str) -> Path:
    candidate = Path(raw_value or default)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer.") from exc


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a float.") from exc


@dataclass(slots=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    telegram_bot_token: str | None
    mistral_api_key: str | None
    mistral_model: str
    mistral_base_url: str
    raw_data_dir: Path
    processed_data_dir: Path
    index_dir: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    vector_weight: float
    keyword_weight: float
    min_relevance_score: float
    embedding_batch_size: int
    llm_timeout_seconds: int
    llm_max_retries: int
    llm_temperature: float
    llm_max_tokens: int
    log_level: str
    env_path: Path
    env_file_exists: bool

    @classmethod
    def load(cls, env_path: Path | None = None) -> "Settings":
        """Load settings from `.env` and process environment."""

        chosen_env_path = env_path or DEFAULT_ENV_PATH
        if chosen_env_path.exists():
            load_dotenv(chosen_env_path, override=True)

        settings = cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
            mistral_api_key=os.getenv("MISTRAL_API_KEY") or None,
            mistral_model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            mistral_base_url=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"),
            raw_data_dir=_resolve_path(os.getenv("RAW_DATA_DIR"), "data/raw"),
            processed_data_dir=_resolve_path(os.getenv("PROCESSED_DATA_DIR"), "data/processed"),
            index_dir=_resolve_path(os.getenv("INDEX_DIR"), "data/index"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            chunk_size=_get_int("CHUNK_SIZE", 1200),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 200),
            top_k=_get_int("TOP_K", 5),
            vector_weight=_get_float("VECTOR_WEIGHT", 0.65),
            keyword_weight=_get_float("KEYWORD_WEIGHT", 0.35),
            min_relevance_score=_get_float("MIN_RELEVANCE_SCORE", 0.15),
            embedding_batch_size=_get_int("EMBEDDING_BATCH_SIZE", 32),
            llm_timeout_seconds=_get_int("LLM_TIMEOUT_SECONDS", 45),
            llm_max_retries=_get_int("LLM_MAX_RETRIES", 3),
            llm_temperature=_get_float("LLM_TEMPERATURE", 0.1),
            llm_max_tokens=_get_int("LLM_MAX_TOKENS", 500),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            env_path=chosen_env_path,
            env_file_exists=chosen_env_path.exists(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Validate scalar settings that should always be sane."""

        if self.chunk_size <= 0:
            raise ConfigError("CHUNK_SIZE must be positive.")
        if self.chunk_overlap < 0:
            raise ConfigError("CHUNK_OVERLAP cannot be negative.")
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.top_k <= 0:
            raise ConfigError("TOP_K must be positive.")
        if self.embedding_batch_size <= 0:
            raise ConfigError("EMBEDDING_BATCH_SIZE must be positive.")
        if not 0 <= self.vector_weight <= 1:
            raise ConfigError("VECTOR_WEIGHT must be between 0 and 1.")
        if not 0 <= self.keyword_weight <= 1:
            raise ConfigError("KEYWORD_WEIGHT must be between 0 and 1.")
        if abs((self.vector_weight + self.keyword_weight) - 1.0) > 1e-6:
            raise ConfigError("VECTOR_WEIGHT and KEYWORD_WEIGHT must sum to 1.0.")
        if self.min_relevance_score < 0:
            raise ConfigError("MIN_RELEVANCE_SCORE cannot be negative.")
        if self.llm_timeout_seconds <= 0:
            raise ConfigError("LLM_TIMEOUT_SECONDS must be positive.")
        if self.llm_max_retries < 0:
            raise ConfigError("LLM_MAX_RETRIES cannot be negative.")

    def ensure_runtime_directories(self) -> None:
        """Create runtime directories if they do not exist."""

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def require_telegram_token(self) -> str:
        """Return a Telegram token or raise a clear error."""

        if self.telegram_bot_token:
            return self.telegram_bot_token
        raise ConfigError(
            "TELEGRAM_BOT_TOKEN is not configured. Create `.env` from `.env.example` "
            "and set the Telegram bot token."
        )

    def require_mistral_api_key(self) -> str:
        """Return a Mistral API key or raise a clear error."""

        if self.mistral_api_key:
            return self.mistral_api_key
        raise ConfigError(
            "MISTRAL_API_KEY is not configured. Create `.env` from `.env.example` "
            "and set the Mistral API key."
        )

    @property
    def index_files(self) -> dict[str, Path]:
        """Return paths of essential index artifacts."""

        return {
            "faiss": self.index_dir / "faiss.index",
            "chunks": self.index_dir / "chunks.jsonl",
            "bm25": self.index_dir / "bm25.pkl",
            "meta": self.index_dir / "index_meta.json",
        }


def load_settings(env_path: Path | None = None) -> Settings:
    """Convenience helper to load settings."""

    return Settings.load(env_path=env_path)

