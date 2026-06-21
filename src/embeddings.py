"""Sentence-transformer based embedding utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


class EmbeddingService:
    """Wrapper around sentence-transformers for document and query embeddings."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        LOGGER.info("Loading embedding model: %s", model_name)

        cached_model_path = self._find_cached_model_path()
        if cached_model_path is not None:
            LOGGER.info("Loading embedding model from local cache: %s", cached_model_path)
            self.model = SentenceTransformer(str(cached_model_path), local_files_only=True)
        else:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load embedding model. Ensure internet access for the first download "
                    "or pre-populate the Hugging Face cache."
                ) from exc
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode_documents(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode document passages with the appropriate prefix."""

        prepared = [self._prepare_document_text(text) for text in texts]
        return self._encode(prepared, batch_size=batch_size, show_progress=show_progress)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query with the appropriate prefix."""

        prepared = [self._prepare_query_text(text)]
        return self._encode(prepared, batch_size=1, show_progress=False)[0]

    def _encode(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        batches: list[np.ndarray] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=total_batches, desc="Embedding batches")

        for start in iterator:
            batch = texts[start : start + batch_size]
            encoded = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            batches.append(encoded.astype(np.float32))

        return np.vstack(batches)

    def _prepare_document_text(self, text: str) -> str:
        if self._requires_e5_prefix():
            return f"passage: {text}"
        return text

    def _prepare_query_text(self, text: str) -> str:
        if self._requires_e5_prefix():
            return f"query: {text}"
        return text

    def _requires_e5_prefix(self) -> bool:
        return "e5" in self.model_name.lower()

    def _find_cached_model_path(self) -> Path | None:
        cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub_root = cache_root / "hub"
        repo_dir = hub_root / f"models--{self.model_name.replace('/', '--')}"
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            return None

        snapshot_dirs = sorted(
            [path for path in snapshots_dir.iterdir() if path.is_dir()],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return snapshot_dirs[0] if snapshot_dirs else None
