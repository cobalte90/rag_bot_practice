import numpy as np
import pytest
import faiss
from pathlib import Path

from src.chunker import DocumentChunk
from src.config import Settings
from src.keyword_search import BM25Index
from src.retriever import HybridRetriever, IndexNotFoundError
from src.utils import write_json, write_jsonl


class FakeEmbeddingService:
    def encode_query(self, question: str) -> np.ndarray:
        if "индекс" in question.lower():
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        telegram_bot_token=None,
        mistral_api_key=None,
        mistral_model="mistral-small-latest",
        mistral_base_url="https://api.mistral.ai/v1",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        index_dir=tmp_path / "index",
        embedding_model="intfloat/multilingual-e5-small",
        chunk_size=1200,
        chunk_overlap=200,
        top_k=5,
        vector_weight=0.65,
        keyword_weight=0.35,
        min_relevance_score=0.15,
        embedding_batch_size=32,
        llm_timeout_seconds=45,
        llm_max_retries=3,
        llm_temperature=0.1,
        llm_max_tokens=500,
        log_level="INFO",
        env_path=tmp_path / ".env",
        env_file_exists=False,
    )


def test_retriever_raises_clear_error_when_index_is_missing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    with pytest.raises(IndexNotFoundError, match="Run `python scripts/ingest.py` first"):
        HybridRetriever(settings=settings, embedding_service=FakeEmbeddingService())


def test_retriever_returns_result_from_small_test_index(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        DocumentChunk(
            chunk_id="chunk_0",
            source_file="postgresql-18-US.pdf",
            page_start=10,
            page_end=10,
            chunk_index=0,
            text="Indexes allow the database server to find rows much faster.",
            char_start=0,
            char_end=63,
        ),
        DocumentChunk(
            chunk_id="chunk_1",
            source_file="postgresql-18-US.pdf",
            page_start=20,
            page_end=20,
            chunk_index=1,
            text="Transactions group multiple steps into an all-or-nothing operation.",
            char_start=64,
            char_end=133,
        ),
    ]
    write_jsonl(settings.index_files["chunks"], [chunk.to_dict() for chunk in chunks])

    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = faiss.IndexFlatIP(2)
    index.add(embeddings)
    settings.index_files["faiss"].write_bytes(faiss.serialize_index(index).tobytes())

    bm25 = BM25Index.build([chunk.text for chunk in chunks])
    bm25.save(settings.index_files["bm25"])

    write_json(
        settings.index_files["meta"],
        {
            "source_files": ["postgresql-18-US.pdf"],
            "chunk_count": 2,
            "embedding_model": settings.embedding_model,
        },
    )

    retriever = HybridRetriever(settings=settings, embedding_service=FakeEmbeddingService())
    response = retriever.search("Что такое индекс?", top_k=1)

    assert not response.insufficient_context
    assert response.chunks
    assert response.chunks[0].chunk_id == "chunk_0"
