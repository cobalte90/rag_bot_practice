"""Hybrid retriever combining FAISS and BM25."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

from src.chunker import DocumentChunk
from src.config import Settings
from src.embeddings import EmbeddingService
from src.keyword_search import BM25Index, KeywordHit
from src.utils import read_jsonl
from src.vector_store import FaissVectorStore, VectorHit


LOGGER = logging.getLogger(__name__)


class IndexNotFoundError(RuntimeError):
    """Raised when retrieval artifacts are missing."""


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by hybrid retrieval."""

    chunk_id: str
    text: str
    source_file: str
    page_start: int
    page_end: int
    score: float
    vector_score: float = 0.0
    keyword_score: float = 0.0


@dataclass(slots=True)
class RetrievalResponse:
    """Result of a hybrid retrieval request."""

    question: str
    chunks: list[RetrievedChunk]
    insufficient_context: bool
    max_score: float
    debug: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Hybrid retriever using multilingual embeddings and BM25."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService | Any | None = None,
    ) -> None:
        self.settings = settings
        self.index_files = settings.index_files
        if not self.index_exists(settings.index_dir):
            raise IndexNotFoundError(
                f"Index not found in {settings.index_dir}. "
                "Run `python scripts/ingest.py` first."
            )

        self.chunks = [
            DocumentChunk.from_dict(item)
            for item in read_jsonl(self.index_files["chunks"])
        ]
        self.vector_store = FaissVectorStore.load(self.index_files["faiss"])
        self.keyword_index = BM25Index.load(self.index_files["bm25"])
        self.embedding_service = embedding_service or EmbeddingService(settings.embedding_model)
        with self.index_files["meta"].open("r", encoding="utf-8") as file_obj:
            self.index_meta = json.load(file_obj)

    @staticmethod
    def index_exists(index_dir: Path) -> bool:
        required = {
            index_dir / "faiss.index",
            index_dir / "chunks.jsonl",
            index_dir / "bm25.pkl",
            index_dir / "index_meta.json",
        }
        return all(path.exists() for path in required)

    def search(self, question: str, top_k: int | None = None) -> RetrievalResponse:
        if not question or not question.strip():
            raise ValueError("Question is empty. Please provide a non-empty query.")

        effective_top_k = top_k or self.settings.top_k
        candidate_count = max(effective_top_k * 3, effective_top_k)

        query_embedding = self.embedding_service.encode_query(question.strip())
        vector_hits = self.vector_store.search(query_embedding, top_n=candidate_count)
        keyword_hits = self.keyword_index.search(question.strip(), top_n=candidate_count)

        normalized_vector = self._normalize_vector_hits(vector_hits)
        normalized_keyword = self._normalize_keyword_hits(keyword_hits)

        merged_scores: dict[str, RetrievedChunk] = {}
        for hit in vector_hits:
            chunk = self.chunks[hit.index_id]
            merged_scores[chunk.chunk_id] = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_file=chunk.source_file,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                score=0.0,
                vector_score=normalized_vector.get(hit.index_id, 0.0),
                keyword_score=0.0,
            )

        for hit in keyword_hits:
            chunk = self.chunks[hit.index_id]
            existing = merged_scores.get(chunk.chunk_id)
            keyword_score = normalized_keyword.get(hit.index_id, 0.0)
            if existing:
                existing.keyword_score = keyword_score
            else:
                merged_scores[chunk.chunk_id] = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_file=chunk.source_file,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    score=0.0,
                    vector_score=0.0,
                    keyword_score=keyword_score,
                )

        ranked_chunks: list[RetrievedChunk] = []
        for chunk in merged_scores.values():
            chunk.score = (
                self.settings.vector_weight * chunk.vector_score
                + self.settings.keyword_weight * chunk.keyword_score
            )
            ranked_chunks.append(chunk)

        ranked_chunks.sort(key=lambda item: item.score, reverse=True)
        top_chunks = ranked_chunks[:effective_top_k]
        max_score = top_chunks[0].score if top_chunks else 0.0
        insufficient_context = not top_chunks or max_score < self.settings.min_relevance_score

        return RetrievalResponse(
            question=question,
            chunks=top_chunks,
            insufficient_context=insufficient_context,
            max_score=max_score,
            debug={
                "vector_hits": {self.chunks[item.index_id].chunk_id: item.score for item in vector_hits},
                "keyword_hits": {self.chunks[item.index_id].chunk_id: item.score for item in keyword_hits},
            },
        )

    @staticmethod
    def _normalize_vector_hits(hits: list[VectorHit]) -> dict[int, float]:
        return _normalize_scores({hit.index_id: hit.score for hit in hits})

    @staticmethod
    def _normalize_keyword_hits(hits: list[KeywordHit]) -> dict[int, float]:
        return _normalize_scores({hit.index_id: hit.score for hit in hits})


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}

    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return {key: 1.0 for key, value in scores.items() if value > 0}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


QUERY_EXPANSIONS = {
    "индекс": ["index", "indexes", "index scan"],
    "join": ["join", "inner join", "left join"],
    "соедин": ["join", "inner join", "left join"],
    "таблиц": ["table", "create table", "relation"],
    "create table": ["create table", "table"],
    "group by": ["group by", "aggregate", "aggregation"],
    "групп": ["group by", "aggregate"],
    "транзак": ["transaction", "commit", "rollback", "atomic"],
    "explain": ["explain", "query plan", "execution plan"],
    "план выполн": ["explain", "query plan", "execution plan"],
    "внешн": ["foreign key", "references", "constraint"],
    "foreign key": ["foreign key", "references", "constraint"],
    "backup": ["backup", "base backup", "pg_basebackup", "restore"],
    "restore": ["restore", "backup", "pg_restore"],
    "vacuum": ["vacuum", "autovacuum"],
    "delete": ["delete", "truncate"],
    "truncate": ["truncate", "delete"],
    "тип": ["data type", "type system", "types"],
    "role": ["role", "privilege", "grant", "user"],
    "прав": ["privilege", "grant", "role", "user"],
    "constraint": ["constraint", "check", "unique", "primary key", "foreign key"],
}


def _expand_query(question: str) -> str:
    lowered = question.lower()
    additions: list[str] = []
    for needle, terms in QUERY_EXPANSIONS.items():
        if needle in lowered:
            additions.extend(terms)

    if not additions:
        return question

    unique_terms = []
    seen: set[str] = set()
    for term in additions:
        if term not in seen:
            unique_terms.append(term)
            seen.add(term)
    return question + " " + " ".join(unique_terms)
