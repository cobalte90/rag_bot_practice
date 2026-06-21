"""FAISS-backed vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(slots=True)
class VectorHit:
    """A vector-search hit."""

    index_id: int
    score: float


class FaissVectorStore:
    """A thin wrapper around a cosine-similarity FAISS index."""

    def __init__(self, index: faiss.Index, dimension: int) -> None:
        self.index = index
        self.dimension = dimension

    @classmethod
    def build(cls, embeddings: np.ndarray) -> "FaissVectorStore":
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        dimension = int(embeddings.shape[1])
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))
        return cls(index=index, dimension=dimension)

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        if not path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {path}")
        raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        index = faiss.deserialize_index(raw)
        return cls(index=index, dimension=index.d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = faiss.serialize_index(self.index)
        path.write_bytes(serialized.tobytes())

    def search(self, query_embedding: np.ndarray, top_n: int) -> list[VectorHit]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores, ids = self.index.search(query_embedding.astype(np.float32), top_n)
        results: list[VectorHit] = []
        for index_id, score in zip(ids[0], scores[0]):
            if index_id < 0:
                continue
            results.append(VectorHit(index_id=int(index_id), score=float(score)))
        return results
