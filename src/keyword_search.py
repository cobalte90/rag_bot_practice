"""BM25 keyword search index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize_text(text: str) -> list[str]:
    """Tokenize text for BM25 in a simple SQL-friendly way."""

    normalized = text.lower()
    tokens = TOKEN_RE.findall(normalized)
    bigrams = [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    return tokens + bigrams


@dataclass(slots=True)
class KeywordHit:
    """A BM25 search hit."""

    index_id: int
    score: float


class BM25Index:
    """A picklable BM25 index with its tokenized corpus."""

    def __init__(self, bm25: BM25Okapi, tokenized_corpus: list[list[str]]) -> None:
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus

    @classmethod
    def build(cls, texts: list[str]) -> "BM25Index":
        tokenized_corpus = [tokenize_text(text) for text in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(bm25=bm25, tokenized_corpus=tokenized_corpus)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")
        with path.open("rb") as file_obj:
            payload = pickle.load(file_obj)
        return cls(bm25=payload["bm25"], tokenized_corpus=payload["tokenized_corpus"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file_obj:
            pickle.dump({"bm25": self.bm25, "tokenized_corpus": self.tokenized_corpus}, file_obj)

    def search(self, query: str, top_n: int) -> list[KeywordHit]:
        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        if not isinstance(scores, np.ndarray):
            scores = np.asarray(scores, dtype=float)

        if scores.size == 0:
            return []

        ranked_ids = np.argsort(scores)[::-1][:top_n]
        return [
            KeywordHit(index_id=int(index_id), score=float(scores[index_id]))
            for index_id in ranked_ids
            if scores[index_id] > 0
        ]
