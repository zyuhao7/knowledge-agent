from __future__ import annotations

import math
from collections import Counter

from .embeddings import Embedder, tokenize
from .models import Chunk, SearchHit


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class VectorStore:
    """Hybrid retrieval: hashed/remote vectors plus BM25, both cheap enough for a demo."""

    def __init__(self, embedder: Embedder, k1: float = 1.5, b: float = 0.75) -> None:
        self.embedder = embedder
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.titles: dict[str, str] = {}
        self.vectors: list[list[float]] = []
        self.tokens: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.doc_freq: Counter[str] = Counter()

    def add(self, chunks: list[Chunk], titles: dict[str, str]) -> None:
        if not chunks:
            return
        self.titles.update(titles)
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            counts = Counter(tokenize(chunk.text))
            self.chunks.append(chunk)
            self.vectors.append(vector)
            self.tokens.append(counts)
            self.lengths.append(sum(counts.values()))
            self.doc_freq.update(counts.keys())

    @property
    def avg_length(self) -> float:
        return (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    def _bm25(self, query_tokens: list[str]) -> list[float]:
        total = len(self.chunks)
        average = self.avg_length or 1.0
        scores: list[float] = []
        for counts, length in zip(self.tokens, self.lengths):
            score = 0.0
            for token in query_tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.doc_freq.get(token, 0)
                inverse = math.log(
                    1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / average)
                score += inverse * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not self.chunks:
            return []
        query_vector = self.embedder.embed([query])[0]
        query_tokens = tokenize(query)
        keyword = self._bm25(query_tokens)
        keyword_max = max(keyword) or 1.0
        hits: list[SearchHit] = []
        for chunk, vector, keyword_score in zip(self.chunks, self.vectors, keyword):
            vector_score = max(0.0, _cosine(query_vector, vector))
            normalized_keyword = keyword_score / keyword_max
            score = 0.65 * vector_score + 0.35 * normalized_keyword
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=score,
                    vector_score=vector_score,
                    keyword_score=normalized_keyword,
                    title=self.titles.get(chunk.doc_id, chunk.doc_id),
                )
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
