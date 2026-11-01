from __future__ import annotations

import hashlib
import math
import re
from itertools import pairwise
from typing import Protocol

import httpx

from .config import Settings

_LATIN = re.compile(r"[a-zA-Z][a-zA-Z0-9_.+-]*|\d+(?:\.\d+)?")
_CJK = re.compile(r"[一-鿿]")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def tokenize(text: str) -> list[str]:
    """Latin words plus CJK unigrams and bigrams; enough for a dependency-free demo."""
    lower = text.lower()
    tokens = [token.lower() for token in _LATIN.findall(lower)]
    chars = _CJK.findall(lower)
    tokens.extend(chars)
    tokens.extend(a + b for a, b in pairwise(chars))
    return tokens


class HashingEmbedder:
    """Deterministic feature hashing, so the project runs with no API key or model download."""

    name = "hashing-512"
    dim = 512

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in tokenize(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest, "big") % self.dim
                vector[bucket] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class OpenAIEmbedder:
    """Remote embeddings for production use; enable with USE_REMOTE_EMBEDDINGS=1."""

    name = "openai"

    def __init__(self, settings: Settings, batch_size: int = 64) -> None:
        if not settings.api_key:
            raise ValueError("OPENAI_API_KEY is required for remote embeddings")
        self.settings = settings
        self.batch_size = batch_size
        self.dim = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=60) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                response = client.post(
                    f"{self.settings.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.settings.api_key}"},
                    json={"model": self.settings.embedding_model, "input": batch},
                )
                response.raise_for_status()
                data = response.json()["data"]
                vectors.extend(
                    item["embedding"] for item in sorted(data, key=lambda item: item["index"])
                )
        if vectors:
            self.dim = len(vectors[0])
        return vectors


def get_embedder(settings: Settings) -> Embedder:
    import os

    if os.environ.get("USE_REMOTE_EMBEDDINGS") == "1" and settings.api_key:
        return OpenAIEmbedder(settings)
    return HashingEmbedder()
