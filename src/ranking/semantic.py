"""Embedding-based semantic relevance scoring."""

from __future__ import annotations

import math
import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from src.config import CACHE_DIR, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from src.models.paper import Paper


Vector = Sequence[float]


class EmbeddingProvider(Protocol):
    """Anything capable of embedding a batch of texts."""

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        ...


class SemanticScoringError(RuntimeError):
    """Raised when semantic similarity cannot be computed."""


class EmbeddingConfigurationError(SemanticScoringError):
    """Raised when the embedding backend is not configured."""


class OpenAIEmbeddingProvider:
    """OpenAI embedding backend with batching and simple caching."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        model: str = OPENAI_EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        cache_path: str | Path | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError(
                "embedding model must not be empty"
            )

        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive"
            )

        self.model = model
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}
        self._cache_lock = RLock()
        self._metrics: dict[str, int] = {
            "embedding_requests": 0,
            "embedding_cache_hits": 0,
            "persistent_embedding_cache_hits": 0,
            "new_embeddings": 0,
        }
        self._embedding_cache_path = (
            Path(cache_path)
            if cache_path is not None
            else None
        )
        self._embedding_connection: sqlite3.Connection | None = None

        if client is not None:
            self.client = client
        else:
            key = api_key or OPENAI_API_KEY

            if not key:
                raise EmbeddingConfigurationError(
                    "OPENAI_API_KEY is not configured"
                )

            try:
                from openai import OpenAI
            except ImportError as exc:
                raise EmbeddingConfigurationError(
                    "semantic scoring requires the 'openai' package"
                ) from exc

            self.client = OpenAI(
                api_key=key,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )

        if self._embedding_cache_path is None and client is None:
            self._embedding_cache_path = CACHE_DIR / "research_gap.sqlite3"

        if self._embedding_cache_path is not None:
            self._embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._embedding_connection = sqlite3.connect(
                self._embedding_cache_path,
                timeout=30.0,
                check_same_thread=False,
            )
            self._embedding_connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    PRIMARY KEY (text_hash, model)
                )
                """
            )
            self._embedding_connection.commit()

    def metrics_snapshot(self) -> dict[str, int]:
        """Return cumulative embedding work counters."""

        with self._cache_lock:
            return dict(self._metrics)

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _persistent_get(self, text: str) -> list[float] | None:
        if self._embedding_connection is None:
            return None

        with self._cache_lock:
            row = self._embedding_connection.execute(
                """
                SELECT embedding
                FROM embedding_cache
                WHERE text_hash = ? AND model = ?
                """,
                (self._text_hash(text), self.model),
            ).fetchone()

        if row is None:
            return None

        try:
            return _validated_vector(json.loads(row[0]))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    def _persistent_put(self, text: str, vector: list[float]) -> None:
        if self._embedding_connection is None:
            return

        with self._cache_lock:
            self._embedding_connection.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                    (text_hash, model, embedding)
                VALUES (?, ?, ?)
                """,
                (
                    self._text_hash(text),
                    self.model,
                    json.dumps(vector, separators=(",", ":")),
                ),
            )
            self._embedding_connection.commit()

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return self._embed_texts(texts, persist=True)

    def embed_query(self, text: str) -> list[float]:
        """Embed a query without adding it to the persistent paper cache."""

        vectors = self._embed_texts([text], persist=False)
        return vectors[0]

    def _embed_texts(
        self,
        texts: Sequence[str],
        *,
        persist: bool,
    ) -> list[list[float]]:
        if not texts:
            return []

        normalized = [
            " ".join(text.split())
            for text in texts
        ]

        if any(not text for text in normalized):
            raise SemanticScoringError(
                "embedding text must not be empty"
            )

        missing: list[str] = []
        seen_missing: set[str] = set()

        for text in normalized:
            with self._cache_lock:
                in_memory = text in self._cache

            if in_memory:
                with self._cache_lock:
                    self._metrics["embedding_cache_hits"] += 1
                continue

            persistent = self._persistent_get(text) if persist else None
            if persistent is not None:
                with self._cache_lock:
                    self._cache[text] = persistent
                    self._metrics["embedding_cache_hits"] += 1
                    self._metrics["persistent_embedding_cache_hits"] += 1
                continue

            if text not in seen_missing:
                seen_missing.add(text)
                missing.append(text)

        for start in range(
            0,
            len(missing),
            self.batch_size,
        ):
            batch = missing[
                start : start + self.batch_size
            ]

            try:
                with self._cache_lock:
                    self._metrics["embedding_requests"] += 1
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
            except Exception as exc:
                raise SemanticScoringError(
                    f"embedding request failed: {exc}"
                ) from exc

            data = list(
                getattr(response, "data", [])
                or []
            )

            if len(data) != len(batch):
                raise SemanticScoringError(
                    "embedding provider returned "
                    "an unexpected number of embeddings"
                )

            ordered = sorted(
                data,
                key=lambda item: getattr(
                    item,
                    "index",
                    -1,
                ),
            )

            for expected_index, (text, item) in enumerate(
                zip(batch, ordered)
            ):
                index = getattr(
                    item,
                    "index",
                    expected_index,
                )

                if index != expected_index:
                    raise SemanticScoringError(
                        "embedding provider returned malformed indexes"
                    )

                vector = _validated_vector(
                    getattr(
                        item,
                        "embedding",
                        None,
                    )
                )
                with self._cache_lock:
                    self._cache[text] = vector
                    self._metrics["new_embeddings"] += 1
                if persist:
                    self._persistent_put(text, vector)

        return [
            list(self._cache[text])
            for text in normalized
        ]


class SemanticScorer:
    """Score papers by embedding similarity to the research idea."""

    def __init__(
        self,
        provider: EmbeddingProvider,
    ) -> None:
        self.provider = provider

    def score_many(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> list[float]:
        if not papers:
            return []

        cleaned_query = " ".join(
            query.split()
        )

        if not cleaned_query:
            raise ValueError(
                "semantic query must not be empty"
            )

        paper_texts = [
            paper.embedding_text()
            for paper in papers
        ]

        unique_texts = list(
            dict.fromkeys(paper_texts)
        )

        embed_query = getattr(self.provider, "embed_query", None)
        if callable(embed_query):
            query_vector = _validated_vector(
                embed_query(cleaned_query)
            )
            vectors = self.provider.embed_documents(unique_texts)
        else:
            vectors = self.provider.embed_documents(
                [
                    cleaned_query,
                    *unique_texts,
                ]
            )

            if len(vectors) != len(unique_texts) + 1:
                raise SemanticScoringError(
                    "embedding provider returned "
                    "an unexpected vector count"
                )

            query_vector = _validated_vector(
                vectors[0]
            )

        if len(vectors) != len(unique_texts):
            raise SemanticScoringError(
                "embedding provider returned "
                "an unexpected paper vector count"
            )

        paper_vector_values = (
            vectors
            if callable(embed_query)
            else vectors[1:]
        )
        paper_vectors = {
            text: _validated_vector(vector)
            for text, vector in zip(
                unique_texts,
                paper_vector_values,
            )
        }

        return [
            _cosine(
                query_vector,
                paper_vectors[text],
            )
            for text in paper_texts
        ]


def _validated_vector(
    value: object,
) -> list[float]:
    """Validate and normalize an embedding vector."""

    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
    ):
        raise SemanticScoringError(
            "embedding provider returned a non-vector"
        )

    vector: list[float] = []

    for component in value:
        if (
            isinstance(component, bool)
            or not isinstance(
                component,
                (int, float),
            )
        ):
            raise SemanticScoringError(
                "embedding vector contains non-numeric data"
            )

        number = float(component)

        if not math.isfinite(number):
            raise SemanticScoringError(
                "embedding vector contains non-finite data"
            )

        vector.append(number)

    if not vector:
        raise SemanticScoringError(
            "embedding provider returned an empty vector"
        )

    return vector


def _cosine(
    left: Vector,
    right: Vector,
) -> float:
    """Return cosine similarity between two embedding vectors."""

    if len(left) != len(right):
        raise SemanticScoringError(
            "embedding dimensions do not match"
        )

    left_norm = math.sqrt(
        sum(value * value for value in left)
    )

    right_norm = math.sqrt(
        sum(value * value for value in right)
    )

    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(
        a * b
        for a, b in zip(left, right)
    )

    return dot_product / (
        left_norm * right_norm
    )
