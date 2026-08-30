"""Embedding-based semantic relevance scoring."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

from src.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
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

        if client is not None:
            self.client = client
            return

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

    def embed_documents(
        self,
        texts: Sequence[str],
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

        missing = list(
            dict.fromkeys(
                text
                for text in normalized
                if text not in self._cache
            )
        )

        for start in range(
            0,
            len(missing),
            self.batch_size,
        ):
            batch = missing[
                start : start + self.batch_size
            ]

            try:
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

                self._cache[text] = _validated_vector(
                    getattr(
                        item,
                        "embedding",
                        None,
                    )
                )

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

        paper_vectors = {
            text: _validated_vector(vector)
            for text, vector in zip(
                unique_texts,
                vectors[1:],
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