"""Deterministic BM25-style lexical relevance scoring."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from src.models.paper import Paper


_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
}


class LexicalScorer:
    """Score paper title/abstract relevance against a research idea."""

    def __init__(
        self,
        *,
        title_boost: int = 3,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if title_boost < 1:
            raise ValueError("title_boost must be positive")

        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("invalid BM25 parameters")

        self.title_boost = title_boost
        self.k1 = k1
        self.b = b

    def score_many(
        self,
        query: str,
        papers: Sequence[Paper],
    ) -> list[float]:
        """Score papers relative to the same candidate corpus."""

        if not papers:
            return []

        query_tokens = _meaningful_tokens(query)

        if not query_tokens:
            return [0.0] * len(papers)

        unique_query_tokens = list(
            dict.fromkeys(query_tokens)
        )

        title_tokens = [
            _meaningful_tokens(paper.title)
            for paper in papers
        ]

        abstract_tokens = [
            _meaningful_tokens(paper.abstract or "")
            for paper in papers
        ]

        # Repeating title tokens gives title matches more weight than
        # abstract-only matches.
        documents = [
            title * self.title_boost + abstract
            for title, abstract in zip(
                title_tokens,
                abstract_tokens,
            )
        ]

        average_length = (
            sum(len(document) for document in documents)
            / len(documents)
        ) or 1.0

        document_frequency = {
            token: sum(
                token in document
                for document in documents
            )
            for token in unique_query_tokens
        }

        query_set = set(unique_query_tokens)
        normalized_query = _normalized_phrase(query)

        scores: list[float] = []

        for paper, title, abstract, document in zip(
            papers,
            title_tokens,
            abstract_tokens,
            documents,
        ):
            frequencies = Counter(document)

            length_normalization = self.k1 * (
                1
                - self.b
                + self.b
                * len(document)
                / average_length
            )

            bm25 = 0.0

            for token in unique_query_tokens:
                frequency = frequencies[token]

                if not frequency:
                    continue

                df = document_frequency[token]

                inverse_frequency = math.log(
                    1
                    + (
                        len(documents)
                        - df
                        + 0.5
                    )
                    / (df + 0.5)
                )

                bm25 += inverse_frequency * (
                    frequency * (self.k1 + 1)
                    / (
                        frequency
                        + length_normalization
                    )
                )

            title_coverage = (
                len(query_set.intersection(title))
                / len(query_set)
            )

            abstract_coverage = (
                len(query_set.intersection(abstract))
                / len(query_set)
            )

            phrase_bonus = 0.0

            if normalized_query:
                if normalized_query in _normalized_phrase(
                    paper.title
                ):
                    phrase_bonus = 2.0

                elif normalized_query in _normalized_phrase(
                    paper.abstract or ""
                ):
                    phrase_bonus = 0.75

            scores.append(
                bm25
                + 1.5 * title_coverage
                + 0.5 * abstract_coverage
                + phrase_bonus
            )

        return scores


def _meaningful_tokens(
    text: str,
) -> list[str]:
    normalized = unicodedata.normalize(
        "NFKC",
        text,
    ).casefold()

    normalized = re.sub(
        r"[-./]",
        " ",
        normalized,
    )

    tokens = re.findall(
        r"[^\W_]+(?:[+#][^\W_]*)*",
        normalized,
        re.UNICODE,
    )

    useful = [
        token
        for token in tokens
        if token not in _STOPWORDS
    ]

    return useful or tokens


def _normalized_phrase(
    text: str,
) -> str:
    return " ".join(
        _meaningful_tokens(text)
    )