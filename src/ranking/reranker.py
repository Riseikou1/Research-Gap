"""Centralized score normalization, fusion, and deterministic reranking."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from src.models.idea import ResearchIdea
from src.models.paper import Paper
from src.ranking.lexical import LexicalScorer
from src.ranking.semantic import SemanticScorer, SemanticScoringError
from src.retrieval.deduplication import normalize_title


LOGGER = logging.getLogger(__name__)

DEFAULT_CONSTRAINT_WEIGHT = 0.15


@dataclass(frozen=True, slots=True)
class RankingResult:
    papers: list[Paper]
    mode: str
    notice: str | None = None


class HybridReranker:
    """Combine lexical and semantic relevance into one final ranking."""

    def __init__(
        self,
        lexical_scorer: LexicalScorer,
        semantic_scorer: SemanticScorer | None,
        *,
        lexical_weight: float = 0.4,
        semantic_weight: float = 0.6,
        constraint_weight: float = DEFAULT_CONSTRAINT_WEIGHT,
    ) -> None:
        if lexical_weight < 0 or semantic_weight < 0:
            raise ValueError("ranking weights must be non-negative")
        if lexical_weight + semantic_weight <= 0:
            raise ValueError("at least one ranking weight must be positive")
        if not 0 <= constraint_weight <= 1:
            raise ValueError("constraint_weight must be between 0 and 1")

        self.lexical_scorer = lexical_scorer
        self.semantic_scorer = semantic_scorer
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.constraint_weight = constraint_weight

    def rerank(self, idea: ResearchIdea, papers: Sequence[Paper]) -> RankingResult:
        if not papers:
            return RankingResult(papers=[], mode="lexical_only")

        ranked = [paper.model_copy(deep=True) for paper in papers]

        lexical_raw = self.lexical_scorer.score_many(idea.original_text, ranked)
        lexical_scores = _normalize_scores(lexical_raw)

        semantic_raw: list[float] | None = None
        semantic_scores: list[float] | None = None
        notice: str | None = None

        if self.semantic_scorer is not None and self.semantic_weight > 0:
            try:
                semantic_raw = self.semantic_scorer.score_many(idea.original_text, ranked)
                semantic_scores = _normalize_scores(semantic_raw)
            except SemanticScoringError as exc:
                notice = "Semantic scoring failed; using lexical-only ranking."
                LOGGER.warning("%s error=%s", notice, exc)

        elif self.semantic_weight > 0:
            notice = "Semantic scorer is unavailable; using lexical-only ranking."
            LOGGER.warning(notice)

        mode = "hybrid" if semantic_scores is not None else "lexical_only"
        base_scores = self._fuse_scores(lexical_scores, semantic_scores)

        for index, paper in enumerate(ranked):
            paper.lexical_raw_score = lexical_raw[index]
            paper.lexical_score = lexical_scores[index]

            if semantic_scores is not None and semantic_raw is not None:
                paper.semantic_raw_score = semantic_raw[index]
                paper.semantic_score = semantic_scores[index]
            else:
                paper.semantic_raw_score = None
                paper.semantic_score = None

            paper.constraint_score = None
            paper.final_score = base_scores[index]
            paper.ranking_mode = mode

        constraint_queries = tuple(
            " ".join(constraint.split())
            for constraint in idea.constraints
            if " ".join(constraint.split())
        )
        if constraint_queries:
            constraint_scores = self._constraint_scores(constraint_queries, ranked, mode)
            for paper, constraint_score in zip(ranked, constraint_scores):
                paper.constraint_score = constraint_score
                base_score = paper.final_score or 0.0
                paper.final_score = min(
                    1.0,
                    base_score * (1.0 + self.constraint_weight * constraint_score),
                )

        ranked.sort(key=_sort_key)

        LOGGER.info("ranking mode=%s paper_count=%d", mode, len(ranked))

        return RankingResult(papers=ranked, mode=mode, notice=notice)

    def _constraint_scores(
        self,
        constraints: Sequence[str],
        papers: Sequence[Paper],
        mode: str,
    ) -> list[float]:
        per_constraint: list[list[float]] = []
        for constraint in constraints:
            query = " ".join(constraint.split())
            if not query:
                continue

            lexical = _normalize_scores(
                self.lexical_scorer.score_many(query, papers)
            )
            semantic: list[float] | None = None
            if self.semantic_scorer is not None and mode == "hybrid":
                try:
                    semantic = _normalize_scores(
                        self.semantic_scorer.score_many(query, papers)
                    )
                except SemanticScoringError as exc:
                    LOGGER.debug(
                        "Constraint semantic scoring failed for %r: %s",
                        query,
                        exc,
                    )

            if semantic is None:
                per_constraint.append(lexical)
            else:
                per_constraint.append(self._fuse_scores(lexical, semantic))

        if not per_constraint:
            return [0.0] * len(papers)
        return [
            sum(scores[index] for scores in per_constraint) / len(per_constraint)
            for index in range(len(papers))
        ]

    def _fuse_scores(
        self,
        lexical: Sequence[float],
        semantic: Sequence[float] | None,
    ) -> list[float]:
        if semantic is None:
            return list(lexical)
        total_weight = self.lexical_weight + self.semantic_weight
        return [
            (
                self.lexical_weight * lexical[index]
                + self.semantic_weight * semantic[index]
            )
            / total_weight
            for index in range(len(lexical))
        ]


def _normalize_scores(values: Sequence[float]) -> list[float]:
    """Normalize scores into the [0, 1] range."""

    if not values:
        return []

    if any(not math.isfinite(value) for value in values):
        raise ValueError("ranking scores must be finite")

    low = min(values)
    high = max(values)

    if math.isclose(low, high, abs_tol=1e-12):
        constant = 1.0 if high > 0 else 0.0
        return [constant] * len(values)

    return [(value - low) / (high - low) for value in values]


def _sort_key(paper: Paper) -> tuple[float, float, float, float, str, int, str]:
    """Provide deterministic ordering when scores tie."""

    return (
        -(paper.final_score or 0.0),
        -(paper.constraint_score if paper.constraint_score is not None else -1.0),
        -(paper.semantic_score if paper.semantic_score is not None else -1.0),
        -(paper.lexical_score or 0.0),
        normalize_title(paper.title),
        paper.publication_year or 0,
        paper.id.casefold(),
    )
