"""Evaluation-only execution of the six existing retrieval ablations."""

from enum import StrEnum
from collections.abc import Sequence
from dataclasses import dataclass

from src.models.idea import ResearchIdea
from src.models.paper import Paper
from src.models.query import SearchQuery
from src.query.deterministic import clean_idea_text
from src.query.generator import DeterministicQueryGenerator
from src.query.planner import QueryPlanner


class AblationVariant(StrEnum):
    ORIGINAL_ONLY = "original_only"
    DETERMINISTIC_EXPANSION = "deterministic_expansion"
    LLM_EXPANSION = "llm_expansion"
    COMBINED_EXPANSION = "combined_expansion"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    HYBRID_RERANKED = "hybrid_reranked"


@dataclass(frozen=True, slots=True)
class AblationPrediction:
    """One generated retrieval prediction and its execution status."""

    variant: AblationVariant
    available: bool
    retrieved_ids: tuple[str, ...] = ()
    queries: tuple[SearchQuery, ...] = ()
    papers: tuple[Paper, ...] = ()
    ranking_mode: str | None = None
    unavailable_dependencies: tuple[str, ...] = ()

    @property
    def unavailable_reason(self) -> str | None:
        if self.available:
            return None
        return ", ".join(self.unavailable_dependencies) or "unknown dependency"


class AblationPredictionGenerator:
    """Compose production components for offline/live ablation evaluation.

    The first four variants use the existing bounded lexical retrieval path;
    the last two use the existing hybrid route construction, with the final
    variant applying the existing reranker. No provider, query, or ranking
    implementation is duplicated here.
    """

    def __init__(
        self,
        *,
        decomposer,
        retriever,
        deterministic_generator=None,
        llm_generator=None,
        query_planner: QueryPlanner | None = None,
        reranker=None,
        retrieval_limit: int | None = None,
    ) -> None:
        self.decomposer = decomposer
        self.retriever = retriever
        self.deterministic_generator = (
            deterministic_generator or DeterministicQueryGenerator()
        )
        self.llm_generator = llm_generator
        self.query_planner = query_planner or QueryPlanner()
        self.reranker = reranker
        self.retrieval_limit = retrieval_limit

    def generate(
        self,
        idea: str | ResearchIdea,
        variant: AblationVariant | str,
    ) -> AblationPrediction:
        variant = AblationVariant(variant)
        missing: list[str] = []
        if self.retriever is None:
            missing.append("retriever")
        if isinstance(idea, str):
            if self.decomposer is None:
                missing.append("decomposer")
                return _unavailable(variant, missing)
            idea_model = self.decomposer.decompose(idea)
        else:
            idea_model = idea

        if variant in {
            AblationVariant.LLM_EXPANSION,
            AblationVariant.COMBINED_EXPANSION,
            AblationVariant.HYBRID_RETRIEVAL,
            AblationVariant.HYBRID_RERANKED,
        } and self.llm_generator is None:
            missing.append("llm_generator")
        if variant is AblationVariant.HYBRID_RERANKED and self.reranker is None:
            missing.append("reranker")
        if missing:
            return _unavailable(variant, missing)

        original = SearchQuery(
            text=clean_idea_text(idea_model.original_text),
            strategy="original",
            source="deterministic",
        )
        deterministic = (
            self.deterministic_generator.generate(idea_model)
            if variant
            in {
                AblationVariant.DETERMINISTIC_EXPANSION,
                AblationVariant.COMBINED_EXPANSION,
                AblationVariant.HYBRID_RETRIEVAL,
                AblationVariant.HYBRID_RERANKED,
            }
            else []
        )
        llm = (
            self.llm_generator.generate(idea_model)
            if variant
            in {
                AblationVariant.LLM_EXPANSION,
                AblationVariant.COMBINED_EXPANSION,
                AblationVariant.HYBRID_RETRIEVAL,
                AblationVariant.HYBRID_RERANKED,
            }
            else []
        )
        if variant is AblationVariant.ORIGINAL_ONLY:
            queries = [original]
        else:
            queries = self.query_planner.plan(idea_model, deterministic, llm)

        if variant in {
            AblationVariant.ORIGINAL_ONLY,
            AblationVariant.DETERMINISTIC_EXPANSION,
            AblationVariant.LLM_EXPANSION,
            AblationVariant.COMBINED_EXPANSION,
        }:
            retrieval = self.retriever.retrieve_verification(
                queries,
                adaptive=False,
                limit=self.retrieval_limit,
            )
        else:
            retrieval = self.retriever.retrieve_hybrid(
                queries,
                limit=self.retrieval_limit,
            )

        papers = list(retrieval.papers)
        ranking_mode = None
        if variant is AblationVariant.HYBRID_RERANKED:
            ranking = self.reranker.rerank(idea_model, papers)
            papers = list(ranking.papers)
            ranking_mode = ranking.mode

        return AblationPrediction(
            variant=variant,
            available=True,
            retrieved_ids=tuple(paper.id for paper in papers),
            queries=tuple(queries),
            papers=tuple(papers),
            ranking_mode=ranking_mode,
        )


def _unavailable(variant: AblationVariant, dependencies: Sequence[str]) -> AblationPrediction:
    return AblationPrediction(
        variant=variant,
        available=False,
        unavailable_dependencies=tuple(dict.fromkeys(dependencies)),
    )
