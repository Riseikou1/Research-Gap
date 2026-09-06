"""Shared pipeline construction and one-analysis application service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.analysis.gap_candidates import GapCandidateGenerator
from src.analysis.verification import GapVerifier
from src.config import Settings
from src.extraction.paper_extractor import PaperExtractor
from src.pipeline import ResearchPipeline
from src.query.deterministic import DeterministicDecomposer
from src.query.openai_decomposer import OpenAIDecomposer
from src.query.openai_generator import OpenAIQueryGenerator
from src.ranking.lexical import LexicalScorer
from src.ranking.reranker import HybridReranker
from src.ranking.semantic import OpenAIEmbeddingProvider, SemanticScorer
from src.retrieval.multi_query import MultiQueryRetriever
from src.retrieval.openalex import OpenAlexRetriever

PIPELINE_VERSION = "m8-v1"
DecomposerName = Literal["deterministic", "openai"]
QueryGeneratorName = Literal["deterministic", "openai"]


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    decomposer: DecomposerName = "deterministic"
    query_generator: QueryGeneratorName = "deterministic"
    include_evidence: bool = False
    include_landscape: bool = False
    include_gaps: bool = False


def build_decomposer(name: DecomposerName, settings: Settings):
    if name == "deterministic":
        return DeterministicDecomposer()
    if name == "openai":
        return OpenAIDecomposer(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            cache_path=settings.cache_directory / "research_gap.sqlite3",
        )
    raise ValueError(f"Unsupported decomposer: {name}")


def build_pipeline(options: PipelineOptions, settings: Settings) -> ResearchPipeline:
    """Build the pipeline used by both the CLI and local HTTP service."""
    openalex = settings.openalex
    retriever = MultiQueryRetriever(
        OpenAlexRetriever(
            timeout=openalex.timeout_seconds,
            mailto=openalex.mailto,
            api_key=openalex.api_key,
            max_retries=openalex.max_retries,
        ),
        max_candidates=openalex.max_candidates,
        per_route_limit=openalex.per_route_limit,
        max_workers=openalex.max_workers,
        cache_path=settings.cache_directory / "research_gap.sqlite3",
        retrieval_cache_ttl_seconds=openalex.retrieval_cache_ttl_seconds,
    )

    semantic_scorer = None
    if settings.openai_api_key:
        semantic_scorer = SemanticScorer(
            OpenAIEmbeddingProvider(
                api_key=settings.openai_api_key,
                model=settings.ranking.embedding_model,
                batch_size=settings.ranking.embedding_batch_size,
            )
        )
    reranker = HybridReranker(
        LexicalScorer(),
        semantic_scorer,
        lexical_weight=settings.ranking.lexical_weight,
        semantic_weight=settings.ranking.semantic_weight,
        constraint_weight=settings.ranking.constraint_weight,
        semantic_fallback=settings.ranking.semantic_fallback,
    )
    llm_generator = None
    if options.query_generator == "openai":
        llm_generator = OpenAIQueryGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            cache_path=settings.cache_directory / "research_gap.sqlite3",
        )

    needs_evidence = options.include_evidence or options.include_landscape or options.include_gaps
    extractor = None
    if needs_evidence:
        extractor = PaperExtractor(
            api_key=settings.openai_api_key,
            model=settings.extraction_model,
            evidence_limit=settings.evidence_limit,
            max_workers=settings.extraction_workers,
            batch_size=settings.extraction_batch_size,
            cache_path=settings.cache_directory / "research_gap.sqlite3",
        )
    gap_generator = GapCandidateGenerator() if options.include_gaps else None
    gap_verifier = GapVerifier(retriever, extractor) if options.include_gaps and extractor else None
    return ResearchPipeline(
        decomposer=build_decomposer(options.decomposer, settings),
        retriever=retriever,
        reranker=reranker,
        llm_generator=llm_generator,
        extractor=extractor,
        gap_generator=gap_generator,
        gap_verifier=gap_verifier,
        evidence_limit=settings.evidence_limit,
    )


class AnalysisService:
    """Run one complete persisted analysis through the existing pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def configuration_snapshot(
        self, *, decomposer: DecomposerName, query_generator: QueryGeneratorName, paper_limit: int,
    ) -> dict[str, object]:
        settings = self.settings
        return {
            "pipeline_version": PIPELINE_VERSION,
            "decomposer": decomposer,
            "query_generator": query_generator,
            "paper_limit": paper_limit,
            "providers": {
                "decomposition": decomposer,
                "query_generation": query_generator,
                "literature": "openalex",
                "extraction": "openai",
                "embeddings": "openai" if settings.openai_api_key else None,
            },
            "models": {
                "planning": settings.openai_model if decomposer == "openai" or query_generator == "openai" else None,
                "extraction": settings.extraction_model,
                "embedding": settings.ranking.embedding_model if settings.openai_api_key else None,
            },
            "ranking": {
                "lexical_weight": settings.ranking.lexical_weight,
                "semantic_weight": settings.ranking.semantic_weight,
                "constraint_weight": settings.ranking.constraint_weight,
                "semantic_fallback": settings.ranking.semantic_fallback,
            },
            "limits": {
                "max_candidates": settings.openalex.max_candidates,
                "evidence": settings.evidence_limit,
                "retrieval_workers": settings.openalex.max_workers,
                "extraction_workers": settings.extraction_workers,
            },
        }

    def run(self, research_idea: str, *, decomposer: DecomposerName, query_generator: QueryGeneratorName,
            paper_limit: int) -> dict[str, object]:
        options = PipelineOptions(
            decomposer=decomposer,
            query_generator=query_generator,
            include_evidence=True,
            include_landscape=True,
            include_gaps=True,
        )
        result = build_pipeline(options, self.settings).run(research_idea, top_k=paper_limit)
        return result.to_dict()
