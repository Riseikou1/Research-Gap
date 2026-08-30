"""Command-line entry point for hybrid literature retrieval and reranking."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from src.config import ConfigurationError, Settings
from src.extraction.paper_extractor import PaperExtractor
from src.models.paper import Paper
from src.models.query import SearchQuery
from src.pipeline import PipelineError, ResearchPipeline, ResearchResult
from src.query.deterministic import DeterministicDecomposer
from src.query.openai_decomposer import (
    OpenAIConfigurationError,
    OpenAIDecomposer,
    OpenAIDecompositionError,
)
from src.query.openai_generator import (
    OpenAIQueryGenerationError,
    OpenAIQueryGenerator,
)
from src.ranking.lexical import LexicalScorer
from src.ranking.reranker import HybridReranker
from src.ranking.semantic import (
    OpenAIEmbeddingProvider,
    SemanticScorer,
    SemanticScoringError,
)
from src.retrieval.multi_query import MultiQueryRetriever
from src.retrieval.openalex import OpenAlexRetriever


MIN_PAPER_LIMIT = 1
MAX_PAPER_LIMIT = 100
DEFAULT_PAPER_LIMIT = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find and hybrid-rerank research papers related to a proposed "
            "research idea."
        )
    )
    parser.add_argument("idea", nargs="*", help="Your proposed research idea")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_PAPER_LIMIT,
        metavar="N",
        help=(
            f"Number of ranked papers to show ({MIN_PAPER_LIMIT}-"
            f"{MAX_PAPER_LIMIT}, default: {DEFAULT_PAPER_LIMIT})"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print JSON")
    parser.add_argument(
        "--decomposer",
        choices=("deterministic", "openai"),
        default="deterministic",
        help="Research-idea decomposer (default: deterministic)",
    )
    parser.add_argument(
        "--query-generator",
        choices=("deterministic", "openai"),
        default="deterministic",
        help=(
            "Query expansion backend; openai supplements the deterministic "
            "baseline (default: deterministic)"
        ),
    )
    parser.add_argument(
        "--show-queries",
        action="store_true",
        help="Show structured facets and query provenance",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Show lexical and semantic component scores",
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Extract and display structured evidence for the top papers",
    )
    return parser


def build_decomposer(
    name: str,
    settings: Settings | None = None,
) -> DeterministicDecomposer | OpenAIDecomposer:
    if name == "deterministic":
        return DeterministicDecomposer()
    if name == "openai":
        return OpenAIDecomposer(
            api_key=settings.openai_api_key if settings else None,
            model=settings.openai_model if settings else None,
        )
    raise ValueError(f"Unsupported decomposer: {name}")


def build_pipeline(args: argparse.Namespace, settings: Settings) -> ResearchPipeline:
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
    )

    semantic_scorer: SemanticScorer | None = None
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
    )
    llm_generator = (
        OpenAIQueryGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        if args.query_generator == "openai"
        else None
    )
    extractor = PaperExtractor(
        api_key=settings.openai_api_key,
        model=settings.extraction_model,
        evidence_limit=settings.evidence_limit,
    ) if args.show_evidence else None
    return ResearchPipeline(
        decomposer=build_decomposer(args.decomposer, settings),
        retriever=retriever,
        reranker=reranker,
        llm_generator=llm_generator,
        extractor=extractor,
        evidence_limit=settings.evidence_limit,
    )


def read_research_idea(parts: list[str]) -> str:
    idea = " ".join(parts).strip()
    if idea:
        return idea
    try:
        return input("Enter your research idea: ").strip()
    except EOFError:
        return ""


def print_query_context(result: ResearchResult) -> None:
    print("Facets:")
    for name, value in result.idea.model_dump().items():
        if name != "original_text":
            print(f"  {name}: {json.dumps(value, ensure_ascii=False)}")
    print("Planned queries:")
    for index, query in enumerate(result.queries, start=1):
        origins = ", ".join(
            f"{origin.source}/{origin.strategy}"
            + (f"/{origin.provider}" if origin.provider else "")
            for origin in query.origins
        )
        print(f"  {index}. {query.text} [{origins}]")


def print_papers(papers: list[Paper], *, show_scores: bool = False) -> None:
    if not papers:
        print("No papers found.")
        return
    for index, paper in enumerate(papers, start=1):
        authors = ", ".join(paper.authors) or "Unknown authors"
        year = paper.publication_year or "year unknown"
        link = paper.doi or paper.url or "Unavailable"
        abstract = paper.abstract or "Abstract unavailable"
        print(f"\n{index}. {paper.title}")
        print(f"   Authors: {authors}")
        print(f"   Year: {year}")
        if paper.final_score is not None:
            print(f"   Final relevance: {paper.final_score:.3f}")
        if show_scores:
            lexical = _format_score(paper.lexical_score)
            semantic = _format_score(paper.semantic_score)
            print(f"   Lexical: {lexical} | Semantic: {semantic}")
            routes = [
                f"{item.mode.value}: {item.query.text} "
                f"({item.query.source}/{item.query.strategy})"
                for item in paper.provenance
            ]
            if routes:
                print(f"   Retrieval routes: {'; '.join(routes)}")
        print(f"   Citations: {paper.citation_count}")
        print(f"   Link: {link}")
        print(f"   Abstract: {abstract}")


def handle_retrieval_failures(result: ResearchResult) -> None:
    if not result.retrieval_failures:
        return
    details = "; ".join(
        f"{failure.mode}/{failure.query}: {failure.error}"
        for failure in result.retrieval_failures
    )
    print(f"Warning: partial retrieval; failed routes: {details}", file=sys.stderr)


def _format_score(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _paper_json(paper: Paper) -> dict[str, object]:
    value = paper.to_legacy_dict()
    value.update(
        {
            "publication_date": (
                paper.publication_date.isoformat()
                if paper.publication_date
                else None
            ),
            "source": paper.source,
            "matched_queries": paper.matched_queries,
            "retrieval_modes": paper.retrieval_modes,
            "lexical_score": paper.lexical_score,
            "semantic_score": paper.semantic_score,
            "final_score": paper.final_score,
            "ranking_mode": paper.ranking_mode,
        }
    )
    return value


def print_evidence(result: ResearchResult) -> None:
    if not result.evidence:
        print("\nNo structured evidence was extracted.")
        return
    by_id = {item.paper_id: item for item in result.evidence}
    for index, paper in enumerate(result.papers, start=1):
        item = by_id.get(paper.id)
        if item is None:
            continue
        print(f"\n{index}. {item.title}\n  Research objective: "
              f"{item.research_objective.value if item.research_objective else '—'}")
        for label, values in (("Methods", item.method_or_intervention),
                              ("Datasets", item.datasets),
                              ("Baselines", item.comparison_or_baseline),
                              ("Metrics", item.evaluation_metrics),
                              ("Main findings", item.main_findings),
                              ("Author-stated limitations", item.limitations),
                              ("Future work", item.future_work)):
            print(f"  {label}:")
            for value in values:
                print(f"    - {value.value}")
        print(f"  Missing: {', '.join(item.missing_fields) or 'none'}")
    for failure in result.extraction_failures:
        print(f"Warning: {failure}", file=sys.stderr)


def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    idea = read_research_idea(args.idea)
    if not idea:
        print("Error: a research idea is required.", file=sys.stderr)
        return 2
    if not MIN_PAPER_LIMIT <= args.limit <= MAX_PAPER_LIMIT:
        print(
            f"Error: --limit must be between {MIN_PAPER_LIMIT} "
            f"and {MAX_PAPER_LIMIT}.",
            file=sys.stderr,
        )
        return 2

    try:
        settings = Settings.from_env()
        if args.limit > settings.openalex.max_candidates:
            raise ConfigurationError(
                "--limit cannot exceed RESEARCH_GAP_MAX_CANDIDATES "
                f"({settings.openalex.max_candidates})"
            )
        result = build_pipeline(args, settings).run(idea, top_k=args.limit)
    except (
        ConfigurationError,
        OpenAIConfigurationError,
        OpenAIDecompositionError,
        OpenAIQueryGenerationError,
        PipelineError,
        SemanticScoringError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    handle_retrieval_failures(result)
    for notice in result.notices:
        print(f"Notice: {notice}", file=sys.stderr)

    if args.json:
        payload: Any
        if args.show_queries:
            payload = result.to_dict()
        elif args.show_evidence:
            payload = result.to_dict()
        else:
            payload = [_paper_json(paper) for paper in result.papers]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.show_queries:
        print_query_context(result)
    print(f"Found {result.candidate_count} unique candidate papers.")
    print(
        f"Showing top {len(result.papers)} ranked results for: {idea} "
        f"({result.ranking_mode})"
    )
    print_papers(result.papers, show_scores=args.show_scores)
    if args.show_evidence:
        print_evidence(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
