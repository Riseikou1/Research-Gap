"""Command-line entry point for hybrid literature retrieval and reranking."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import logging
import pstats
import sys
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

from src.config import CACHE_DIR, ConfigurationError, Settings
from src.analysis.gap_candidates import GapCandidateGenerator, is_concrete_entity
from src.analysis.verification import GapVerifier
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
from src.reporting.landscape import format_landscape


MIN_PAPER_LIMIT = 1
MAX_PAPER_LIMIT = 100
DEFAULT_PAPER_LIMIT = 20


class TimingTracker:
    """Collect lightweight wall-clock timings for major CLI stages."""

    def __init__(self) -> None:
        self.started_at = perf_counter()
        self.timings: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started_at
            self.timings[name] = self.timings.get(name, 0.0) + elapsed

    def print_report(
        self,
        metrics: dict[str, int] | None = None,
        stage_timings: dict[str, float] | None = None,
    ) -> None:
        total = perf_counter() - self.started_at

        print("\nTiming", file=sys.stderr)
        print("======", file=sys.stderr)
        for name, elapsed in self.timings.items():
            print(f"{name:24} {elapsed:8.2f}s", file=sys.stderr)
        print(f"{'total':24} {total:8.2f}s", file=sys.stderr)
        if stage_timings:
            print("\nPipeline stages", file=sys.stderr)
            print("===============", file=sys.stderr)
            for name, elapsed in stage_timings.items():
                print(f"{name:24} {elapsed:8.2f}s", file=sys.stderr)
        if metrics:
            print("\nWork accounting", file=sys.stderr)
            print("===============", file=sys.stderr)
            for name, value in metrics.items():
                print(f"{name:42} {value:8d}", file=sys.stderr)


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
    parser.add_argument(
        "--show-gaps",
        action="store_true",
        help="Extract evidence and display cautious candidate research gaps",
    )
    parser.add_argument(
        "--show-landscape",
        action="store_true",
        help=(
            "Extract evidence and display the deterministic literature "
            "landscape"
        ),
    )
    parser.add_argument(
        "--show-timings",
        action="store_true",
        help="Show wall-clock timing for major pipeline stages",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile ResearchPipeline.run() and show the slowest internal functions",
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
            cache_path=CACHE_DIR / "research_gap.sqlite3",
        )
    raise ValueError(f"Unsupported decomposer: {name}")


def build_pipeline(args: argparse.Namespace, settings: Settings) -> ResearchPipeline:
    show_gaps = getattr(args, "show_gaps", False)
    show_landscape = getattr(args, "show_landscape", False)
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
        cache_path=CACHE_DIR / "research_gap.sqlite3",
        retrieval_cache_ttl_seconds=openalex.retrieval_cache_ttl_seconds,
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
        constraint_weight=settings.ranking.constraint_weight,
    )
    llm_generator = (
        OpenAIQueryGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            cache_path=CACHE_DIR / "research_gap.sqlite3",
        )
        if args.query_generator == "openai"
        else None
    )
    extractor = PaperExtractor(
        api_key=settings.openai_api_key,
        model=settings.extraction_model,
        evidence_limit=settings.evidence_limit,
        max_workers=settings.extraction_workers,
        batch_size=settings.extraction_batch_size,
        cache_path=CACHE_DIR / "research_gap.sqlite3",
    ) if (args.show_evidence or show_gaps or show_landscape) else None
    gap_generator = GapCandidateGenerator() if show_gaps else None
    gap_verifier = GapVerifier(
        retriever,
        extractor,
    ) if show_gaps and extractor is not None else None
    return ResearchPipeline(
        decomposer=build_decomposer(args.decomposer, settings),
        retriever=retriever,
        reranker=reranker,
        llm_generator=llm_generator,
        extractor=extractor,
        gap_generator=gap_generator,
        gap_verifier=gap_verifier,
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
            constraint = _format_score(paper.constraint_score)
            print(
                f"   Lexical: {lexical} | Semantic: {semantic} | "
                f"Constraint: {constraint}"
            )
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
            "constraint_score": paper.constraint_score,
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
        print(f"\n{index}. {item.title}\n  Study type: {item.study_type}\n"
              "  Research objective: "
              f"{item.research_objective.value if item.research_objective else '—'}")
        for label, values in (("Methods", [value for value in item.method_or_intervention if is_concrete_entity(value.value)]),
                              ("Datasets", [value for value in item.datasets if is_concrete_entity(value.value)]),
                              ("Sample size", [item.sample_size] if item.sample_size else []),
                              ("Comparisons", [value for value in item.comparison_or_baseline if is_concrete_entity(value.value)]),
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


def print_gaps(result: ResearchResult) -> None:
    print_idea_assessment(result)
    print("\nCandidate Research Gap Assessments")
    print("==================================")
    if not result.gaps:
        print("No evidence-backed candidate gaps were identified.")
    for index, gap in enumerate(result.gaps, start=1):
        print(f"\n{index}. {gap.title}")
        label = gap.final_label or (gap.verification.label if gap.verification else "uncertain")
        print(f"   Assessment: {label.replace('_', ' ').capitalize()}")
        print(f"   Pattern: {gap.pattern_type}")
        print(f"   Hypothesis: {gap.description}")
        print(f"   Rationale: {gap.rationale}")
        if gap.landscape_basis:
            print("   Landscape basis:")
            for basis in gap.landscape_basis:
                print(f"   - {basis.dimension}={basis.value}: {basis.count}/{basis.total}")
        if gap.supporting_evidence:
            print("   Supporting evidence:")
            for item in gap.supporting_evidence[:5]:
                print(f"   - {item.paper_id} ({item.role}, {item.evidence_type}): {item.value}")
        verification = gap.verification
        if verification:
            print("   Verification searches:")
            for item in verification.verification_queries:
                print(f"   - {item.query}")
            print(f"   Searched papers: {len(verification.searched_paper_ids)}")
            if verification.contradicting_paper_ids:
                print("   Counterexamples:")
                for paper_id in verification.contradicting_paper_ids:
                    print(f"   - {paper_id} (confirmed contradiction)")
            elif verification.potential_contradiction_paper_ids:
                print("   Potential counterexamples (not confirmed):")
                for paper_id in verification.potential_contradiction_paper_ids:
                    print(f"   - {paper_id}")
            else:
                print("   Counterexamples: none found after targeted verification")
            print("   Verification: " + verification.reason)
            for note in verification.coverage_notes:
                print(f"   Coverage: {note}")
            for failure in verification.failures:
                print(f"   Verification failure: {failure.provider}: {failure.error}")
    print(
        "\nGlobal qualification\n--------------------\n"
        "These assessments describe retrieved and verified evidence only; they do not prove global novelty."
    )


def print_idea_assessment(result: ResearchResult) -> None:
    assessment = result.idea_assessment
    print("\nResearch Idea Assessment")
    print("========================")
    if assessment is None:
        print("Assessment: Uncertain")
        print("Rationale: Direct idea verification did not execute.")
        return
    print(f"Assessment: {assessment.label.replace('_', ' ').capitalize()}")
    print(f"Rationale: {assessment.rationale}")
    if assessment.counterexample_paper_ids:
        print("Counterexamples / direct matches:")
        for paper_id in assessment.counterexample_paper_ids:
            print(f"- {paper_id}")
            facets = assessment.matched_facets.get(paper_id, [])
            if facets:
                print(f"  matched: {', '.join(facets)}")
    else:
        print("Counterexamples / direct matches: none confirmed")
    if assessment.partial_match_paper_ids:
        print("Partial/contextual support:")
        for paper_id in assessment.partial_match_paper_ids:
            facets = assessment.matched_facets.get(paper_id, [])
            suffix = f" (matched: {', '.join(facets)})" if facets else ""
            print(f"- {paper_id}{suffix}")
    if assessment.potential_match_paper_ids:
        print("Potential matches:")
        for paper_id in assessment.potential_match_paper_ids:
            facets = assessment.matched_facets.get(paper_id, [])
            suffix = f" (matched: {', '.join(facets)})" if facets else ""
            print(f"- {paper_id}{suffix}")
    if assessment.supporting_evidence:
        print("Supporting evidence:")
        for item in assessment.supporting_evidence[:8]:
            print(f"- {item.paper_id} ({item.role}, {item.evidence_type}): {item.value}")
    print("Verification searches:")
    for item in assessment.verification_queries:
        print(f"- {item.query}")
    for note in assessment.coverage_notes:
        print(f"Coverage: {note}")
    for failure in assessment.failures:
        print(f"Verification failure: {failure.provider}: {failure.error}")

def run_pipeline(
    pipeline: ResearchPipeline,
    idea: str,
    top_k: int,
    *,
    profile: bool = False,
) -> ResearchResult:
    if not profile:
        return pipeline.run(idea, top_k=top_k)

    profiler = cProfile.Profile()
    profiler.enable()

    try:
        return pipeline.run(idea, top_k=top_k)
    finally:
        profiler.disable()

        output = io.StringIO()
        stats = pstats.Stats(profiler, stream=output)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(30)

        print("\nPipeline Profile", file=sys.stderr)
        print("================", file=sys.stderr)
        print(
            "Top 30 functions by cumulative execution time:",
            file=sys.stderr,
        )
        print(output.getvalue(), file=sys.stderr)

def main() -> int:
    timings = TimingTracker()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with timings.measure("argument_parsing"):
        args = build_parser().parse_args()
        idea = read_research_idea(args.idea)

    if not idea:
        print("Error: a research idea is required.", file=sys.stderr)
        if args.show_timings:
            timings.print_report()
        return 2

    if not MIN_PAPER_LIMIT <= args.limit <= MAX_PAPER_LIMIT:
        print(
            f"Error: --limit must be between {MIN_PAPER_LIMIT} "
            f"and {MAX_PAPER_LIMIT}.",
            file=sys.stderr,
        )
        if args.show_timings:
            timings.print_report()
        return 2

    try:
        with timings.measure("settings_load"):
            settings = Settings.from_env()

        if args.limit > settings.openalex.max_candidates:
            raise ConfigurationError(
                "--limit cannot exceed RESEARCH_GAP_MAX_CANDIDATES "
                f"({settings.openalex.max_candidates})"
            )

        with timings.measure("pipeline_build"):
            pipeline = build_pipeline(args, settings)

        with timings.measure("pipeline_run"):
            result = run_pipeline(
                pipeline,
                idea,
                args.limit,
                profile=args.profile,
            )

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
        if args.show_timings:
            timings.print_report()
        return 1

    with timings.measure("reporting"):
        handle_retrieval_failures(result)

        for notice in result.notices:
            print(f"Notice: {notice}", file=sys.stderr)

        for notice in result.analysis_notices:
            print(f"Analysis notice: {notice}", file=sys.stderr)

        if args.json:
            payload: Any

            if args.show_queries:
                payload = result.to_dict()
            elif args.show_evidence or args.show_gaps or args.show_landscape:
                payload = result.to_dict()
            else:
                payload = [_paper_json(paper) for paper in result.papers]

            print(json.dumps(payload, ensure_ascii=False, indent=2))

        else:
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

            if args.show_gaps:
                print_gaps(result)

            if args.show_landscape:
                print(f"\nRanked papers shown: {len(result.papers)}")
                print(
                    format_landscape(result.landscape)
                    if result.landscape
                    else "No literature landscape was generated."
                )

    if args.show_timings:
        timings.print_report(result.work_metrics, result.stage_timings)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
