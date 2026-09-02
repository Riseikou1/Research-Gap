"""Application service orchestrating retrieval through Milestone 5 analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal

from src.models.idea import ResearchIdea
from src.models.paper import Paper
from src.models.query import SearchQuery
from src.analysis.clustering import LandscapeAnalyzer
from src.analysis.gap_candidates import GapCandidateGenerator
from src.analysis.models import GapCandidate, IdeaAssessment
from src.analysis.verification import GapVerifier
from src.models.landscape import LiteratureLandscape
from src.extraction.evidence import PaperEvidence
from src.extraction.paper_extractor import PaperExtractor
from src.query.base import QueryDecomposer, QueryGenerator
from src.query.generator import DeterministicQueryGenerator
from src.query.planner import QueryPlanner
from src.ranking.reranker import HybridReranker
from src.retrieval.multi_query import MultiQueryRetriever, RetrievalFailure


class PipelineError(RuntimeError):
    """Raised when the complete literature pipeline cannot produce a result."""


@dataclass(slots=True)
class ResearchResult:
    idea: ResearchIdea
    queries: list[SearchQuery]
    candidate_count: int
    papers: list[Paper]
    retrieval_failures: list[RetrievalFailure] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    ranking_mode: Literal["hybrid", "lexical_only"] = "lexical_only"
    evidence: list[PaperEvidence] = field(default_factory=list)
    extraction_failures: list[str] = field(default_factory=list)
    gaps: list[GapCandidate] = field(default_factory=list)
    analysis_notices: list[str] = field(default_factory=list)
    landscape: LiteratureLandscape | None = None
    idea_assessment: IdeaAssessment | None = None
    work_metrics: dict[str, int] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        gap_payloads = []
        for item in self.gaps:
            payload = item.model_dump(mode="json")
            # This legacy/internal field is not a novelty probability and is
            # intentionally omitted from user-facing JSON.
            payload.pop("confidence", None)
            payload.pop("idea_relevance", None)
            gap_payloads.append(payload)
        return {
            "idea": self.idea.model_dump(mode="json"),
            "queries": [query.model_dump(mode="json") for query in self.queries],
            "candidate_count": self.candidate_count,
            "papers": [paper.model_dump(mode="json") for paper in self.papers],
            "retrieval_failures": [
                failure.to_dict() for failure in self.retrieval_failures
            ],
            "notices": list(self.notices),
            "ranking_mode": self.ranking_mode,
            "evidence": [item.model_dump(mode="json") for item in self.evidence],
            "extraction_failures": list(self.extraction_failures),
            "gaps": gap_payloads,
            "analysis_notices": list(self.analysis_notices),
            "landscape": self.landscape.model_dump(mode="json") if self.landscape else None,
            "idea_assessment": self.idea_assessment.model_dump(mode="json") if self.idea_assessment else None,
            "work_metrics": dict(self.work_metrics),
            "stage_timings": dict(self.stage_timings),
        }


class ResearchPipeline:
    """Decompose, plan, retrieve, deduplicate, score, and rerank an idea."""

    def __init__(
        self,
        *,
        decomposer: QueryDecomposer,
        retriever: MultiQueryRetriever,
        reranker: HybridReranker,
        deterministic_generator: QueryGenerator | None = None,
        llm_generator: QueryGenerator | None = None,
        query_planner: QueryPlanner | None = None,
        extractor: PaperExtractor | None = None,
        gap_generator: GapCandidateGenerator | None = None,
        gap_verifier: GapVerifier | None = None,
        landscape_analyzer: LandscapeAnalyzer | None = None,
        evidence_limit: int = 10,
    ) -> None:
        self.decomposer = decomposer
        self.retriever = retriever
        self.reranker = reranker
        self.deterministic_generator = (
            deterministic_generator or DeterministicQueryGenerator()
        )
        self.llm_generator = llm_generator
        self.query_planner = query_planner or QueryPlanner()
        self.extractor = extractor
        self.gap_generator = gap_generator
        self.gap_verifier = gap_verifier
        self.landscape_analyzer = landscape_analyzer or LandscapeAnalyzer()
        self.evidence_limit = evidence_limit

    def run(self, idea_text: str, *, top_k: int = 20) -> ResearchResult:
        if not 1 <= top_k <= self.retriever.max_candidates:
            raise ValueError(
                f"top_k must be between 1 and {self.retriever.max_candidates}"
            )
        reset_verifier_metrics = getattr(self.gap_verifier, "reset_metrics", None)
        if reset_verifier_metrics is not None:
            reset_verifier_metrics()

        extractor_before = _component_metrics(self.extractor)
        extractor_timings_before = _component_timings(self.extractor)
        embedding_provider = _embedding_provider(self.reranker)
        embedding_before = _component_metrics(embedding_provider)
        verifier_before = _component_metrics(self.gap_verifier)
        decomposer_before = _component_metrics(self.decomposer)
        generator_before = _component_metrics(self.llm_generator)
        retriever_before = _component_metrics(self.retriever)
        stage_timings: dict[str, float] = {}

        started = perf_counter()
        idea = self.decomposer.decompose(idea_text)
        deterministic = self.deterministic_generator.generate(idea)
        llm = self.llm_generator.generate(idea) if self.llm_generator else []
        queries = self.query_planner.plan(idea, deterministic, llm)
        stage_timings["planning"] = perf_counter() - started

        started = perf_counter()
        retrieval = self.retriever.retrieve_hybrid(queries)
        stage_timings["initial_retrieval"] = perf_counter() - started

        if not retrieval.papers and retrieval.failures:
            details = "; ".join(
                f"{failure.mode}/{failure.query}: {failure.error}"
                for failure in retrieval.failures
            )
            raise PipelineError(f"all retrieval routes failed: {details}")

        started = perf_counter()
        ranking = self.reranker.rerank(idea, retrieval.papers)
        stage_timings["ranking_embeddings"] = perf_counter() - started
        notices: list[str] = []
        if ranking.notice:
            notices.append(ranking.notice)
        if not ranking.papers:
            notices.append("No candidate papers were found.")

        selected = ranking.papers[:top_k]
        evidence: list[PaperEvidence] = []
        extraction_failures: list[str] = []
        gaps: list[GapCandidate] = []
        analysis_notices: list[str] = []
        landscape: LiteratureLandscape | None = None
        idea_assessment: IdeaAssessment | None = None
        if self.extractor:
            get_many = getattr(
                self.extractor,
                "get_many_or_extract",
                self.extractor.extract_many,
            )
            started = perf_counter()
            evidence = get_many(selected, limit=self.evidence_limit)
            stage_timings["evidence_lookup_extraction"] = perf_counter() - started
            extractor_timings_after = _component_timings(self.extractor)
            stage_timings["initial_evidence_extraction_api_wait"] = max(
                0.0,
                extractor_timings_after.get(
                    "initial_evidence_extraction_api_wait",
                    0.0,
                )
                - extractor_timings_before.get(
                    "initial_evidence_extraction_api_wait",
                    0.0,
                ),
            )
            extraction_failures = [str(error) for error in self.extractor.failures]
            started = perf_counter()
            landscape = self.landscape_analyzer.analyze(evidence, selected)
            stage_timings["landscape"] = perf_counter() - started
            if self.gap_verifier:
                prime_evidence = getattr(self.gap_verifier, "prime_evidence", None)
                if prime_evidence is not None:
                    prime_evidence(selected, evidence)
                # The complete idea is a first-class Milestone-6 subject. It
                # is assessed before candidate generation and even when the
                # generator later returns no narrower hypotheses.
                started = perf_counter()
                idea_assessment = self.gap_verifier.assess_idea(idea, landscape, evidence)
                stage_timings["direct_verification"] = perf_counter() - started
            if self.gap_generator:
                started = perf_counter()
                gaps = self.gap_generator.generate(idea, landscape, evidence)
                stage_timings["candidate_generation"] = perf_counter() - started
                analysis_notices.extend(self.gap_generator.notices)
                if self.gap_verifier:
                    started = perf_counter()
                    gaps = self.gap_verifier.verify_many(idea, gaps, evidence)
                    stage_timings["candidate_verification"] = perf_counter() - started
                    analysis_notices.extend(self.gap_verifier.notices)
                else:
                    if gaps:
                        analysis_notices.append("verification_skipped: candidate gaps require targeted counterexample verification")
                    analysis_notices.append("idea_assessment_skipped: direct idea verification requires a verifier")

        work_metrics = _build_work_metrics(
            retrieval_papers=retrieval.papers,
            selected_papers=selected,
            extractor_before=extractor_before,
            extractor_after=_component_metrics(self.extractor),
            embedding_before=embedding_before,
            embedding_after=_component_metrics(embedding_provider),
            verifier_before=verifier_before,
            verifier_after=_component_metrics(self.gap_verifier),
            generator_after=_component_metrics(self.gap_generator),
            displayed_candidates=len(gaps),
            decomposer_before=decomposer_before,
            decomposer_after=_component_metrics(self.decomposer),
            query_generator_before=generator_before,
            query_generator_after=_component_metrics(self.llm_generator),
            retriever_before=retriever_before,
            retriever_after=_component_metrics(self.retriever),
        )
        stage_timings.update(_component_timings(self.gap_verifier))

        return ResearchResult(
            idea=idea,
            queries=queries,
            candidate_count=len(retrieval.papers),
            papers=selected,
            retrieval_failures=retrieval.failures,
            notices=notices,
            ranking_mode=ranking.mode,
            evidence=evidence,
            extraction_failures=extraction_failures,
            gaps=gaps,
            analysis_notices=analysis_notices,
            landscape=landscape,
            idea_assessment=idea_assessment,
            work_metrics=work_metrics,
            stage_timings=stage_timings,
        )


def _component_metrics(component) -> dict[str, int]:
    if component is None:
        return {}
    snapshot = getattr(component, "metrics_snapshot", None)
    if snapshot is None:
        return {}
    try:
        value = snapshot()
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _component_timings(component) -> dict[str, float]:
    if component is None:
        return {}
    snapshot = getattr(component, "timings_snapshot", None)
    if snapshot is None:
        return {}
    try:
        value = snapshot()
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _embedding_provider(reranker: HybridReranker):
    semantic = getattr(reranker, "semantic_scorer", None)
    return getattr(semantic, "provider", None)


def _metric_delta(
    before: dict[str, int],
    after: dict[str, int],
    key: str,
) -> int:
    return max(0, after.get(key, 0) - before.get(key, 0))


def _build_work_metrics(
    *,
    retrieval_papers: list[Paper],
    selected_papers: list[Paper],
    extractor_before: dict[str, int],
    extractor_after: dict[str, int],
    embedding_before: dict[str, int],
    embedding_after: dict[str, int],
    verifier_before: dict[str, int],
    verifier_after: dict[str, int],
    generator_after: dict[str, int],
    displayed_candidates: int,
    decomposer_before: dict[str, int],
    decomposer_after: dict[str, int],
    query_generator_before: dict[str, int],
    query_generator_after: dict[str, int],
    retriever_before: dict[str, int],
    retriever_after: dict[str, int],
) -> dict[str, int]:
    if "evidence_requested" in extractor_before or "evidence_requested" in extractor_after:
        evidence_requested = _metric_delta(
            extractor_before,
            extractor_after,
            "evidence_requested",
        )
    else:
        # Preserve useful accounting for lightweight/custom extractors that
        # implement the legacy extract_many interface without metrics.
        evidence_requested = len(selected_papers)

    evidence_requested += _metric_delta(
        verifier_before,
        verifier_after,
        "verification_evidence_requested",
    )

    return {
        "retrieved_papers": len(retrieval_papers),
        "unique_scholarly_works": len(retrieval_papers),
        "candidate_hypotheses_generated": generator_after.get(
            "candidate_hypotheses_generated",
            0,
        ),
        "candidate_hypotheses_after_pruning": verifier_after.get(
            "candidate_hypotheses_after_pruning",
            generator_after.get("candidate_hypotheses_after_pruning", 0),
        ),
        "candidate_hypotheses_verified": verifier_after.get(
            "candidate_hypotheses_verified",
            0,
        ),
        "candidate_hypotheses_displayed": displayed_candidates,
        "evidence_requested": evidence_requested,
        "memory_cache_hits": _metric_delta(
            extractor_before,
            extractor_after,
            "memory_cache_hits",
        ),
        "memory_evidence_cache_hits": _metric_delta(
            extractor_before,
            extractor_after,
            "memory_cache_hits",
        ),
        "persistent_cache_hits": _metric_delta(
            extractor_before,
            extractor_after,
            "persistent_cache_hits",
        ),
        "persistent_evidence_cache_hits": _metric_delta(
            extractor_before,
            extractor_after,
            "persistent_cache_hits",
        ),
        "new_evidence_extractions": _metric_delta(
            extractor_before,
            extractor_after,
            "new_evidence_extractions",
        ),
        "openai_extraction_requests": _metric_delta(
            extractor_before,
            extractor_after,
            "openai_extraction_requests",
        ),
        "verification_queries_executed": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_queries_executed",
        ),
        "verification_queries_planned": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_queries_planned",
        ),
        "verification_queries_cache_hits": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_queries_cache_hits",
        ),
        "verification_queries_skipped_redundant": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_queries_skipped_redundant",
        ),
        "verification_candidates_early_stopped": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_candidates_early_stopped",
        ),
        "verification_candidates_reused": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_candidates_reused",
        ),
        "candidate_verification_results_reused_exact": _metric_delta(
            verifier_before,
            verifier_after,
            "candidate_verification_results_reused_exact",
        ),
        "candidate_verification_results_reused_nonidentical": _metric_delta(
            verifier_before,
            verifier_after,
            "candidate_verification_results_reused_nonidentical",
        ),
        "observed_candidates_rejected_preverification": _metric_delta(
            verifier_before,
            verifier_after,
            "observed_candidates_rejected_preverification",
        ),
        "candidate_rejected_unanchored": _metric_delta(
            verifier_before,
            verifier_after,
            "candidate_rejected_unanchored",
        ),
        "candidate_rejected_observed": _metric_delta(
            verifier_before,
            verifier_after,
            "candidate_rejected_observed",
        ),
        "candidate_rejected_dominated": _metric_delta(
            verifier_before,
            verifier_after,
            "candidate_rejected_dominated",
        ),
        "candidate_rejected_invalid_role": _metric_delta(
            verifier_before,
            verifier_after,
            "candidate_rejected_invalid_role",
        ),
        "verification_unique_papers": verifier_after.get(
            "verification_unique_papers",
            0,
        ),
        "verification_papers_already_cached": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_papers_already_cached",
        ),
        "new_verification_extractions": _metric_delta(
            verifier_before,
            verifier_after,
            "new_verification_extractions",
        ),
        "verification_papers_rejected_by_prescreen": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_papers_rejected_by_prescreen",
        ),
        "verification_papers_prescreen_accepted": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_papers_prescreen_accepted",
        ),
        "verification_papers_prescreen_uncertain": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_papers_prescreen_uncertain",
        ),
        "prescreen_rejections": _metric_delta(
            verifier_before,
            verifier_after,
            "verification_papers_rejected_by_prescreen",
        ),
        "embedding_cache_hits": _metric_delta(
            embedding_before,
            embedding_after,
            "embedding_cache_hits",
        ),
        "new_embeddings": _metric_delta(
            embedding_before,
            embedding_after,
            "new_embeddings",
        ),
        "embedding_requests": _metric_delta(
            embedding_before,
            embedding_after,
            "embedding_requests",
        ),
        "openai_decomposition_requests": _metric_delta(
            decomposer_before,
            decomposer_after,
            "openai_decomposition_requests",
        ),
        "openai_query_generation_requests": _metric_delta(
            query_generator_before,
            query_generator_after,
            "openai_query_generation_requests",
        ),
        # Decomposition and query generation intentionally remain separate to
        # preserve deterministic/LLM mixed modes and independent fallbacks.
        "openai_combined_planning_requests": 0,
        "planning_cache_hits": (
            _metric_delta(decomposer_before, decomposer_after, "planning_cache_hits")
            + _metric_delta(query_generator_before, query_generator_after, "planning_cache_hits")
        ),
        "decomposition_cache_hits": _metric_delta(
            decomposer_before, decomposer_after, "decomposition_cache_hits"
        ),
        "query_generation_cache_hits": _metric_delta(
            query_generator_before, query_generator_after, "query_generation_cache_hits"
        ),
        "retrieval_cache_hits": _metric_delta(
            retriever_before, retriever_after, "retrieval_cache_hits"
        ),
        "retrieval_cache_misses": _metric_delta(
            retriever_before, retriever_after, "retrieval_cache_misses"
        ),
        "retrieval_provider_requests": _metric_delta(
            retriever_before, retriever_after, "retrieval_provider_requests"
        ),
        "verification_retrieval_cache_hits": _metric_delta(
            retriever_before, retriever_after, "verification_retrieval_cache_hits"
        ),
        "retrieval_failure_cache_hits": _metric_delta(
            retriever_before, retriever_after, "retrieval_failure_cache_hits"
        ),
    }
