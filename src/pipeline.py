"""Application service orchestrating Milestone 3 retrieval and reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.models.idea import ResearchIdea
from src.models.paper import Paper
from src.models.query import SearchQuery
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

    def to_dict(self) -> dict[str, object]:
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
        self.evidence_limit = evidence_limit

    def run(self, idea_text: str, *, top_k: int = 20) -> ResearchResult:
        if not 1 <= top_k <= self.retriever.max_candidates:
            raise ValueError(
                f"top_k must be between 1 and {self.retriever.max_candidates}"
            )
        idea = self.decomposer.decompose(idea_text)
        deterministic = self.deterministic_generator.generate(idea)
        llm = self.llm_generator.generate(idea) if self.llm_generator else []
        queries = self.query_planner.plan(idea, deterministic, llm)
        retrieval = self.retriever.retrieve_hybrid(queries)

        if not retrieval.papers and retrieval.failures:
            details = "; ".join(
                f"{failure.mode}/{failure.query}: {failure.error}"
                for failure in retrieval.failures
            )
            raise PipelineError(f"all retrieval routes failed: {details}")

        ranking = self.reranker.rerank(idea.original_text, retrieval.papers)
        notices: list[str] = []
        if ranking.notice:
            notices.append(ranking.notice)
        if not ranking.papers:
            notices.append("No candidate papers were found.")

        selected = ranking.papers[:top_k]
        evidence: list[PaperEvidence] = []
        extraction_failures: list[str] = []
        if self.extractor:
            evidence = self.extractor.extract_many(selected, limit=self.evidence_limit)
            extraction_failures = [str(error) for error in self.extractor.failures]

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
        )
