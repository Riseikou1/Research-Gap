from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from src.analysis.landscape import LandscapeAnalyzer
from src.api.routes.analyses import _markdown_report
from src.api.safety import public_analysis_result
from src.extraction.evidence import (
    EvidenceItem,
    ExtractionCoverage,
    PaperCoverageRecord,
    PaperEvidence,
)
from src.pipeline import ResearchPipeline
from src.query.deterministic import DeterministicDecomposer
from src.ranking.lexical import LexicalScorer
from src.ranking.reranker import HybridReranker
from src.retrieval.multi_query import MultiQueryRetriever
from src.retrieval.openalex import _parse_work


FIXTURE_PATH = (
    Path(__file__).parents[1] / "fixtures" / "production_openalex_aliases.json"
)


class ProductionRouteRetriever:
    provider_name = "openalex"

    def __init__(self) -> None:
        raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["works"]
        self.works = [_parse_work(item) for item in raw]

    def search(self, request):
        route_indexes = {
            "broad_lexical": [0, 3, 5, 7],
            "title_abstract": [1, 4, 6, 7],
            "semantic": [2],
        }
        result = []
        for rank, index in enumerate(route_indexes[request.mode.value], 1):
            paper = self.works[index].model_copy(deep=True)
            from src.models.paper import RetrievalProvenance

            paper.provenance = [RetrievalProvenance(
                query=request.query,
                provider=self.provider_name,
                mode=request.mode,
                retrieved_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
                provider_rank=rank,
            )]
            result.append(paper)
        return result


class RecordingExtractor:
    failures = []

    def __init__(self) -> None:
        self.coverage_records: list[PaperCoverageRecord] = []
        self.workflows: list[list[str]] = []
        self.location_counts: dict[str, int] = {}

    def extract_many(self, papers, limit=None):
        requested = list(papers)[:limit]
        self.workflows.append([paper.id for paper in requested])
        self.location_counts.update({
            paper.id: len(paper.full_text_locations) for paper in requested
        })
        records = []
        self.coverage_records = []
        for paper in requested:
            coverage = ExtractionCoverage(
                source_level="metadata_only",
                full_text_status="unavailable",
                full_text_requested=True,
                full_text_attempted=False,
                fallback_explanation=(
                    "Open full text was unavailable; title metadata fallback succeeded."
                ),
            )
            records.append(PaperEvidence(
                paper_id=paper.id,
                title=paper.title,
                study_type="other",
                method_or_intervention=[EvidenceItem(
                    value="retrieval augmented generation",
                    canonical_value="retrieval augmented generation",
                    evidence_text="Retrieval Augmented Generation",
                    source="title",
                    confidence=0.9,
                )],
                extraction_confidence=0.8,
                coverage=coverage,
            ))
            self.coverage_records.append(PaperCoverageRecord(
                paper_id=paper.id,
                title=paper.title,
                aliases=[*paper.openalex_aliases, *paper.doi_aliases],
                full_text_requested=True,
                full_text_attempted=False,
                final_evidence_level="metadata_only",
                full_text_status="unavailable",
                fallback_explanation=coverage.fallback_explanation,
                final_state="success",
            ))
        return records


class RecordingReranker:
    def __init__(self, alias) -> None:
        self.delegate = HybridReranker(LexicalScorer(), None)
        self.alias = alias
        self.received: list[str] = []
        self.returned_count = 0

    def rerank(self, idea, papers):
        self.received = [paper.id for paper in papers]
        result = self.delegate.rerank(idea, papers)
        result.papers.append(self.alias.model_copy(deep=True))
        self.returned_count = len(result.papers)
        return result


def test_production_aliases_are_canonical_before_extraction_and_reporting() -> None:
    regression = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert set(regression["source_reports"]) == {
        "naacl_abstract_only_m9_v3.md",
        "naacl_full_text_m9_v3.md",
    }
    assert regression["reported_duplicate_titles"][
        "Reducing hallucination in structured outputs via Retrieval-Augmented Generation"
    ] == 3
    for report_name in regression["source_reports"]:
        assert set(regression["report_regression"][report_name][
            "per_paper_coverage_ids"
        ]) == {
            "https://openalex.org/W4394838812",
            "https://openalex.org/W6966460441",
            "https://openalex.org/W4401042735",
        }
    extractor = RecordingExtractor()
    reranker = RecordingReranker(_parse_work(regression["works"][0]))
    pipeline = ResearchPipeline(
        decomposer=DeterministicDecomposer(),
        retriever=MultiQueryRetriever(
            ProductionRouteRetriever(),
            max_candidates=20,
            per_route_limit=10,
            max_workers=2,
        ),
        reranker=reranker,
        extractor=extractor,
        landscape_analyzer=LandscapeAnalyzer(),
    )

    result = pipeline.run(
        "retrieval augmented generation for enterprise workflows",
        top_k=20,
    )
    naacl_title = (
        "Reducing hallucination in structured outputs via "
        "Retrieval-Augmented Generation"
    )

    assert result.candidate_count == 4
    assert len(reranker.received) == 4
    assert len(set(reranker.received)) == 4
    assert reranker.returned_count == 5
    assert len(result.papers) == 4
    assert len(extractor.workflows) == 1
    assert len(extractor.workflows[0]) == 4
    assert len(set(extractor.workflows[0])) == 4
    naacl = next(paper for paper in result.papers if paper.title == naacl_title)
    assert len(naacl.openalex_aliases) == 3
    assert len(naacl.doi_aliases) == 3
    assert set(naacl.retrieval_modes) == {
        "broad_lexical", "title_abstract", "semantic",
    }
    assert naacl.retrieved_by == ["openalex"]
    assert extractor.location_counts[naacl.id] == 6

    assert result.landscape is not None
    assert result.landscape.total_papers == 4
    rag_frequency = next(
        item
        for item in result.landscape.frequencies
        if item.dimension == "method"
        and item.value == "retrieval augmented generation"
    )
    assert rag_frequency.count == 4
    assert len(set(rag_frequency.paper_ids)) == 4

    raw = result.to_dict()
    raw["full_text_requested"] = True
    public = public_analysis_result(raw)
    assert len(public["papers"]) == 4
    assert len(public["evidence"]) == 4
    assert len(public["paper_coverage"]) == 4
    naacl_coverage = next(
        item for item in public["paper_coverage"] if item["title"] == naacl_title
    )
    assert len(naacl_coverage["aliases"]) == 6
    accounting = public["extraction_coverage"]
    assert accounting["requested_for_extraction"] == (
        accounting["full_text_successes"]
        + accounting["abstract_successes"]
        + accounting["abstract_fallback_successes"]
        + accounting["metadata_only_successes"]
        + accounting["final_failures"]
    )

    markdown = _markdown_report("test idea", "full", raw)
    relevant = markdown.split("## Relevant papers", 1)[1].split(
        "## Coverage limitations", 1
    )[0]
    assert relevant.count(naacl_title) == 1
    coverage = markdown.split("## Per-paper coverage", 1)[1].split(
        "## Relevant papers", 1
    )[0]
    assert coverage.count(naacl_title) == 1
    assert "W4394838812" in coverage
    assert "W6966460441" in coverage
    assert "W4401042735" in coverage
