from __future__ import annotations

from src.analysis.citation_graph import build_citation_graph
from src.models.paper import Paper
from src.retrieval.deduplication import deduplicate_paper_models
from src.retrieval.openalex import OpenAlexRetriever, _parse_work
from src.retrieval.base import RetrievalRequest
from src.models.query import RetrievalMode, SearchQuery
from src.api.routes.analyses import _markdown_report
from src.persistence.database import Database
from src.persistence.models import NewAnalysis
from src.persistence.repositories import AnalysisRepository
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch

from src.models.paper import RetrievalProvenance
from src.pipeline import ResearchPipeline
from src.query.deterministic import DeterministicDecomposer
from src.ranking.lexical import LexicalScorer
from src.ranking.reranker import HybridReranker
from src.retrieval.multi_query import MultiQueryRetriever


def _paper(identifier: str, title: str, *, references: list[str] | None = None) -> Paper:
    return Paper(
        id=f"https://openalex.org/{identifier}",
        openalex_id=f"https://openalex.org/{identifier}",
        openalex_aliases=[f"https://openalex.org/{identifier}"],
        title=title,
        referenced_work_ids=references or [],
    )


def test_openalex_reference_parsing_is_bounded_and_malformed_safe() -> None:
    paper = _parse_work({
        "id": "https://openalex.org/W1",
        "display_name": "NAACL fixture paper",
        "referenced_works": [
            "https://openalex.org/W2", "https://openalex.org/w2/", "bad", None, "W3",
        ],
    })
    assert paper.referenced_work_ids == ["https://openalex.org/W2", "https://openalex.org/W3"]


def test_graph_resolves_aliases_deduplicates_edges_and_preserves_direction() -> None:
    naacl = _paper("W1", "NAACL RAG paper", references=[
        "https://openalex.org/W2", "W2", "https://openalex.org/W1", "W999", "not-an-id",
    ])
    cited = _paper("W2", "Earlier baseline")
    duplicate = Paper(
        id="doi:duplicate", title="NAACL RAG paper", doi="10.1/naacl",
        doi_aliases=["10.1/naacl"], openalex_aliases=["https://openalex.org/W1"],
        referenced_work_ids=["https://openalex.org/W2"],
    )
    naacl.doi = "10.1/naacl"
    naacl.doi_aliases = ["10.1/naacl"]
    isolated = _paper("W3", "Unconnected paper")

    canonical = deduplicate_paper_models([duplicate, cited, naacl, isolated])
    graph = build_citation_graph(canonical, selected_paper_ids={canonical[0].id})

    assert graph.node_count == 3
    assert graph.edge_count == 1
    assert [(edge.citing_paper_id, edge.cited_paper_id) for edge in graph.edges] == [
        (next(item.id for item in canonical if item.openalex_id and item.openalex_id.endswith("W1")), cited.id)
    ]
    assert graph.isolated_node_count == 1
    assert graph.component_count == 2
    assert graph.external_reference_count == 1
    assert sum(node.out_degree for node in graph.nodes) == 1
    assert sum(node.in_degree for node in graph.nodes) == 1


def test_graph_is_deterministic_and_does_not_mutate_scores() -> None:
    citing = _paper("W20", "Citing", references=["W10"])
    cited = _paper("W10", "Cited")
    citing.final_score = 0.8
    cited.final_score = 0.2
    before = [(paper.id, paper.final_score) for paper in (citing, cited)]

    first = build_citation_graph([citing, cited], selected_paper_ids={citing.id})
    second = build_citation_graph([cited, citing], selected_paper_ids={citing.id})

    assert first.model_dump() == second.model_dump()
    assert [(paper.id, paper.final_score) for paper in (citing, cited)] == before


def test_old_analysis_graph_default_is_unavailable() -> None:
    from src.api.safety import public_analysis_result

    graph = public_analysis_result({"papers": [], "evidence": []})["citation_graph"]
    assert graph["status"] == "unavailable"
    assert graph["nodes"] == []


def test_graph_is_hard_bounded_even_for_an_oversized_direct_input() -> None:
    papers = [_paper(f"W{index}", f"Paper {index}") for index in range(1, 12)]
    graph = build_citation_graph(papers, selected_paper_ids=set(), max_nodes=5)
    assert graph.node_count == 5


def test_graph_survives_result_json_and_markdown_export(tmp_path) -> None:
    graph = build_citation_graph(
        [_paper("W2", "Citing paper", references=["W1"]), _paper("W1", "Cited paper")],
        selected_paper_ids={"https://openalex.org/W2"},
    )
    database = Database(tmp_path / "graph-persistence.sqlite3")
    database.migrate()
    repository = AnalysisRepository(database)
    repository.create(NewAnalysis(
        analysis_id="graph-analysis", research_idea="citation graph",
        decomposer="deterministic", query_generator="deterministic", paper_limit=2,
        configuration={}, owner_kind="local", owner_id="local", mode="full",
    ))
    assert repository.mark_running("graph-analysis")
    payload = {"mode": "full", "papers": [], "evidence": [], "citation_graph": graph.model_dump(mode="json")}
    assert repository.mark_completed("graph-analysis", payload)
    loaded = repository.get("graph-analysis")
    assert loaded is not None and loaded.result is not None
    assert loaded.result["citation_graph"] == graph.model_dump(mode="json")
    markdown = _markdown_report("citation graph", "full", loaded.result)
    assert "## Citation context" in markdown
    assert "Citing paper cites Cited paper" in markdown


def test_existing_naacl_alias_fixture_collapses_before_graph_construction() -> None:
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "production_openalex_aliases.json").read_text()
    )
    works = fixture["works"][:3]
    works[0]["referenced_works"] = ["https://openalex.org/W999999"]
    works[1]["referenced_works"] = ["https://openalex.org/W4401042735"]
    works[2]["referenced_works"] = ["https://openalex.org/W4394838812"]
    papers = deduplicate_paper_models([_parse_work(work) for work in works])

    graph = build_citation_graph(papers, selected_paper_ids={papers[0].id})

    assert graph.node_count == 1
    assert graph.edge_count == 0
    assert graph.isolated_node_count == 1
    assert graph.external_reference_count == 1


def test_quick_pipeline_does_not_call_graph_construction() -> None:
    class Retriever:
        provider_name = "fake"

        def search(self, request):
            paper = _paper("W1", "One paper")
            paper.provenance = [RetrievalProvenance(
                query=request.query, provider="fake", mode=request.mode,
                retrieved_at=datetime.now(timezone.utc), provider_rank=1,
            )]
            return [paper]

    pipeline = ResearchPipeline(
        decomposer=DeterministicDecomposer(),
        retriever=MultiQueryRetriever(Retriever(), max_candidates=10, per_route_limit=5),
        reranker=HybridReranker(LexicalScorer(), None),
        include_citation_graph=False,
    )
    with patch("src.pipeline.build_citation_graph", side_effect=AssertionError("must not run")):
        result = pipeline.run("retrieval augmented generation", top_k=1)
    assert result.citation_graph.status == "unavailable"

    quick_openalex = OpenAlexRetriever(include_references=False)
    parameters = quick_openalex._build_params(
        RetrievalRequest(
            query=SearchQuery(text="retrieval", strategy="original", source="deterministic"),
            mode=RetrievalMode.BROAD_LEXICAL, limit=5,
        ),
        per_page=5,
    )
    assert "referenced_works" not in parameters["select"]


def test_graph_toggle_does_not_change_scientific_outputs() -> None:
    class Retriever:
        provider_name = "fake"

        def search(self, request):
            papers = [_paper("W2", "Citing paper", references=["W1"]), _paper("W1", "Cited paper")]
            for index, paper in enumerate(papers, 1):
                paper.provenance = [RetrievalProvenance(
                    query=request.query, provider="fake", mode=request.mode,
                    retrieved_at=datetime(2026, 1, 1, tzinfo=timezone.utc), provider_rank=index,
                )]
            return papers

    def run(include_graph: bool):
        return ResearchPipeline(
            decomposer=DeterministicDecomposer(),
            retriever=MultiQueryRetriever(Retriever(), max_candidates=10, per_route_limit=5),
            reranker=HybridReranker(LexicalScorer(), None),
            include_citation_graph=include_graph,
        ).run("retrieval augmented generation", top_k=2)

    with_graph, without_graph = run(True), run(False)
    assert with_graph.candidate_count == without_graph.candidate_count
    assert [(paper.id, paper.final_score) for paper in with_graph.papers] == [
        (paper.id, paper.final_score) for paper in without_graph.papers
    ]
    assert with_graph.evidence == without_graph.evidence
    assert with_graph.gaps == without_graph.gaps
