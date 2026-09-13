"""Deterministic induced citation graph construction."""

from __future__ import annotations

import networkx as nx
import re

from src.models.citation_graph import CitationGraph, CitationGraphEdge, CitationGraphNode
from src.models.paper import Paper
from src.retrieval.deduplication import deduplicate_paper_models, normalize_openalex_id


def build_citation_graph(
    papers: list[Paper], *, selected_paper_ids: set[str], max_nodes: int = 500,
) -> CitationGraph:
    """Build a bounded graph without retrieval or scientific-score side effects."""

    if not 1 <= max_nodes <= 500:
        raise ValueError("citation graph max_nodes must be between 1 and 500")
    canonical = deduplicate_paper_models(papers)[:max_nodes]
    if not canonical:
        return CitationGraph(status="available")

    alias_to_id: dict[str, str] = {}
    for paper in canonical:
        for value in [paper.openalex_id, *paper.openalex_aliases]:
            alias = _valid_openalex_alias(value)
            if alias:
                alias_to_id.setdefault(alias, paper.id)

    graph = nx.DiGraph()
    graph.add_nodes_from(paper.id for paper in canonical)
    external_by_node: dict[str, set[str]] = {paper.id: set() for paper in canonical}
    for paper in canonical:
        own_aliases = {
            alias for alias in (
                _valid_openalex_alias(value)
                for value in [paper.openalex_id, *paper.openalex_aliases]
            ) if alias
        }
        for reference in paper.referenced_work_ids:
            alias = _valid_openalex_alias(reference)
            if not alias or alias in own_aliases:
                continue
            cited_id = alias_to_id.get(alias)
            if cited_id is None:
                external_by_node[paper.id].add(alias)
            elif cited_id != paper.id:
                graph.add_edge(paper.id, cited_id)

    nodes = [
        CitationGraphNode(
            paper_id=paper.id,
            aliases=sorted(
                set([*paper.openalex_aliases, *paper.doi_aliases]), key=str.casefold,
            ),
            title=paper.title,
            publication_year=paper.publication_year,
            citation_count=paper.citation_count,
            relevance_score=paper.final_score,
            selected_for_analysis=paper.id in selected_paper_ids,
            in_degree=graph.in_degree(paper.id),
            out_degree=graph.out_degree(paper.id),
            external_reference_count=len(external_by_node[paper.id]),
        )
        for paper in sorted(canonical, key=lambda item: item.id.casefold())
    ]
    edges = [
        CitationGraphEdge(citing_paper_id=left, cited_paper_id=right)
        for left, right in sorted(graph.edges(), key=lambda pair: (pair[0].casefold(), pair[1].casefold()))
    ]
    components = nx.number_weakly_connected_components(graph) if nodes else 0
    isolated = sum(graph.degree(node.paper_id) == 0 for node in nodes)
    return CitationGraph(
        status="available",
        nodes=nodes,
        edges=edges,
        node_count=len(nodes),
        edge_count=len(edges),
        component_count=components,
        isolated_node_count=isolated,
        external_reference_count=sum(len(values) for values in external_by_node.values()),
    )


def _valid_openalex_alias(value: str | None) -> str | None:
    alias = normalize_openalex_id(value)
    return alias if alias and re.fullmatch(r"w[1-9][0-9]*", alias) else None
