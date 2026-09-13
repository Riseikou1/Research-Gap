"""Provider-independent descriptive citation graph models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictGraphModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


class CitationGraphNode(_StrictGraphModel):
    paper_id: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    title: str = Field(min_length=1)
    publication_year: int | None = None
    citation_count: int = Field(default=0, ge=0)
    relevance_score: float | None = Field(default=None, ge=0, le=1)
    selected_for_analysis: bool = False
    in_degree: int = Field(default=0, ge=0)
    out_degree: int = Field(default=0, ge=0)
    external_reference_count: int = Field(default=0, ge=0)


class CitationGraphEdge(_StrictGraphModel):
    citing_paper_id: str = Field(min_length=1)
    cited_paper_id: str = Field(min_length=1)


class CitationGraph(_StrictGraphModel):
    status: Literal["available", "unavailable"] = "unavailable"
    scope: Literal["canonical_retrieved_pool"] = "canonical_retrieved_pool"
    nodes: list[CitationGraphNode] = Field(default_factory=list)
    edges: list[CitationGraphEdge] = Field(default_factory=list)
    node_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)
    component_count: int = Field(default=0, ge=0)
    isolated_node_count: int = Field(default=0, ge=0)
    external_reference_count: int = Field(default=0, ge=0)
    limitation: str = (
        "Only citations whose endpoints are in the bounded canonical retrieved pool are shown; "
        "external references are counted but are not recursively retrieved."
    )

    @model_validator(mode="after")
    def validate_accounting(self) -> "CitationGraph":
        if self.node_count != len(self.nodes) or self.edge_count != len(self.edges):
            raise ValueError("citation graph counts must match serialized members")
        node_ids = [node.paper_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("citation graph nodes must be unique")
        node_set = set(node_ids)
        edge_pairs = [(edge.citing_paper_id, edge.cited_paper_id) for edge in self.edges]
        if len(edge_pairs) != len(set(edge_pairs)):
            raise ValueError("citation graph edges must be unique")
        if any(left == right or left not in node_set or right not in node_set for left, right in edge_pairs):
            raise ValueError("citation graph edges must be non-self induced edges")
        return self

    @classmethod
    def unavailable(cls) -> "CitationGraph":
        return cls()
