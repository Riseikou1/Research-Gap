"""Stable identifiers for Milestone 3 retrieval ablations."""

from enum import StrEnum


class AblationVariant(StrEnum):
    ORIGINAL_ONLY = "original_only"
    DETERMINISTIC_EXPANSION = "deterministic_expansion"
    LLM_EXPANSION = "llm_expansion"
    COMBINED_EXPANSION = "combined_expansion"
    HYBRID_RETRIEVAL = "hybrid_retrieval"
    HYBRID_RERANKED = "hybrid_reranked"
