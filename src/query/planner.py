"""Deterministic merging and bounding of generated search queries."""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.models.idea import ResearchIdea
from src.models.query import SearchQuery
from src.query.deterministic import clean_idea_text


MAX_PLANNED_QUERIES = 6
MAX_LLM_QUERIES = 3

_MIN_MEANINGFUL_TERMS = 2
_BOOLEAN_OPERATORS = {"and", "or"}


class QueryPlanner:
    """Combine deterministic and optional LLM queries without losing origins.

    The planner always keeps the user's original research idea first, then
    prioritizes up to two deterministic reformulations followed by up to three
    unique LLM-generated queries. Remaining deterministic queries may fill any
    unused capacity.
    """

    def __init__(self, *, max_queries: int = MAX_PLANNED_QUERIES) -> None:
        if not 1 <= max_queries <= MAX_PLANNED_QUERIES:
            raise ValueError(
                f"max_queries must be between 1 and {MAX_PLANNED_QUERIES}"
            )
        self.max_queries = max_queries

    def plan(
        self,
        idea: ResearchIdea,
        deterministic_queries: Sequence[SearchQuery],
        llm_queries: Sequence[SearchQuery] = (),
    ) -> list[SearchQuery]:
        """Return a bounded, deduplicated query plan.

        Selection priority is:

        1. Original user query.
        2. Up to two unique deterministic reformulations.
        3. Up to ``MAX_LLM_QUERIES`` unique LLM reformulations.
        4. Remaining deterministic queries until ``max_queries`` is reached.

        Queries with equivalent comparison keys are merged so provenance is
        preserved across generators.
        """
        original = SearchQuery(
            text=clean_idea_text(idea.original_text),
            strategy="original",
            source="deterministic",
        )

        valid_deterministic = _valid_nonoriginal(
            deterministic_queries,
            original,
        )
        valid_llm = _valid_nonoriginal(
            llm_queries,
            original,
        )

        deterministic_keys = _unique_keys(valid_deterministic)

        # Deduplicate before applying the LLM limit. Otherwise repeated LLM
        # queries near the front of the list could consume the quota and hide
        # later, genuinely distinct reformulations.
        llm_keys = _unique_keys(valid_llm)[:MAX_LLM_QUERIES]

        merged = _merge_queries(
            original,
            deterministic_queries,
            llm_queries,
        )

        selected_keys: list[str] = [original.comparison_key]

        for key in deterministic_keys[:2]:
            _append_key(selected_keys, key, self.max_queries)

        for key in llm_keys:
            _append_key(selected_keys, key, self.max_queries)

        for key in deterministic_keys[2:]:
            _append_key(selected_keys, key, self.max_queries)

        return [merged[key] for key in selected_keys]


def _merge_queries(
    original: SearchQuery,
    deterministic_queries: Sequence[SearchQuery],
    llm_queries: Sequence[SearchQuery],
) -> dict[str, SearchQuery]:
    """Merge equivalent queries while preserving their provenance."""
    merged: dict[str, SearchQuery] = {
        original.comparison_key: original,
    }

    for query in (*deterministic_queries, *llm_queries):
        key = query.comparison_key

        if key in merged:
            merged[key] = merged[key].merged_with(query)
        else:
            merged[key] = query

    return merged


def _valid_nonoriginal(
    queries: Sequence[SearchQuery],
    original: SearchQuery,
) -> list[SearchQuery]:
    """Remove original-equivalent and overly weak generated queries."""
    return [
        query
        for query in queries
        if query.comparison_key != original.comparison_key
        and _meaningful_term_count(query.text) >= _MIN_MEANINGFUL_TERMS
    ]


def _unique_keys(queries: Sequence[SearchQuery]) -> list[str]:
    """Return comparison keys in first-seen order without duplicates."""
    return list(
        dict.fromkeys(query.comparison_key for query in queries)
    )


def _append_key(values: list[str], key: str, limit: int) -> None:
    """Append a query key if capacity remains and it is not already selected."""
    if len(values) < limit and key not in values:
        values.append(key)


def _meaningful_term_count(text: str) -> int:
    """Count non-boolean search terms in a query."""
    return sum(
        1
        for token in re.findall(r"[\w+#.-]+", text, flags=re.UNICODE)
        if token.casefold() not in _BOOLEAN_OPERATORS
    )