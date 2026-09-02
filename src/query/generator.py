"""Bounded deterministic lexical-query generation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.models.idea import ResearchIdea
from src.models.query import SearchQuery
from src.query.deterministic import clean_idea_text


MAX_QUERIES = 6

# Prevent individual generated queries from becoming giant bags of terms.
MAX_COMBINED_TERMS = 6
MAX_KEYWORD_TERMS = 6

# Synonym expansion must also remain bounded.
MAX_SYNONYM_GROUPS = 3
MAX_SYNONYMS_PER_GROUP = 4

_TERM_RE = re.compile(r"[\w+#.-]+", re.UNICODE)


def generate_queries(
    idea: ResearchIdea,
    max_queries: int = MAX_QUERIES,
) -> list[str]:
    """
    Generate a small, deterministic set of complementary lexical queries.

    Query strategies, in priority order:

    1. Cleaned original research idea.
    2. Method + problem.
    3. Problem + population/domain.
    4. Method + outcome.
    5. Explicit synonym-expanded Boolean query.
    6. Keyword baseline.

    Queries are bounded, normalized, and deduplicated.
    """

    return [
        query.text
        for query in DeterministicQueryGenerator(
            max_queries=max_queries,
        ).generate(idea)
    ]


class DeterministicQueryGenerator:
    """Always-available, provenance-preserving query generator."""

    def __init__(self, *, max_queries: int = MAX_QUERIES) -> None:
        if not 1 <= max_queries <= MAX_QUERIES:
            raise ValueError(
                f"max_queries must be between 1 and {MAX_QUERIES}"
            )
        self.max_queries = max_queries

    def generate(self, idea: ResearchIdea) -> list[SearchQuery]:
        original = clean_idea_text(idea.original_text)
        candidates: list[tuple[str, str]] = [
            (original, "original"),
            (
                _combine(
                    idea.intervention_or_method,
                    [*idea.data_or_modality, *idea.problem],
                ),
                "method_problem",
            ),
            (
                _combine(
                    [*idea.problem, *idea.data_or_modality],
                    [*idea.population, *idea.domain],
                ),
                "problem_context",
            ),
            (
                _combine(
                    idea.intervention_or_method,
                    [*idea.data_or_modality, *idea.outcomes],
                ),
                "method_outcome",
            ),
            (_synonym_query(idea.synonyms), "synonym_expansion"),
        ]

        if idea.keywords:
            candidates.append(
                (_keyword_query(idea.keywords), "keyword_baseline")
            )

        finalized = _finalize_queries(
            (text for text, _ in candidates),
            original=original,
            max_queries=self.max_queries,
        )
        strategies: dict[str, str] = {}
        for text, strategy in candidates:
            normalized = _normalize_query(text)
            if normalized:
                strategies.setdefault(_query_key(normalized), strategy)

        return [
            SearchQuery(
                text=text,
                strategy=strategies[_query_key(text)],
                source="deterministic",
            )
            for text in finalized
        ]


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _combine(
    left: list[str],
    right: list[str],
    *,
    max_terms: int = MAX_COMBINED_TERMS,
) -> str:
    """
    Combine two facet groups while preserving phrases and removing duplicates.

    Example:

        ["reinforcement learning"]
        +
        ["medical QA", "medical QA"]

    becomes:

        reinforcement learning medical QA
    """

    terms = _unique_terms(
        [*left, *right],
        limit=max_terms,
    )

    return " ".join(terms)


def _keyword_query(
    keywords: list[str],
    *,
    max_terms: int = MAX_KEYWORD_TERMS,
) -> str:
    """
    Build a complementary query while preserving multi-word keyword phrases.

    Phrase-level keywords are more useful than exploding every concept into
    isolated tokens.
    """

    terms = _unique_terms(
        keywords,
        limit=max_terms,
    )

    return " ".join(terms)


# ---------------------------------------------------------------------------
# Synonym query
# ---------------------------------------------------------------------------


def _synonym_query(
    synonyms: dict[str, list[str]],
    *,
    max_groups: int = MAX_SYNONYM_GROUPS,
    max_terms_per_group: int = MAX_SYNONYMS_PER_GROUP,
) -> str:
    """
    Build one bounded synonym-expanded Boolean query.

    Example:

        {
            "retrieval augmented generation": ["RAG"],
            "low-rank adaptation": ["LoRA"]
        }

    becomes:

        ("retrieval augmented generation" OR RAG)
        AND
        ("low-rank adaptation" OR LoRA)
    """

    groups: list[str] = []

    for canonical, alternatives in synonyms.items():
        terms = _unique_terms(
            [canonical, *alternatives],
            limit=max_terms_per_group,
        )

        # A group with only one term provides no synonym expansion.
        if len(terms) < 2:
            continue

        quoted = [
            _quote(term)
            for term in terms
        ]

        groups.append(
            f"({' OR '.join(quoted)})"
        )

        if len(groups) >= max_groups:
            break

    return " AND ".join(groups)


def _quote(term: str) -> str:
    """
    Quote multi-word terms for Boolean query construction.

    Examples:

        RAG
            -> RAG

        retrieval augmented generation
            -> "retrieval augmented generation"
    """

    normalized = _normalize_term(term)

    escaped = normalized.replace(
        '"',
        '\\"',
    )

    if _term_count(escaped) > 1:
        return f'"{escaped}"'

    return escaped


# ---------------------------------------------------------------------------
# Final validation / deduplication
# ---------------------------------------------------------------------------


def _finalize_queries(
    candidates: Iterable[str],
    *,
    original: str,
    max_queries: int,
) -> list[str]:
    """
    Normalize, validate, deduplicate, and bound generated queries.
    """

    queries: list[str] = []
    seen: set[str] = set()

    original_key = _query_key(original)
    original_is_short = _term_count(original) < 2

    for candidate in candidates:
        cleaned = _normalize_query(candidate)

        if not cleaned:
            continue

        key = _query_key(cleaned)

        if key in seen:
            continue

        term_count = _term_count(cleaned)

        # Generated single-term searches tend to be excessively broad.
        #
        # Preserve the original when the user's entire idea genuinely consists
        # of one meaningful term.
        is_original = key == original_key

        if term_count < 2 and not (
            is_original and original_is_short
        ):
            continue

        seen.add(key)
        queries.append(cleaned)

        if len(queries) >= max_queries:
            break

    return queries


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _unique_terms(
    values: Iterable[str],
    *,
    limit: int,
) -> list[str]:
    """
    Normalize and case-insensitively deduplicate semantic terms.

    Original casing is preserved for the first occurrence.
    """

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = _normalize_term(value)

        if not normalized:
            continue

        key = normalized.casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(normalized)

        if len(result) >= limit:
            break

    return result


def _normalize_term(term: str) -> str:
    return " ".join(
        term.split()
    ).strip(" ,.;")


def _normalize_query(query: str) -> str:
    return " ".join(
        query.split()
    ).strip()


def _query_key(query: str) -> str:
    return _normalize_query(query).casefold()


def _term_count(query: str) -> int:
    """
    Count lexical search terms.

    Boolean operators are ignored so they do not artificially make a query
    appear meaningful.
    """

    tokens = _TERM_RE.findall(query)

    return sum(
        1
        for token in tokens
        if token.casefold() not in {
            "and",
            "or",
        }
    )
