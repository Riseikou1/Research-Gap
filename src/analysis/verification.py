"""Targeted counterexample search and conservative research-gap assessment."""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.analysis.comparison import (
    dataset_types,
    normalize_constraint,
    normalize_dataset,
    normalize_method_family,
    normalize_metric,
    normalize_problem,
)
from src.analysis.models import (
    EvidenceRole,
    GapAssessmentLabel,
    GapCandidate,
    GapEvidence,
    GapVerification,
    IdeaAssessment,
    VerificationFailure,
    VerificationQuery,
)
from src.extraction.evidence import EvidenceItem, PaperEvidence, canonical_evidence_key
from src.extraction.paper_extractor import PaperExtractor
from src.models.idea import ResearchIdea
from src.models.landscape import LiteratureLandscape
from src.models.paper import Paper
from src.models.query import SearchQuery
from src.ranking.lexical import LexicalScorer
from src.retrieval.multi_query import MultiQueryRetriever

from .gap_candidates import is_concrete_entity


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "for", "from", "in", "is", "of", "on", "or",
    "the", "to", "under", "with",
}

_GENERIC_BUCKETS = {
    "other",
    "unknown",
    "misc",
    "miscellaneous",
    "unspecified",
    "uncategorized",
    "none",
}

_PATTERN_TERMS = {
    "narrow_dataset_setting": "alternative validation setting",
    "combination_gap": "joint study evaluation",
    "missing_comparison": "direct comparison benchmark",
    "repeated_limitation": "address evaluate limitation",
    "repeated_future_work": "evaluate future direction",
    "comparable_conflict": "resolve conflicting findings",
    "underrepresented_population": "broader population evaluation",
    "limited_external_validation": "independent validation",
    "limited_real_world_validation": "deployment validation",
    "evaluation_gap": "direct evaluation",
    "replication_gap": "replication validation",
    "method_domain_transfer": "method transfer evaluation",
}

_COMPARISON_RELATION_PATTERN = re.compile(
    r"\b(?:"
    r"compar(?:e|ed|es|ing|ison)|"
    r"versus|vs|against|"
    r"baseline|"
    r"benchmark(?:ed|ing)?|"
    r"outperform(?:s|ed|ing)?|"
    r"underperform(?:s|ed|ing)?|"
    r"better\s+than|"
    r"worse\s+than|"
    r"superior\s+to|"
    r"inferior\s+to"
    r")\b",
    re.I,
)

_STUDY_ACTION_PATTERN = re.compile(
    r"\b(?:"
    r"assess|assessed|"
    r"compare|compared|"
    r"deploy|deployed|"
    r"evaluate|evaluated|"
    r"implement|implemented|"
    r"investigate|investigated|"
    r"measure|measured|"
    r"replicate|replicated|"
    r"test|tested|"
    r"validate|validated"
    r")\b",
    re.I,
)

_CONFLICT_RESOLUTION_PATTERN = re.compile(
    r"\b(?:"
    r"explain|explained|"
    r"reconcile|reconciled|"
    r"moderator|moderating|"
    r"subgroup|"
    r"heterogeneity|"
    r"interaction|"
    r"conditions?\s+(?:under|in which)|"
    r"source\s+of\s+(?:the\s+)?difference"
    r")\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Generic text helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    normalized = canonical_evidence_key(text)

    normalized = re.sub(
        r"\b(?:other|unknown|misc|miscellaneous|unspecified|"
        r"uncategorized|none)"
        r"(?:\s+(?:method|methods|model|models|approach|approaches|"
        r"technique|techniques|population|populations|setting|settings|"
        r"dataset|datasets|data|baseline|baselines))?\b",
        " ",
        normalized,
    )

    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in _STOPWORDS
        and token not in _GENERIC_BUCKETS
    ]


def _concept_tokens(text: str) -> set[str]:
    """Normalize a few harmless grammatical variants."""

    forms = {
        "reduce": "reduce",
        "reduces": "reduce",
        "reduced": "reduce",
        "reducing": "reduce",
        "reduction": "reduce",
        "reductions": "reduce",

        "decrease": "reduce",
        "decreases": "reduce",
        "decreased": "reduce",
        "decreasing": "reduce",

        "improve": "improve",
        "improves": "improve",
        "improved": "improve",
        "improving": "improve",
        "improvement": "improve",
        "improvements": "improve",

        "increase": "increase",
        "increases": "increase",
        "increased": "increase",
        "increasing": "increase",
    }

    result: set[str] = set()

    for token in _tokens(text):
        token = forms.get(token, token)

        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 5 and token.endswith("s"):
            token = token[:-1]

        result.add(token)

    return result


def _phrase_matches(value: str, text: str) -> bool:
    """Require substantial concept containment rather than loose overlap."""

    required = _concept_tokens(value)
    observed = _concept_tokens(text)

    if not required or not observed:
        return False

    return required <= observed or observed <= required


def _short_phrase(
    *parts: str,
    max_terms: int = 24,
) -> str:
    """Bound a query without destroying meaningful repeated phrase tokens."""

    terms: list[str] = []

    for part in parts:
        if not part:
            continue

        terms.extend(
            _tokens(part)
        )

    return " ".join(
        terms[:max_terms]
    )[:1000].strip()


# ---------------------------------------------------------------------------
# Requirement grouping
# ---------------------------------------------------------------------------


def _normalize_requirement(
    value: str,
    dimension: str,
) -> str:
    if dimension in {
        "method",
        "method_family",
        "comparison",
    }:
        return normalize_method_family(value)

    if dimension == "problem":
        return normalize_problem(value)

    if dimension == "constraint":
        return normalize_constraint(value)

    if dimension == "dataset":
        return normalize_dataset(value)

    if dimension in {
        "metric",
        "performance_metric",
        "efficiency_metric",
    }:
        return normalize_metric(value)

    return canonical_evidence_key(value)


def _requirement_key(
    value: str,
    dimension: str,
) -> str:
    normalized = _normalize_requirement(
        value,
        dimension,
    )

    return normalized or canonical_evidence_key(value)


def _synonym_requirement_keys(
    idea: ResearchIdea,
    value: str,
    dimension: str,
) -> set[str]:
    """Return all explicit equivalents associated with one requirement."""

    keys = {
        _requirement_key(
            value,
            dimension,
        )
    }

    value_key = canonical_evidence_key(
        value
    )

    for canonical, alternatives in idea.synonyms.items():
        linked = [
            canonical,
            *alternatives,
        ]

        linked_raw_keys = {
            canonical_evidence_key(item)
            for item in linked
        }

        linked_requirement_keys = {
            _requirement_key(
                item,
                dimension,
            )
            for item in linked
        }

        if (
            value_key in linked_raw_keys
            or keys & linked_requirement_keys
        ):
            keys.update(
                linked_requirement_keys
            )

    return {
        key
        for key in keys
        if key
    }


def _linked_requirement_values(
    idea: ResearchIdea,
    value: str,
    dimension: str,
) -> list[str]:
    """Return one requirement plus explicitly declared equivalents."""

    result = [value]

    raw_key = canonical_evidence_key(
        value
    )
    requirement_key = _requirement_key(
        value,
        dimension,
    )

    for canonical, alternatives in idea.synonyms.items():
        linked = [
            canonical,
            *alternatives,
        ]

        raw_keys = {
            canonical_evidence_key(item)
            for item in linked
        }

        requirement_keys = {
            _requirement_key(
                item,
                dimension,
            )
            for item in linked
        }

        if (
            raw_key in raw_keys
            or requirement_key in requirement_keys
        ):
            result.extend(
                linked
            )

    unique: list[str] = []
    seen: set[str] = set()

    for item in result:
        key = canonical_evidence_key(
            item
        )

        if not key or key in seen:
            continue

        seen.add(
            key
        )
        unique.append(
            item
        )

    return unique


def _idea_requirement_groups(
    idea: ResearchIdea,
    values: Sequence[str],
    dimension: str,
) -> list[tuple[str, ...]]:
    """Keep distinct requirements separate while grouping explicit equivalents."""

    groups: list[list[str]] = []

    for value in values:
        if (
            not value
            or not is_concrete_entity(value)
        ):
            continue

        alternatives = _linked_requirement_values(
            idea,
            value,
            dimension,
        )

        alternative_keys = {
            _requirement_key(
                item,
                dimension,
            )
            for item in alternatives
        }

        matching_group: list[str] | None = None

        for group in groups:
            group_keys = {
                _requirement_key(
                    item,
                    dimension,
                )
                for item in group
            }

            if alternative_keys & group_keys:
                matching_group = group
                break

        if matching_group is None:
            groups.append(
                alternatives
            )
        else:
            matching_group.extend(
                alternatives
            )

    return [
        tuple(
            dict.fromkeys(group)
        )
        for group in groups
    ]


def _all_requirement_groups_match(
    groups: Sequence[Sequence[str]],
    matcher,
) -> bool:
    return bool(groups) and all(
        any(
            matcher(value)
            for value in group
        )
        for group in groups
    )


# ---------------------------------------------------------------------------
# ResearchIdea facets
# ---------------------------------------------------------------------------


def _idea_methods(
    idea: ResearchIdea,
) -> list[str]:
    return [
        value
        for value in idea.intervention_or_method
        if is_concrete_entity(value)
    ]


def _idea_problems(
    idea: ResearchIdea,
) -> list[str]:
    if idea.problem:
        return [
            value
            for value in idea.problem
            if is_concrete_entity(value)
        ]

    inferred = normalize_problem(
        idea.original_text
    )

    return (
        [inferred]
        if inferred
        and inferred != "other"
        else []
    )


def _idea_constraints(
    idea: ResearchIdea,
) -> list[str]:
    return [
        value
        for value in idea.constraints
        if is_concrete_entity(value)
    ]


def _idea_context_values(
    idea: ResearchIdea,
) -> list[str]:
    return [
        value
        for value in [
            *idea.population,
            *idea.domain,
        ]
        if is_concrete_entity(value)
    ]


def _idea_all_terms(
    idea: ResearchIdea,
) -> str:
    return " ".join(
        [
            idea.original_text,
            *idea.problem,
            *idea.population,
            *idea.intervention_or_method,
            *idea.comparison,
            *idea.outcomes,
            *idea.domain,
            *idea.constraints,
            *idea.keywords,
        ]
    )


# ---------------------------------------------------------------------------
# Verification query construction
# ---------------------------------------------------------------------------


def _facet_query_variants(
    idea: ResearchIdea,
    value: str,
    dimension: str,
) -> tuple[str, ...]:
    values = [
        value,
    ]

    normalized = _normalize_requirement(
        value,
        dimension,
    )

    if normalized:
        values.append(
            normalized
        )

    value_key = canonical_evidence_key(
        value
    )

    for canonical, alternatives in idea.synonyms.items():
        linked = [
            canonical,
            *alternatives,
        ]

        if value_key in {
            canonical_evidence_key(item)
            for item in linked
        }:
            values.extend(
                linked
            )

    return tuple(
        dict.fromkeys(
            item
            for item in values
            if _tokens(item)
        )
    )


def _query_terms_for_requirements(
    idea: ResearchIdea,
    values: Sequence[str],
    dimension: str,
) -> str:
    groups = _idea_requirement_groups(
        idea,
        values,
        dimension,
    )

    parts: list[str] = []

    for group in groups:
        variants = _facet_query_variants(
            idea,
            group[0],
            dimension,
        )

        parts.extend(
            variants[:2]
        )

    return " ".join(
        parts
    )


def build_idea_verification_queries(
    idea: ResearchIdea,
    *,
    max_queries: int = 3,
) -> list[VerificationQuery]:
    """Build bounded searches preserving every explicit research requirement."""

    if not 1 <= max_queries <= 3:
        raise ValueError(
            "max_queries must be between 1 and 3"
        )

    method = _query_terms_for_requirements(
        idea,
        _idea_methods(idea),
        "method",
    )

    problem = _query_terms_for_requirements(
        idea,
        _idea_problems(idea),
        "problem",
    )

    context = _query_terms_for_requirements(
        idea,
        _idea_context_values(idea),
        "context",
    )

    constraints = _query_terms_for_requirements(
        idea,
        _idea_constraints(idea),
        "constraint",
    )

    comparison = _query_terms_for_requirements(
        idea,
        idea.comparison,
        "comparison",
    )

    outcomes = _query_terms_for_requirements(
        idea,
        idea.outcomes,
        "outcome",
    )

    raw_queries = [
        _short_phrase(
            method,
            comparison,
            problem,
            outcomes,
            context,
            constraints,
        ),
        _short_phrase(
            problem,
            method,
            context,
            comparison,
            constraints,
            outcomes,
        ),
        _short_phrase(
            method,
            problem,
            constraints,
            comparison,
            outcomes,
            context,
        ),
    ]

    result: list[VerificationQuery] = []
    seen: set[str] = set()

    for text in raw_queries:
        key = " ".join(
            _tokens(text)
        )

        if (
            not key
            or key in seen
            or len(key.split()) < 2
        ):
            continue

        seen.add(
            key
        )

        result.append(
            VerificationQuery(
                candidate_id="idea",
                query=text,
                pattern_type="direct_idea_assessment",
                strategy="verification_direct_idea",
                source="deterministic",
            )
        )

        if len(result) >= max_queries:
            break

    return result


def _candidate_support_variants(
    candidate: GapCandidate,
    dimension: str,
    value: str,
) -> list[str]:
    """Recover wording variants from the candidate's own provenance."""

    target = _normalize_requirement(
        value,
        dimension,
    )

    result: list[str] = []

    for item in candidate.supporting_evidence:
        item_dimension = {
            "method_or_intervention": "method",
            "method": "method",
            "comparison_or_baseline": "comparison",
            "comparison": "comparison",
            "constraint": "constraint",
            "constraints": "constraint",
            "dataset": "dataset",
            "datasets": "dataset",
            "population_or_setting": "population_or_setting",
            "research_objective": "problem",
        }.get(
            item.evidence_type,
            item.evidence_type,
        )

        if dimension in {
            "method",
            "method_family",
        }:
            compatible = item_dimension == "method"
        else:
            compatible = (
                item_dimension == dimension
            )

        if not compatible:
            continue

        normalized = _normalize_requirement(
            item.value,
            dimension,
        )

        if (
            target
            and normalized
            and normalized == target
        ):
            result.append(
                item.value
            )

    return result


def _candidate_facet_groups(
    candidate: GapCandidate,
) -> list[tuple[str, ...]]:
    groups: list[tuple[str, ...]] = []

    for basis in candidate.landscape_basis:
        if (
            not basis.value
            or not _tokens(basis.value)
        ):
            continue

        values = [
            basis.value,
        ]

        normalized = _normalize_requirement(
            basis.value,
            basis.dimension,
        )

        if normalized:
            values.append(
                normalized
            )

        values.extend(
            _candidate_support_variants(
                candidate,
                basis.dimension,
                basis.value,
            )
        )

        groups.append(
            tuple(
                dict.fromkeys(
                    value
                    for value in values
                    if _tokens(value)
                )
            )
        )

    if groups:
        return groups

    fallback = " ".join(
        _tokens(candidate.title)
    )

    return [
        (fallback,)
    ] if fallback else []


def _query_preserves_candidate_facets(
    query: str,
    candidate: GapCandidate,
) -> bool:
    query_terms = set(
        _tokens(query)
    )

    for alternatives in _candidate_facet_groups(
        candidate
    ):
        if not any(
            set(_tokens(value)) <= query_terms
            for value in alternatives
        ):
            return False

    return True


def build_verification_queries(
    idea: ResearchIdea,
    candidate: GapCandidate,
    *,
    max_queries: int = 3,
) -> list[VerificationQuery]:
    """Build deterministic candidate-specific counterexample searches."""

    if not 1 <= max_queries <= 3:
        raise ValueError(
            "max_queries must be between 1 and 3"
        )

    basis_terms = " ".join(
        item.value
        for item in candidate.landscape_basis[:4]
    )

    candidate_facets = " ".join(
        group[0]
        for group in _candidate_facet_groups(candidate)
        if group
    )

    methods = " ".join(
        _idea_methods(idea)[:2]
    )

    problems = " ".join(
        _idea_problems(idea)[:2]
    )

    context = " ".join(
        _idea_context_values(idea)[:2]
    )

    pattern_terms = _PATTERN_TERMS.get(
        candidate.pattern_type,
        "direct evaluation",
    )

    raw_queries = [
        _short_phrase(
            candidate_facets,
            methods,
            problems,
            context,
            pattern_terms,
        ),
        _short_phrase(
            candidate_facets,
            candidate.title,
            problems,
            pattern_terms,
        ),
        _short_phrase(
            candidate_facets,
            basis_terms,
            methods,
            pattern_terms,
        ),
    ]

    original_key = " ".join(
        _tokens(idea.original_text)
    )

    result: list[VerificationQuery] = []
    seen: set[str] = set()

    for text in raw_queries:
        key = " ".join(
            _tokens(text)
        )

        if (
            not key
            or key == original_key
            or key in seen
            or len(key.split()) < 2
            or not _query_preserves_candidate_facets(
                text,
                candidate,
            )
        ):
            continue

        seen.add(
            key
        )

        result.append(
            VerificationQuery(
                candidate_id=candidate.id,
                query=text,
                pattern_type=candidate.pattern_type,
                strategy="verification_counterexample",
                source="deterministic",
            )
        )

        if len(result) >= max_queries:
            break

    return result


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------


class GapVerifier:
    """Verify ideas and candidate gaps using targeted retrieval."""

    def __init__(
        self,
        retriever: MultiQueryRetriever,
        extractor: PaperExtractor,
        *,
        lexical_scorer: LexicalScorer | None = None,
        max_queries: int = 3,
        per_query_limit: int = 10,
        max_verification_papers: int = 10,
    ) -> None:
        if (
            not 1 <= max_queries <= 3
            or not 1 <= per_query_limit <= 100
            or not 1 <= max_verification_papers <= 100
        ):
            raise ValueError(
                "verification limits are outside their allowed bounds"
            )

        self.retriever = retriever
        self.extractor = extractor
        self.lexical_scorer = (
            lexical_scorer
            or LexicalScorer()
        )
        self.max_queries = max_queries
        self.per_query_limit = per_query_limit
        self.max_verification_papers = max_verification_papers
        self.notices: list[str] = []

    def verify_many(
        self,
        idea: ResearchIdea,
        candidates: Sequence[GapCandidate],
        evidence: Sequence[PaperEvidence],
    ) -> list[GapCandidate]:
        self.notices = []

        return [
            self.verify(
                idea,
                candidate,
                evidence,
            )
            for candidate in candidates
        ]

    def assess_idea(
        self,
        idea: ResearchIdea,
        landscape: LiteratureLandscape | None = None,
        evidence: Sequence[PaperEvidence] = (),
    ) -> IdeaAssessment:
        """Assess the complete user idea independently of candidate gaps."""

        queries = build_idea_verification_queries(
            idea,
            max_queries=self.max_queries,
        )

        if not queries:
            return IdeaAssessment(
                label="uncertain",
                rationale=(
                    "The complete idea did not yield a sufficiently "
                    "specific verification query."
                ),
                coverage_notes=[
                    "Direct idea verification did not execute."
                ],
            )

        retrieval = self.retriever.retrieve_verification(
            [
                SearchQuery(
                    text=item.query,
                    strategy=item.strategy,
                    source=item.source,
                )
                for item in queries
            ],
            per_query_limit=self.per_query_limit,
            limit=self.max_verification_papers,
        )

        papers = _rank_for_idea(
            idea,
            retrieval.papers,
            self.lexical_scorer,
        )

        extracted = self.extractor.extract_many(
            papers,
            limit=self.max_verification_papers,
        )

        # Targeted verification evidence takes precedence when the same paper
        # also appeared in the initial landscape.
        records = {
            item.paper_id: item
            for item in evidence
        }

        records.update(
            {
                item.paper_id: item
                for item in extracted
            }
        )

        failures = [
            VerificationFailure(
                query=item.query,
                provider=item.provider,
                error=item.error,
            )
            for item in retrieval.failures
        ]

        extraction_failures = list(
            getattr(
                self.extractor,
                "failures",
                [],
            )
        )

        failures.extend(
            VerificationFailure(
                query="evidence_extraction",
                provider="evidence_extractor",
                error=str(item),
            )
            for item in extraction_failures
        )

        direct_ids: list[str] = []
        partial_ids: list[str] = []
        potential_ids: list[str] = []

        direct_evidence: list[GapEvidence] = []
        partial_evidence: list[GapEvidence] = []
        potential_evidence: list[GapEvidence] = []

        matched_facets: dict[str, list[str]] = {}

        retrieved_abstract_ids = {
            paper.id
            for paper in papers
            if paper.abstract
        }

        for record in records.values():
            complete, matched = _idea_match_strength(
                idea,
                record,
            )

            abstract_supported = (
                record.paper_id
                in retrieved_abstract_ids
                or _record_has_abstract_evidence(record)
            )

            if complete and abstract_supported:
                role: EvidenceRole = "confirmed_direct_match"
                target_ids = direct_ids
                target_evidence = direct_evidence

            elif matched and abstract_supported:
                role = "contextual_or_partial_support"
                target_ids = partial_ids
                target_evidence = partial_evidence

            elif matched or _title_may_match_idea(
                idea,
                record,
            ):
                role = "potential_match"
                target_ids = potential_ids
                target_evidence = potential_evidence

            else:
                continue

            target_ids.append(
                record.paper_id
            )

            matched_facets[
                record.paper_id
            ] = list(matched)

            evidence_items = _idea_evidence(
                idea,
                record,
                matched,
            )[:10]

            for field_name, item in evidence_items:
                target_evidence.append(
                    GapEvidence(
                        paper_id=record.paper_id,
                        evidence_type=field_name,
                        value=item.value,
                        evidence_text=item.evidence_text,
                        study_type=record.study_type,
                        role=role,
                    )
                )

            if (
                role == "potential_match"
                and not evidence_items
            ):
                target_evidence.append(
                    GapEvidence(
                        paper_id=record.paper_id,
                        evidence_type="title",
                        value=record.title,
                        evidence_text=record.title,
                        study_type=record.study_type,
                        role=role,
                    )
                )

        coverage_ok, coverage_detail = _idea_coverage(
            idea,
            evidence,
        )

        coverage_notes = [
            (
                f"Executed {len(queries)} direct idea verification "
                f"queries and retrieved {len(papers)} unique papers."
            ),
            (
                "Direct verification used structured title/abstract "
                "evidence only."
            ),
            *coverage_detail,
        ]

        if extraction_failures:
            coverage_notes.append(
                f"Evidence extraction failed for "
                f"{len(extraction_failures)} verification papers."
            )

        if retrieval.failures:
            coverage_notes.append(
                "Verification provider failure prevents a positive "
                "gap assessment."
            )

        if any(
            not paper.abstract
            for paper in papers
        ):
            coverage_notes.append(
                "Some verification papers lacked abstracts; title-only "
                "matches were not treated as direct."
            )

        if direct_ids:
            label: GapAssessmentLabel = "well_studied"
            rationale = (
                "Targeted verification found a paper satisfying all "
                "explicitly represented research-idea requirements "
                "within the same study."
            )

        elif (
            failures
            or not coverage_ok
            or not papers
            or partial_ids
            or potential_ids
        ):
            label = "uncertain"
            rationale = (
                "The available evidence contains failures, incomplete "
                "facet coverage, or only partial/potential matches, so "
                "the complete research idea cannot be decided safely."
            )

        elif _positive_idea_signal(
            idea,
            landscape,
            evidence,
        ):
            label = "promising_gap"
            rationale = (
                "The analyzed evidence covers the explicit idea facets, "
                "targeted verification completed successfully, and no "
                "same-paper complete match was found."
            )

        else:
            label = "uncertain"
            rationale = (
                "Targeted verification completed, but the available "
                "structured evidence is not strong enough to support a "
                "positive gap assessment."
            )

        return IdeaAssessment(
            label=label,
            rationale=rationale,
            supporting_paper_ids=list(
                dict.fromkeys(
                    [
                        *direct_ids,
                        *partial_ids,
                        *potential_ids,
                    ]
                )
            ),
            supporting_evidence=_unique_evidence(
                [
                    *direct_evidence,
                    *partial_evidence,
                    *potential_evidence,
                ]
            ),
            counterexample_paper_ids=list(
                dict.fromkeys(direct_ids)
            ),
            counterexample_evidence=_unique_evidence(
                direct_evidence
            ),
            partial_match_paper_ids=list(
                dict.fromkeys(partial_ids)
            ),
            potential_match_paper_ids=list(
                dict.fromkeys(potential_ids)
            ),
            matched_facets=matched_facets,
            verification_queries=queries,
            searched_paper_ids=[
                paper.id
                for paper in papers
            ],
            coverage_notes=coverage_notes,
            failures=failures,
        )

    def verify(
        self,
        idea: ResearchIdea,
        candidate: GapCandidate,
        evidence: Sequence[PaperEvidence],
    ) -> GapCandidate:
        queries = [
            query
            for query in candidate.verification_queries
            if query.candidate_id == candidate.id
            and _query_preserves_candidate_facets(
                query.query,
                candidate,
            )
        ][: self.max_queries]

        if not queries:
            queries = build_verification_queries(
                idea,
                candidate,
                max_queries=self.max_queries,
            )

        candidate.verification_queries = queries

        if not queries:
            return _apply_verification(
                candidate,
                GapVerification(
                    candidate_id=candidate.id,
                    verification_queries=[],
                    label="uncertain",
                    reason=(
                        "No candidate-specific counterexample query "
                        "could be constructed."
                    ),
                    coverage_notes=[
                        "Targeted verification did not execute."
                    ],
                ),
            )

        retrieval = self.retriever.retrieve_verification(
            [
                SearchQuery(
                    text=item.query,
                    strategy=item.strategy,
                    source=item.source,
                )
                for item in queries
            ],
            per_query_limit=self.per_query_limit,
            limit=self.max_verification_papers,
        )

        papers = _rank_for_candidate(
            candidate,
            retrieval.papers,
            self.lexical_scorer,
        )

        extracted = self.extractor.extract_many(
            papers,
            limit=self.max_verification_papers,
        )

        records = {
            item.paper_id: item
            for item in extracted
        }

        failures = [
            VerificationFailure(
                query=failure.query,
                provider=failure.provider,
                error=failure.error,
            )
            for failure in retrieval.failures
        ]

        extraction_failures = list(
            getattr(
                self.extractor,
                "failures",
                [],
            )
        )

        failures.extend(
            VerificationFailure(
                query="evidence_extraction",
                provider="evidence_extractor",
                error=str(item),
            )
            for item in extraction_failures
        )

        coverage_notes = [
            (
                f"Executed {len(queries)} candidate-specific verification "
                f"queries and retrieved {len(papers)} unique papers."
            ),
            "Verification used structured title/abstract evidence only.",
        ]

        if extraction_failures:
            coverage_notes.append(
                f"Evidence extraction failed for "
                f"{len(extraction_failures)} verification papers."
            )

        if retrieval.failures:
            coverage_notes.append(
                "At least one verification provider route failed; "
                "failure is not treated as absence of counterexamples."
            )

        if any(
            not paper.abstract
            for paper in papers
        ):
            coverage_notes.append(
                "Some verification papers had no abstract; title-only "
                "matches remain potential, not confirmed."
            )

        potential: list[str] = []
        confirmed: list[str] = []
        inspected: list[GapEvidence] = []

        for paper in papers:
            record = records.get(
                paper.id
            )

            role: EvidenceRole | None = None
            items: list[
                tuple[str, EvidenceItem]
            ] = []

            if record is None:
                if _title_possible_match(
                    candidate,
                    paper,
                ):
                    role = "potential_contradiction"

                    items = [
                        (
                            "title",
                            EvidenceItem(
                                value=paper.title,
                                evidence_text=paper.title,
                                source="title",
                                confidence=0.4,
                            ),
                        )
                    ]

            else:
                direct = (
                    _record_has_abstract_evidence(record)
                    and _direct_counterexample(
                        candidate,
                        record,
                        idea,
                    )
                )

                contextual = _contextual_match(
                    candidate,
                    record,
                    idea,
                )

                if direct:
                    role = "confirmed_contradiction"
                elif contextual:
                    role = "potential_contradiction"
                elif (
                    not paper.abstract
                    and _title_possible_match(
                        candidate,
                        paper,
                    )
                ):
                    role = "potential_contradiction"

                items = _relevant_items(
                    candidate,
                    record,
                    idea,
                )

            if role is None:
                continue

            if role == "confirmed_contradiction":
                confirmed.append(
                    paper.id
                )
            else:
                potential.append(
                    paper.id
                )

            for evidence_type, item in items[:4]:
                inspected.append(
                    GapEvidence(
                        paper_id=paper.id,
                        evidence_type=evidence_type,
                        value=item.value,
                        evidence_text=item.evidence_text,
                        study_type=(
                            record.study_type
                            if record
                            else None
                        ),
                        role=role,
                    )
                )

        confirmed = list(
            dict.fromkeys(confirmed)
        )
        potential = list(
            dict.fromkeys(potential)
        )

        label, reason = _assessment(
            candidate,
            papers,
            records,
            confirmed,
            potential,
            failures,
        )

        verification = GapVerification(
            candidate_id=candidate.id,
            verification_queries=list(queries),
            searched_paper_ids=[
                paper.id
                for paper in papers
            ],
            supporting_paper_ids=list(
                candidate.supporting_paper_ids
            ),
            potential_contradiction_paper_ids=potential,
            contradicting_paper_ids=confirmed,
            evidence=_unique_evidence(inspected),
            coverage_notes=coverage_notes,
            failures=failures,
            label=label,
            reason=reason,
        )

        return _apply_verification(
            candidate,
            verification,
        )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _rank_for_idea(
    idea: ResearchIdea,
    papers: Sequence[Paper],
    scorer: LexicalScorer,
) -> list[Paper]:
    query = _idea_all_terms(
        idea
    )

    scores = scorer.score_many(
        query,
        papers,
    )

    return [
        paper
        for _, paper in sorted(
            zip(scores, papers),
            key=lambda pair: (
                -pair[0],
                pair[1].title.casefold(),
                pair[1].id.casefold(),
            ),
        )
    ]


def _rank_for_candidate(
    candidate: GapCandidate,
    papers: Sequence[Paper],
    scorer: LexicalScorer,
) -> list[Paper]:
    query = " ".join(
        [
            candidate.title,
            candidate.description,
            *(
                item.value
                for item in candidate.landscape_basis
            ),
        ]
    )

    scores = scorer.score_many(
        query,
        papers,
    )

    return [
        paper
        for _, paper in sorted(
            zip(scores, papers),
            key=lambda pair: (
                -pair[0],
                pair[1].title.casefold(),
                pair[1].id.casefold(),
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Evidence access
# ---------------------------------------------------------------------------


def _record_items(
    record: PaperEvidence,
) -> list[tuple[str, EvidenceItem]]:
    fields = (
        "research_objective",
        "population_or_setting",
        "method_or_intervention",
        "comparison_or_baseline",
        "datasets",
        "evaluation_metrics",
        "main_findings",
        "constraints",
        "limitations",
        "future_work",
    )

    result: list[
        tuple[str, EvidenceItem]
    ] = []

    for field_name in fields:
        value = getattr(
            record,
            field_name,
        )

        items = (
            value
            if isinstance(value, list)
            else [value]
            if value
            else []
        )

        result.extend(
            (
                field_name,
                item,
            )
            for item in items
        )

    return result


def _experimental_items(
    record: PaperEvidence,
) -> list[tuple[str, EvidenceItem]]:
    allowed = {
        "research_objective",
        "population_or_setting",
        "method_or_intervention",
        "comparison_or_baseline",
        "datasets",
        "evaluation_metrics",
        "main_findings",
        "constraints",
    }

    return [
        (field_name, item)
        for field_name, item in _record_items(record)
        if field_name in allowed
    ]


def _record_text(
    record: PaperEvidence,
) -> str:
    return " ".join(
        [
            record.title,
            *(
                f"{item.value} {item.evidence_text}"
                for _, item in _record_items(record)
            ),
        ]
    )


def _experimental_record_text(
    record: PaperEvidence,
) -> str:
    return " ".join(
        f"{item.value} {item.evidence_text}"
        for _, item in _experimental_items(record)
    )


def _record_has_abstract_evidence(
    record: PaperEvidence,
) -> bool:
    return any(
        item.source == "abstract"
        for _, item in _record_items(record)
    )


# ---------------------------------------------------------------------------
# Direct idea matching
# ---------------------------------------------------------------------------


def _problem_item_matches(
    value: str,
    item: EvidenceItem,
) -> bool:
    target = normalize_problem(
        value
    )

    observed = normalize_problem(
        item.value
    )

    if not target or not observed:
        return False

    if _phrase_matches(
        value,
        f"{item.value} {item.evidence_text}",
    ):
        return True

    if target != observed:
        return False

    # If the requirement contains domain/task qualifiers beyond the broad
    # normalized task label, require at least one such qualifier too.
    specific_terms = (
        _concept_tokens(value)
        - _concept_tokens(target)
    )

    if not specific_terms:
        return True

    observed_terms = _concept_tokens(
        f"{item.value} {item.evidence_text}"
    )

    return bool(
        specific_terms
        & observed_terms
    )


def _method_item_matches(
    value: str,
    item: EvidenceItem,
) -> bool:
    if _phrase_matches(
        value,
        item.value,
    ):
        return True

    target = normalize_method_family(
        value
    )

    observed = normalize_method_family(
        item.value
    )

    if (
        not target
        or not observed
        or target != observed
    ):
        return False

    specific_terms = (
        _concept_tokens(value)
        - _concept_tokens(target)
    )

    if not specific_terms:
        return True

    observed_terms = _concept_tokens(
        item.value
    )

    return bool(
        specific_terms
        & observed_terms
    )


def _context_item_matches(
    value: str,
    item: EvidenceItem,
) -> bool:
    return _phrase_matches(
        value,
        f"{item.value} {item.evidence_text}",
    )


def _constraint_item_matches(
    value: str,
    item: EvidenceItem,
) -> bool:
    target = normalize_constraint(
        value
    )

    observed = normalize_constraint(
        item.value
    )

    if (
        target
        and observed
        and target == observed
    ):
        return True

    return _phrase_matches(
        value,
        f"{item.value} {item.evidence_text}",
    )


def _comparison_item_matches(
    value: str,
    item: EvidenceItem,
) -> bool:
    return _method_item_matches(
        value,
        item,
    )


def _outcome_item_matches(
    value: str,
    item: EvidenceItem,
) -> bool:
    return _phrase_matches(
        value,
        f"{item.value} {item.evidence_text}",
    )


def _problem_matches(
    values: Sequence[str],
    record: PaperEvidence,
    idea: ResearchIdea,
) -> bool:
    if not record.research_objective:
        return False

    groups = _idea_requirement_groups(
        idea,
        values,
        "problem",
    )

    return _all_requirement_groups_match(
        groups,
        lambda value: _problem_item_matches(
            value,
            record.research_objective,
        ),
    )


def _method_matches(
    values: Sequence[str],
    record: PaperEvidence,
    idea: ResearchIdea,
) -> bool:
    items = [
        item
        for item in record.method_or_intervention
        if is_concrete_entity(item.value)
    ]

    groups = _idea_requirement_groups(
        idea,
        values,
        "method",
    )

    return _all_requirement_groups_match(
        groups,
        lambda value: any(
            _method_item_matches(
                value,
                item,
            )
            for item in items
        ),
    )


def _context_matches(
    values: Sequence[str],
    record: PaperEvidence,
    idea: ResearchIdea,
) -> bool:
    items: list[EvidenceItem] = []

    if record.research_objective:
        items.append(
            record.research_objective
        )

    items.extend(
        record.population_or_setting
    )

    groups = _idea_requirement_groups(
        idea,
        values,
        "context",
    )

    return _all_requirement_groups_match(
        groups,
        lambda value: any(
            _context_item_matches(
                value,
                item,
            )
            for item in items
        ),
    )


def _constraint_matches(
    idea: ResearchIdea,
    record: PaperEvidence,
) -> bool:
    groups = _idea_requirement_groups(
        idea,
        _idea_constraints(idea),
        "constraint",
    )

    items = [
        item
        for item in record.constraints
        if is_concrete_entity(item.value)
    ]

    return _all_requirement_groups_match(
        groups,
        lambda value: any(
            _constraint_item_matches(
                value,
                item,
            )
            for item in items
        ),
    )


def _comparison_matches(
    idea: ResearchIdea,
    record: PaperEvidence,
) -> bool:
    groups = _idea_requirement_groups(
        idea,
        idea.comparison,
        "comparison",
    )

    items = [
        item
        for item in record.comparison_or_baseline
        if is_concrete_entity(item.value)
    ]

    return _all_requirement_groups_match(
        groups,
        lambda value: any(
            _comparison_item_matches(
                value,
                item,
            )
            for item in items
        ),
    )


def _outcome_evidence_items(
    record: PaperEvidence,
) -> list[EvidenceItem]:
    result: list[EvidenceItem] = []

    if record.research_objective:
        result.append(
            record.research_objective
        )

    result.extend(
        record.main_findings
    )
    result.extend(
        record.evaluation_metrics
    )

    return result


def _outcome_matches(
    idea: ResearchIdea,
    record: PaperEvidence,
) -> bool:
    groups = _idea_requirement_groups(
        idea,
        idea.outcomes,
        "outcome",
    )

    items = _outcome_evidence_items(
        record
    )

    return _all_requirement_groups_match(
        groups,
        lambda value: any(
            _outcome_item_matches(
                value,
                item,
            )
            for item in items
        ),
    )


def _idea_match_strength(
    idea: ResearchIdea,
    record: PaperEvidence,
) -> tuple[bool, list[str]]:
    groups: dict[str, bool] = {}

    problems = _idea_problems(
        idea
    )
    methods = _idea_methods(
        idea
    )
    context = _idea_context_values(
        idea
    )
    constraints = _idea_constraints(
        idea
    )

    if problems:
        groups["problem"] = _problem_matches(
            problems,
            record,
            idea,
        )

    if methods:
        groups["method"] = _method_matches(
            methods,
            record,
            idea,
        )

    if context:
        groups["population_or_domain"] = _context_matches(
            context,
            record,
            idea,
        )

    if constraints:
        groups["constraint"] = _constraint_matches(
            idea,
            record,
        )

    if idea.comparison:
        groups["comparison"] = _comparison_matches(
            idea,
            record,
        )

    if idea.outcomes:
        groups["outcome"] = _outcome_matches(
            idea,
            record,
        )

    matched = [
        name
        for name, present in groups.items()
        if present
    ]

    return (
        bool(groups)
        and len(matched) == len(groups),
        matched,
    )


# ---------------------------------------------------------------------------
# Evidence shown for direct matches
# ---------------------------------------------------------------------------


def _matching_requirement_items(
    idea: ResearchIdea,
    values: Sequence[str],
    items: Sequence[EvidenceItem],
    dimension: str,
) -> list[EvidenceItem]:
    groups = _idea_requirement_groups(
        idea,
        values,
        dimension,
    )

    result: list[EvidenceItem] = []

    for group in groups:
        for item in items:
            if dimension == "problem":
                matched = any(
                    _problem_item_matches(
                        value,
                        item,
                    )
                    for value in group
                )

            elif dimension in {
                "method",
                "method_family",
            }:
                matched = any(
                    _method_item_matches(
                        value,
                        item,
                    )
                    for value in group
                )

            elif dimension == "context":
                matched = any(
                    _context_item_matches(
                        value,
                        item,
                    )
                    for value in group
                )

            elif dimension == "constraint":
                matched = any(
                    _constraint_item_matches(
                        value,
                        item,
                    )
                    for value in group
                )

            elif dimension == "comparison":
                matched = any(
                    _comparison_item_matches(
                        value,
                        item,
                    )
                    for value in group
                )

            elif dimension == "outcome":
                matched = any(
                    _outcome_item_matches(
                        value,
                        item,
                    )
                    for value in group
                )

            else:
                matched = any(
                    _phrase_matches(
                        value,
                        f"{item.value} {item.evidence_text}",
                    )
                    for value in group
                )

            if matched:
                result.append(
                    item
                )
                break

    return result


def _idea_evidence(
    idea: ResearchIdea,
    record: PaperEvidence,
    matched: Sequence[str],
) -> list[tuple[str, EvidenceItem]]:
    result: list[
        tuple[str, EvidenceItem]
    ] = []

    if (
        "problem" in matched
        and record.research_objective
    ):
        result.append(
            (
                "research_objective",
                record.research_objective,
            )
        )

    if "method" in matched:
        result.extend(
            (
                "method_or_intervention",
                item,
            )
            for item in _matching_requirement_items(
                idea,
                _idea_methods(idea),
                record.method_or_intervention,
                "method",
            )
        )

    if "population_or_domain" in matched:
        result.extend(
            (
                "population_or_setting",
                item,
            )
            for item in _matching_requirement_items(
                idea,
                _idea_context_values(idea),
                record.population_or_setting,
                "context",
            )
        )

    if "constraint" in matched:
        result.extend(
            (
                "constraints",
                item,
            )
            for item in _matching_requirement_items(
                idea,
                _idea_constraints(idea),
                record.constraints,
                "constraint",
            )
        )

    if "comparison" in matched:
        result.extend(
            (
                "comparison_or_baseline",
                item,
            )
            for item in _matching_requirement_items(
                idea,
                idea.comparison,
                record.comparison_or_baseline,
                "comparison",
            )
        )

    if "outcome" in matched:
        for item in _matching_requirement_items(
            idea,
            idea.outcomes,
            _outcome_evidence_items(record),
            "outcome",
        ):
            if (
                record.research_objective
                and item is record.research_objective
            ):
                field = "research_objective"

            elif item in record.main_findings:
                field = "main_findings"

            else:
                field = "evaluation_metrics"

            result.append(
                (
                    field,
                    item,
                )
            )

    return result


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _idea_coverage(
    idea: ResearchIdea,
    evidence: Sequence[PaperEvidence],
) -> tuple[bool, list[str]]:
    notes = [
        (
            f"Analyzed {len(evidence)} evidence-bearing papers "
            "before direct idea assessment."
        )
    ]

    if len(evidence) < 2:
        notes.append(
            "Fewer than two evidence-bearing papers were available "
            "for coverage assessment."
        )

        return False, notes

    missing: list[str] = []

    checks = (
        (
            "problem",
            _idea_problems(idea),
            "problem",
            lambda value, record: _problem_matches(
                [value],
                record,
                idea,
            ),
        ),
        (
            "method",
            _idea_methods(idea),
            "method",
            lambda value, record: _method_matches(
                [value],
                record,
                idea,
            ),
        ),
        (
            "comparison",
            idea.comparison,
            "comparison",
            lambda value, record: _comparison_matches(
                idea.model_copy(
                    update={
                        "comparison": [value]
                    }
                ),
                record,
            ),
        ),
        (
            "outcome",
            idea.outcomes,
            "outcome",
            lambda value, record: _outcome_matches(
                idea.model_copy(
                    update={
                        "outcomes": [value]
                    }
                ),
                record,
            ),
        ),
    )

    for name, values, dimension, matcher in checks:
        groups = _idea_requirement_groups(
            idea,
            values,
            dimension,
        )

        if groups and not all(
            any(
                matcher(
                    value,
                    record,
                )
                for value in group
                for record in evidence
            )
            for group in groups
        ):
            missing.append(
                name
            )

    context = _idea_context_values(
        idea
    )

    context_groups = _idea_requirement_groups(
        idea,
        context,
        "context",
    )

    if context_groups and not all(
        any(
            _context_matches(
                [value],
                record,
                idea,
            )
            for value in group
            for record in evidence
        )
        for group in context_groups
    ):
        missing.append(
            "population_or_domain"
        )

    constraint_groups = _idea_requirement_groups(
        idea,
        _idea_constraints(idea),
        "constraint",
    )

    if constraint_groups and not all(
        any(
            _constraint_item_matches(
                value,
                item,
            )
            for value in group
            for record in evidence
            for item in record.constraints
        )
        for group in constraint_groups
    ):
        missing.append(
            "constraint"
        )

    if missing:
        notes.append(
            "Important idea facets without explicit structured coverage: "
            + ", ".join(missing)
            + "."
        )

    return (
        not missing,
        notes,
    )


def _positive_idea_signal(
    idea: ResearchIdea,
    landscape: LiteratureLandscape | None,
    evidence: Sequence[PaperEvidence],
) -> bool:
    if (
        landscape is None
        or landscape.total_papers < 2
        or len(evidence) < 2
    ):
        return False

    coverage_ok, _ = _idea_coverage(
        idea,
        evidence,
    )

    return coverage_ok


# ---------------------------------------------------------------------------
# Candidate basis matching
# ---------------------------------------------------------------------------


def _basis_requirement_matches(
    dimension: str,
    value: str,
    record: PaperEvidence,
) -> bool:
    if dimension == "problem":
        return bool(
            record.research_objective
            and _problem_item_matches(
                value,
                record.research_objective,
            )
        )

    if dimension in {
        "method",
        "method_family",
    }:
        return any(
            _method_item_matches(
                value,
                item,
            )
            for item in record.method_or_intervention
            if is_concrete_entity(item.value)
        )

    if dimension in {
        "population",
        "population_or_setting",
        "setting",
    }:
        return any(
            _context_item_matches(
                value,
                item,
            )
            for item in record.population_or_setting
        )

    if dimension == "constraint":
        return any(
            _constraint_item_matches(
                value,
                item,
            )
            for item in record.constraints
        )

    if dimension == "dataset":
        target = normalize_dataset(
            value
        )

        return bool(
            target
            and any(
                normalize_dataset(
                    item.value
                ) == target
                for item in record.datasets
            )
        )

    if dimension == "dataset_type":
        target = canonical_evidence_key(
            value
        )

        return target in {
            canonical_evidence_key(
                item
            )
            for item in dataset_types(record)
        }

    if dimension in {
        "baseline",
        "comparison",
    }:
        return any(
            _comparison_item_matches(
                value,
                item,
            )
            for item in record.comparison_or_baseline
        )

    if dimension in {
        "performance_metric",
        "efficiency_metric",
        "metric",
    }:
        target = normalize_metric(
            value
        )

        return bool(
            target
            and any(
                normalize_metric(
                    item.value
                ) == target
                for item in record.evaluation_metrics
            )
        )

    if dimension == "outcome":
        return any(
            _outcome_item_matches(
                value,
                item,
            )
            for item in record.main_findings
        )

    if dimension == "study_type":
        return (
            canonical_evidence_key(
                record.study_type
            )
            == canonical_evidence_key(value)
        )

    if dimension == "limitation":
        return any(
            _phrase_matches(
                value,
                f"{item.value} {item.evidence_text}",
            )
            for item in record.limitations
        )

    if dimension == "future_work":
        return any(
            _phrase_matches(
                value,
                f"{item.value} {item.evidence_text}",
            )
            for item in record.future_work
        )

    return False


def _record_matches_idea_topic(
    idea: ResearchIdea,
    record: PaperEvidence,
) -> bool:
    """Keep candidate verification tied to the original research context."""

    problems = _idea_problems(
        idea
    )

    if (
        problems
        and record.research_objective
        and any(
            _problem_item_matches(
                value,
                record.research_objective,
            )
            for value in problems
        )
    ):
        return True

    context = _idea_context_values(
        idea
    )

    if context and any(
        _context_item_matches(
            value,
            item,
        )
        for value in context
        for item in record.population_or_setting
    ):
        return True

    methods = _idea_methods(
        idea
    )

    if methods and any(
        _method_item_matches(
            value,
            item,
        )
        for value in methods
        for item in record.method_or_intervention
    ):
        return True

    if not (
        problems
        or context
        or methods
    ):
        return bool(
            set(_tokens(_idea_all_terms(idea)))
            & set(_tokens(_record_text(record)))
        )

    return False


# ---------------------------------------------------------------------------
# Candidate counterexample confirmation
# ---------------------------------------------------------------------------


def _explicit_comparison_covers_families(
    candidate_families: set[str],
    record: PaperEvidence,
) -> bool:
    """Require every candidate method family in one explicit comparison."""

    methods = [
        item
        for item in record.method_or_intervention
        if is_concrete_entity(item.value)
    ]

    baselines = [
        item
        for item in record.comparison_or_baseline
        if is_concrete_entity(item.value)
    ]

    method_families = {
        normalize_method_family(
            item.value
        )
        for item in methods
    }

    baseline_families = {
        normalize_method_family(
            item.value
        )
        for item in baselines
    }

    method_families.discard("")
    baseline_families.discard("")

    observed = (
        method_families
        | baseline_families
    )

    if not candidate_families <= observed:
        return False

    # Standard focal-method vs baseline representation.
    if (
        candidate_families
        & method_families
        and candidate_families
        & baseline_families
    ):
        return True

    # Neutral head-to-head studies can place all compared methods in the
    # primary-method field. Require one structured evidence statement that
    # explicitly expresses the comparison and names every family through one
    # of its actually extracted method entities.
    family_aliases: dict[
        str,
        list[str],
    ] = {}

    for item in [
        *methods,
        *baselines,
    ]:
        family = normalize_method_family(
            item.value
        )

        if family in candidate_families:
            family_aliases.setdefault(
                family,
                []
            ).append(
                item.value
            )

    relationship_items = [
        *methods,
        *baselines,
        *record.main_findings,
    ]

    for item in relationship_items:
        text = (
            f"{item.value} "
            f"{item.evidence_text}"
        )

        if not _COMPARISON_RELATION_PATTERN.search(
            text
        ):
            continue

        if all(
            any(
                _phrase_matches(
                    alias,
                    text,
                )
                for alias in family_aliases.get(
                    family,
                    []
                )
            )
            for family in candidate_families
        ):
            return True

    return False


def _narrow_dataset_counterexample(
    candidate: GapCandidate,
    record: PaperEvidence,
) -> bool:
    dominant_types = {
        canonical_evidence_key(
            item.value
        )
        for item in candidate.landscape_basis
        if item.dimension == "dataset_type"
        and is_concrete_entity(item.value)
    }

    if not dominant_types:
        return False

    observed_types = {
        canonical_evidence_key(
            value
        )
        for value in dataset_types(record)
        if is_concrete_entity(value)
    }

    return bool(
        observed_types
        - dominant_types
    )


def _resolution_counterexample(
    candidate: GapCandidate,
    record: PaperEvidence,
) -> bool:
    """Require actual study/evaluation evidence, not another limitation."""

    experimental = _experimental_items(
        record
    )

    text = " ".join(
        f"{item.value} {item.evidence_text}"
        for _, item in experimental
    )

    if not _STUDY_ACTION_PATTERN.search(
        text
    ):
        return False

    basis_terms = [
        item.value
        for item in candidate.landscape_basis
        if item.dimension in {
            "limitation",
            "future_work",
            "problem",
            "population_or_setting",
            "method",
            "method_family",
            "constraint",
            "dataset",
            "dataset_type",
            "outcome",
        }
    ]

    if not basis_terms:
        basis_terms = [
            candidate.title,
        ]

    return any(
        _phrase_matches(
            value,
            text,
        )
        for value in basis_terms
    )


def _conflict_resolution_counterexample(
    candidate: GapCandidate,
    record: PaperEvidence,
) -> bool:
    text = _experimental_record_text(
        record
    )

    if not _CONFLICT_RESOLUTION_PATTERN.search(
        text
    ):
        return False

    candidate_terms = set(
        _tokens(candidate.title)
    )

    record_terms = set(
        _tokens(text)
    )

    return bool(
        candidate_terms
        & record_terms
    )


def _direct_counterexample(
    candidate: GapCandidate,
    record: PaperEvidence,
    idea: ResearchIdea,
) -> bool:
    """Confirm only pattern-specific evidence with correct structured fields."""

    if not _record_matches_idea_topic(
        idea,
        record,
    ):
        return False

    pattern = candidate.pattern_type

    if pattern == "combination_gap":
        basis = [
            item
            for item in candidate.landscape_basis
            if is_concrete_entity(item.value)
        ]

        if len(basis) < 2:
            return False

        return all(
            _basis_requirement_matches(
                item.dimension,
                item.value,
                record,
            )
            for item in basis
        )

    if pattern == "missing_comparison":
        families = {
            normalize_method_family(
                item.value
            )
            for item in candidate.landscape_basis
            if item.dimension in {
                "method",
                "method_family",
            }
            and is_concrete_entity(item.value)
        }

        families.discard("")

        return (
            len(families) >= 2
            and _explicit_comparison_covers_families(
                families,
                record,
            )
        )

    if pattern == "narrow_dataset_setting":
        return _narrow_dataset_counterexample(
            candidate,
            record,
        )

    if pattern in {
        "repeated_limitation",
        "repeated_future_work",
        "underrepresented_population",
        "limited_external_validation",
        "limited_real_world_validation",
        "evaluation_gap",
        "replication_gap",
        "method_domain_transfer",
    }:
        return _resolution_counterexample(
            candidate,
            record,
        )

    if pattern == "comparable_conflict":
        return _conflict_resolution_counterexample(
            candidate,
            record,
        )

    # Unknown/unsupported gap patterns are never promoted to confirmed
    # contradictions merely by lexical similarity.
    return False


# ---------------------------------------------------------------------------
# Candidate contextual evidence
# ---------------------------------------------------------------------------


def _candidate_text(
    candidate: GapCandidate,
    idea: ResearchIdea,
) -> str:
    return " ".join(
        [
            candidate.title,
            candidate.description,
            *(
                item.value
                for item in candidate.landscape_basis
            ),
            *idea.problem,
            *idea.domain,
            *idea.population,
            *idea.intervention_or_method,
        ]
    )


def _contextual_match(
    candidate: GapCandidate,
    record: PaperEvidence,
    idea: ResearchIdea,
) -> bool:
    """Require meaningful candidate evidence, not generic vocabulary overlap."""

    if not _record_matches_idea_topic(
        idea,
        record,
    ):
        return False

    basis = [
        item
        for item in candidate.landscape_basis
        if is_concrete_entity(item.value)
    ]

    if basis:
        return any(
            _basis_requirement_matches(
                item.dimension,
                item.value,
                record,
            )
            for item in basis
        )

    generic_terms = {
        "method",
        "methods",
        "model",
        "models",
        "constraint",
        "constraints",
        "task",
        "study",
        "studies",
        "evaluation",
        "comparison",
        "dataset",
        "datasets",
        "population",
        "setting",
        "approach",
        "approaches",
    }

    candidate_terms = (
        set(
            _tokens(
                candidate.title
                + " "
                + candidate.description
            )
        )
        - generic_terms
    )

    record_terms = (
        set(
            _tokens(
                _record_text(record)
            )
        )
        - generic_terms
    )

    return bool(
        candidate_terms
        and record_terms
        and len(
            candidate_terms
            & record_terms
        )
        >= 2
    )


def _relevant_items(
    candidate: GapCandidate,
    record: PaperEvidence,
    idea: ResearchIdea,
) -> list[
    tuple[
        str,
        EvidenceItem,
    ]
]:
    terms = set(
        _tokens(
            _candidate_text(
                candidate,
                idea,
            )
        )
    )

    result: list[
        tuple[
            str,
            EvidenceItem,
        ]
    ] = []

    for field_name, item in _record_items(
        record
    ):
        item_terms = set(
            _tokens(
                f"{item.value} {item.evidence_text}"
            )
        )

        if terms & item_terms:
            result.append(
                (
                    field_name,
                    item,
                )
            )

    if result:
        return result

    # Keep some inspected structured evidence for transparency when semantic
    # normalization established the match without literal token overlap.
    for field_name, item in _experimental_items(
        record
    )[:2]:
        result.append(
            (
                field_name,
                item,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Title-only potential matching
# ---------------------------------------------------------------------------


def _title_possible_match(
    candidate: GapCandidate,
    paper: Paper,
) -> bool:
    candidate_terms = set(
        _tokens(
            " ".join(
                [
                    candidate.title,
                    candidate.description,
                    *(
                        item.value
                        for item in candidate.landscape_basis
                    ),
                ]
            )
        )
    )

    title_terms = set(
        _tokens(paper.title)
    )

    required = max(
        2,
        min(
            4,
            len(candidate_terms),
        ),
    )

    return (
        len(
            candidate_terms
            & title_terms
        )
        >= required
    )


def _title_may_match_idea(
    idea: ResearchIdea,
    record: PaperEvidence,
) -> bool:
    idea_terms = set(
        _tokens(
            _idea_all_terms(idea)
        )
    )

    title_terms = set(
        _tokens(record.title)
    )

    return bool(
        idea_terms
        and len(
            idea_terms
            & title_terms
        )
        >= 2
    )


# ---------------------------------------------------------------------------
# Candidate assessment
# ---------------------------------------------------------------------------


def _assessment(
    candidate: GapCandidate,
    papers: Sequence[Paper],
    records: dict[str, PaperEvidence],
    confirmed: Sequence[str],
    potential: Sequence[str],
    failures: Sequence[VerificationFailure],
) -> tuple[
    GapAssessmentLabel,
    str,
]:
    if failures:
        return (
            "uncertain",
            (
                "Targeted verification had provider or extraction failures, "
                "so absence of a counterexample cannot be interpreted safely."
            ),
        )

    if papers and not records:
        return (
            "uncertain",
            (
                "Verification retrieved papers but no usable structured "
                "evidence was available for inspection."
            ),
        )

    if papers and all(
        not records.get(
            paper.id
        )
        or not _record_has_abstract_evidence(
            records[paper.id]
        )
        for paper in papers
    ):
        return (
            "uncertain",
            (
                "Available verification evidence is title-only or lacks "
                "usable abstract evidence."
            ),
        )

    if confirmed:
        return (
            "well_studied",
            (
                "Targeted verification found same-paper structured evidence "
                "that directly addresses the candidate hypothesis."
            ),
        )

    if potential:
        return (
            "uncertain",
            (
                "Targeted verification found possible matches, but the "
                "structured evidence was not sufficient for confirmation."
            ),
        )

    if not papers:
        return (
            "uncertain",
            (
                "No papers were retrieved by targeted verification; "
                "retrieval absence is not evidence of a research gap."
            ),
        )

    return (
        "promising_gap",
        (
            "Targeted counterexample searches completed without a confirmed "
            "direct counterexample. This remains a qualified evidence-based "
            "hypothesis, not a claim of global novelty."
        ),
    )


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def _unique_evidence(
    items: Sequence[GapEvidence],
) -> list[GapEvidence]:
    result: list[GapEvidence] = []
    seen: set[
        tuple[
            str,
            str,
            str,
            str,
        ]
    ] = set()

    for item in items:
        key = (
            item.paper_id,
            item.evidence_type,
            canonical_evidence_key(
                item.value
            ),
            item.role,
        )

        if key in seen:
            continue

        seen.add(
            key
        )
        result.append(
            item
        )

    return result


def _apply_verification(
    candidate: GapCandidate,
    verification: GapVerification,
) -> GapCandidate:
    candidate.verification = verification
    candidate.final_label = verification.label

    candidate.potentially_contradicting_paper_ids = list(
        verification.potential_contradiction_paper_ids
    )

    candidate.contradicting_paper_ids = list(
        verification.contradicting_paper_ids
    )

    candidate.potentially_contradicting_evidence = [
        item
        for item in verification.evidence
        if item.role
        == "potential_contradiction"
    ]

    candidate.contradicting_evidence = [
        item
        for item in verification.evidence
        if item.role
        == "confirmed_contradiction"
    ]

    candidate.verification_status = "verified"

    return candidate


__all__ = [
    "GapVerifier",
    "build_idea_verification_queries",
    "build_verification_queries",
]
