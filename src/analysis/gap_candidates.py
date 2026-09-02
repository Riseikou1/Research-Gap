"""Deterministic, landscape-grounded Milestone-6 candidate generation.

Candidate gaps are derived only from structured evidence and observations
already present in LiteratureLandscape. No domain-specific scientific entities
or named methods/datasets are hard-coded here.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations, product as _product

from src.extraction.evidence import EvidenceItem, PaperEvidence, canonical_evidence_key
from src.models.idea import ResearchIdea
from src.models.landscape import LiteratureLandscape

from .comparison import (
    normalize_constraint,
    normalize_dataset,
    normalize_method_family,
    normalize_problem,
)
from .models import GapCandidate, GapEvidence, GapPattern, LandscapeBasis


MIN_SUPPORT_PAPERS = 2
MIN_COVERAGE_PAPERS = 2
MAX_CANDIDATES = 12

_STOPWORDS = {
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "under",
    "with",
}

_GENERIC_BUCKETS = {
    "",
    "other",
    "unknown",
    "misc",
    "miscellaneous",
    "unspecified",
    "uncategorized",
    "none",
    "not specified",
    "not reported",
    "baseline",
    "method",
    "methods",
    "model",
    "models",
    "approach",
    "approaches",
    "technique",
    "techniques",
    "dataset",
    "datasets",
    "setting",
    "settings",
    "population",
    "populations",
}

_GENERIC_PREFIXES = {
    "other",
    "unknown",
    "misc",
    "miscellaneous",
    "unspecified",
    "uncategorized",
}

_PATTERN_ALIASES: dict[str, GapPattern] = {
    "limitation": "repeated_limitation",
    "future_work": "repeated_future_work",
}

# These feature identities are intentionally generic. They mirror the
# structured PaperFeatures representation and bound candidate generation to
# evidence dimensions already supported by the landscape.
_COMBINATION_DIMENSIONS = (
    "problem",
    "population_or_setting",
    "method",
    "method_family",
    "dataset",
    "dataset_type",
    "baseline",
    "constraint",
)

_COMBINATION_FIELDS = {
    "problem": "problems",
    "population_or_setting": "populations_or_settings",
    "method": "methods",
    "method_family": "method_families",
    "dataset": "datasets",
    "dataset_type": "dataset_types",
    "baseline": "baselines",
    "constraint": "constraints",
}

_MAX_VALUES_PER_DIMENSION = 4

_COMPARISON_LANGUAGE = re.compile(
    r"\b(?:"
    r"compar(?:e|ed|ing|ison)|"
    r"versus|vs|"
    r"against|"
    r"benchmark(?:ed|ing)?|"
    r"relative\s+to|"
    r"outperform(?:s|ed|ing)?|"
    r"underperform(?:s|ed|ing)?"
    r")\b",
    re.I,
)

_LIMITED_LABEL_CONCEPT = re.compile(
    r"\b(?:limited\s+labeled\s+data|few\s+shot|label\s+scarcity|"
    r"annotation\s+scarcity|label\s+budget|annotation\s+budget)\b",
    re.I,
)

_GENERALIZATION_CONCEPT = re.compile(
    r"\b(?:generalization|generalisation|external\s+validation|"
    r"out\s+of\s+distribution|cross\s+domain)\b",
    re.I,
)

_EFFICIENCY_CONCEPT = re.compile(
    r"\b(?:efficiency|inference\s+time|training\s+time|latency|"
    r"computational\s+cost|memory\s+usage|parameter\s+count|flops)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Generic text helpers
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    normalized = canonical_evidence_key(text)

    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in _STOPWORDS
    }


def _concept_key(text: str) -> str:
    """Create a conservative identity key without dropping phrase words."""

    # Stopword removal is useful for relevance scoring, but not for identity:
    # Structured values must remain distinct even when they share a prefix.
    return canonical_evidence_key(text)


def is_concrete_entity(value: str) -> bool:
    """Reject placeholders while preserving arbitrary scientific entities."""

    key = canonical_evidence_key(value)

    if not key or key in _GENERIC_BUCKETS:
        return False

    words = key.split()

    if words and words[0] in _GENERIC_PREFIXES:
        return False

    return True


# ---------------------------------------------------------------------------
# Evidence semantic guardrails
# ---------------------------------------------------------------------------


def validate_evidence_semantics(
    *,
    claim_text: str,
    evidence_type: str,
    evidence: EvidenceItem,
) -> bool:
    """Reject a small set of unsafe evidence equivalences.

    This function deliberately handles only structural mistakes that can be
    checked deterministically. It does not attempt domain understanding.
    """

    claim = canonical_evidence_key(claim_text)
    evidence_text = canonical_evidence_key(
        f"{evidence.value} {evidence.evidence_text}"
    )
    field = evidence_type.casefold().replace(" ", "_")

    if not evidence_text:
        return False

    # A limited-label claim must come from actual constraint evidence and must
    # contain an explicit low-label concept.
    if _LIMITED_LABEL_CONCEPT.search(claim):
        if field not in {"constraint", "constraints"}:
            return False

        if normalize_constraint(evidence.value) != normalize_constraint(claim_text):
            return False

    # A class count alone is not a sample size.
    if field == "sample_size" and re.search(r"\bclasses?\b", evidence_text):
        if not re.search(
            r"\b(?:samples?|participants?|patients?|records?|instances?|"
            r"examples?|images?|documents?)\b",
            evidence_text,
        ):
            return False

    # Generic performance numbers do not prove generalization.
    if _GENERALIZATION_CONCEPT.search(claim):
        if not _GENERALIZATION_CONCEPT.search(evidence_text):
            return False

    # A limitation mentioning computational burden is not automatically an
    # experimentally measured efficiency metric.
# Measured efficiency claims require evaluation/metric evidence.
    if _EFFICIENCY_CONCEPT.search(claim):
        if field not in {
            "evaluation",
            "evaluation_metrics",
            "metric",
            "metrics",
        }:
            return False

        return bool(
            re.search(
                r"\b(?:time|latency|cost|flops?|parameters?|memory|"
                r"overhead|complexity)\b",
                evidence_text,
            )
        )

    return True


@dataclass(slots=True)
class _RecordGenerationContext:
    """Immutable-in-practice prepared state for one evidence record."""

    record: PaperEvidence
    fields: dict[str, tuple[EvidenceItem, ...]]
    item_tokens: dict[int, frozenset[str]]
    normalized_values: dict[tuple[int, str], str]
    objective_problem: str
    method_families: frozenset[str]
    baseline_families: frozenset[str]
    context_terms: frozenset[str]
    full_terms: frozenset[str]


@dataclass(slots=True)
class _GenerationContext:
    """Per-generate-call indexes; nothing is shared across analyses."""

    idea: ResearchIdea
    landscape: LiteratureLandscape
    records: dict[str, PaperEvidence]
    idea_problem_values: frozenset[str] = field(init=False)
    idea_problem_normalized: frozenset[str] = field(init=False)
    idea_basis_problem_normalized: frozenset[str] = field(init=False)
    idea_method_families: frozenset[str] = field(init=False)
    idea_comparison_families: frozenset[str] = field(init=False)
    idea_constraint_values: frozenset[str] = field(init=False)
    idea_dataset_values: frozenset[str] = field(init=False)
    idea_terms: frozenset[str] = field(init=False)
    idea_context_terms: frozenset[str] = field(init=False)
    records_by_id: dict[str, _RecordGenerationContext] = field(init=False)
    item_contexts: dict[int, _RecordGenerationContext] = field(init=False)
    paper_dimension_values: tuple[tuple[str, dict[str, frozenset[str]]], ...] = field(init=False)
    canonical_cache: dict[str, str] = field(default_factory=dict, init=False)
    token_cache: dict[str, frozenset[str]] = field(default_factory=dict, init=False)
    basis_match_cache: dict[tuple[str, str], bool] = field(default_factory=dict, init=False)
    strong_context_cache: dict[tuple[str, str], bool] = field(default_factory=dict, init=False)
    record_anchor_cache: dict[tuple[str, bool], bool] = field(default_factory=dict, init=False)
    relevance_cache: dict[tuple[tuple[str, str], ...], bool] = field(default_factory=dict, init=False)
    semantic_cache: dict[tuple[str, str, int], bool] = field(default_factory=dict, init=False)
    normalization_cache: dict[tuple[str, str], str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        idea = self.idea
        self.idea_problem_values = frozenset(
            value for value in _idea_problem_values(idea) if self.is_concrete(value)
        )
        self.idea_problem_normalized = frozenset(
            normalized
            for value in self.idea_problem_values
            if (normalized := self.normalize_value("problem", value))
        )
        self.idea_basis_problem_normalized = frozenset(
            normalized
            for value in idea.problem
            if (normalized := self.normalize_value("problem", value))
        )
        self.idea_method_families = frozenset(
            normalized
            for value in idea.intervention_or_method
            if self.is_concrete(value)
            and (normalized := self.normalize_value("method_family", value))
        )
        self.idea_comparison_families = frozenset(
            normalized
            for value in (*idea.intervention_or_method, *idea.comparison)
            if (normalized := self.normalize_value("method_family", value))
        )
        self.idea_constraint_values = frozenset(
            normalized
            for value in idea.constraints
            if (normalized := self.normalize_value("constraint", value))
        )
        self.idea_dataset_values = frozenset(
            normalized
            for value in (*idea.keywords, *idea.domain, *idea.population)
            if (normalized := self.normalize_value("dataset", value))
        )
        self.idea_terms = frozenset(_tokens_uncached(_idea_text(idea)))
        self.idea_context_terms = frozenset(
            _tokens_uncached(
                " ".join((*idea.problem, *idea.domain, *idea.population))
            )
        )

        self.records_by_id = {
            paper_id: self._prepare_record(record)
            for paper_id, record in self.records.items()
        }
        self.item_contexts = {
            item_id: prepared
            for prepared in self.records_by_id.values()
            for item_id in prepared.item_tokens
        }
        self.paper_dimension_values = tuple(
            (
                paper.paper_id,
                {
                    dimension: frozenset(
                        self.concept_key(value)
                        for value in getattr(paper, field_name, [])
                        if self.is_concrete(value)
                    )
                    for dimension, field_name in _COMBINATION_FIELDS.items()
                },
            )
            for paper in self.landscape.papers
        )

    def canonical(self, value: str) -> str:
        cached = self.canonical_cache.get(value)
        if cached is None:
            cached = canonical_evidence_key(value)
            self.canonical_cache[value] = cached
        return cached

    def concept_key(self, value: str) -> str:
        return self.canonical(value)

    def normalize_value(self, dimension: str, value: str) -> str:
        kind = (
            "method_family"
            if dimension in {"method", "method_family", "baseline", "comparison"}
            else dimension
        )
        key = (kind, value)
        cached = self.normalization_cache.get(key)
        if cached is not None:
            return cached
        if kind == "problem":
            normalized = normalize_problem(value)
        elif kind == "method_family":
            normalized = normalize_method_family(value)
        elif kind == "constraint":
            normalized = normalize_constraint(value)
        elif kind == "dataset":
            normalized = normalize_dataset(value)
        else:
            normalized = self.canonical(value)
        self.normalization_cache[key] = normalized
        return normalized

    def tokens(self, value: str) -> frozenset[str]:
        cached = self.token_cache.get(value)
        if cached is None:
            cached = frozenset(_tokens_uncached(value, canonical=self.canonical(value)))
            self.token_cache[value] = cached
        return cached

    def is_concrete(self, value: str) -> bool:
        key = self.canonical(value)
        if not key or key in _GENERIC_BUCKETS:
            return False
        words = key.split()
        return not words or words[0] not in _GENERIC_PREFIXES

    def _prepare_record(self, record: PaperEvidence) -> _RecordGenerationContext:
        fields = {
            dimension: tuple(_field_items(record, dimension))
            for dimension in (
                "problem",
                "population_or_setting",
                "method",
                "method_family",
                "constraint",
                "limitation",
                "future_work",
                "finding",
                "outcome",
                "comparison",
                "baseline",
                "dataset",
                "dataset_type",
            )
        }
        all_items = {
            id(item): item
            for values in fields.values()
            for item in values
        }
        item_tokens = {
            item_id: self.tokens(f"{item.value} {item.evidence_text}")
            for item_id, item in all_items.items()
        }
        normalized_values = {
            (item_id, dimension): self.normalize_value(dimension, item.value)
            for item_id, item in all_items.items()
            for dimension in ("problem", "method", "method_family", "baseline", "comparison", "constraint", "dataset")
        }
        objective_problem = (
            normalized_values.get((id(record.research_objective), "problem"), "")
            if record.research_objective
            else ""
        )
        method_families = frozenset(
            normalized_values[(id(item), "method_family")]
            for item in fields["method"]
            if self.is_concrete(item.value)
            and normalized_values[(id(item), "method_family")]
        )
        baseline_families = frozenset(
            normalized_values[(id(item), "baseline")]
            for item in fields["baseline"]
            if self.is_concrete(item.value)
            and normalized_values[(id(item), "baseline")]
        )
        context_source = " ".join(
            (
                f"{record.research_objective.value} {record.research_objective.evidence_text}"
                if record.research_objective
                else "",
                *(
                    f"{item.value} {item.evidence_text}"
                    for item in fields["population_or_setting"]
                ),
            )
        )
        return _RecordGenerationContext(
            record=record,
            fields=fields,
            item_tokens=item_tokens,
            normalized_values=normalized_values,
            objective_problem=objective_problem,
            method_families=method_families,
            baseline_families=baseline_families,
            context_terms=frozenset(self.tokens(context_source)),
            full_terms=frozenset(self.tokens(_record_text(record))),
        )

    def _normalize_item(self, item: EvidenceItem, dimension: str) -> str:
        return self.normalize_value(dimension, item.value)

    def item_tokens_for(self, item: EvidenceItem) -> frozenset[str]:
        prepared = self.item_contexts.get(id(item))
        if prepared is not None:
            return prepared.item_tokens[id(item)]
        return self.tokens(f"{item.value} {item.evidence_text}")

    def normalized_item(self, item: EvidenceItem, dimension: str) -> str:
        prepared = self.item_contexts.get(id(item))
        if prepared is not None:
            normalized = prepared.normalized_values.get((id(item), dimension))
            if normalized is not None:
                return normalized
        return self._normalize_item(item, dimension)

    def field_items(self, record: PaperEvidence, dimension: str) -> tuple[EvidenceItem, ...]:
        prepared = self.records_by_id.get(record.paper_id)
        return prepared.fields.get(dimension, ()) if prepared else tuple(_field_items(record, dimension))

    def record_anchor(self, record: PaperEvidence, *, require_method_and_problem: bool = False) -> bool:
        key = (record.paper_id, require_method_and_problem)
        cached = self.record_anchor_cache.get(key)
        if cached is not None:
            return cached
        prepared = self.records_by_id[record.paper_id]
        problem_match = bool(
            self.idea_problem_normalized
            and prepared.objective_problem in self.idea_problem_normalized
        )
        method_match = bool(self.idea_method_families & prepared.method_families)
        if require_method_and_problem:
            result = problem_match and method_match if self.idea_problem_values and self.idea_method_families else False
        else:
            available_signals = []
            if self.idea_problem_values:
                available_signals.append(problem_match)
            if self.idea_method_families:
                available_signals.append(method_match)
            if self.idea.population or self.idea.domain:
                available_signals.append(bool(self.idea_context_terms & prepared.context_terms))
            result = bool(prepared.full_terms & self.idea_terms) if not available_signals else any(available_signals)
        self.record_anchor_cache[key] = result
        return result

    def basis_matches(self, item: LandscapeBasis) -> bool:
        key = (item.dimension, self.concept_key(item.value))
        cached = self.basis_match_cache.get(key)
        if cached is not None:
            return cached
        dimension = item.dimension
        target = self._normalize_basis(item.value, dimension)
        if dimension in {"method", "method_family", "baseline"}:
            result = bool(target and target in self.idea_comparison_families)
        elif dimension == "problem":
            result = bool(target and target in self.idea_basis_problem_normalized)
        elif dimension == "constraint":
            result = bool(target and target in self.idea_constraint_values)
        elif dimension == "dataset":
            result = bool(target and target in self.idea_dataset_values)
        else:
            terms = self.tokens(item.value)
            result = bool(terms and terms <= self.idea_terms)
        self.basis_match_cache[key] = result
        return result

    def _normalize_basis(self, value: str, dimension: str) -> str:
        return self.normalize_value(dimension, value)

    def item_matches_basis(self, dimension: str, basis_value: str, item: EvidenceItem) -> bool:
        if dimension == "problem":
            return self._normalize_basis(basis_value, dimension) == self.normalized_item(item, dimension)
        if dimension in {"method", "method_family", "baseline", "comparison"}:
            target = self._normalize_basis(basis_value, dimension)
            observed = self.normalized_item(item, dimension)
            return bool(target and observed and target == observed)
        if dimension in {"constraint", "dataset"}:
            target = self._normalize_basis(basis_value, dimension)
            observed = self.normalized_item(item, dimension)
            return bool(target and observed and target == observed)
        target_terms = self.tokens(basis_value)
        observed_terms = self.item_tokens_for(item)
        return bool(target_terms) and (target_terms <= observed_terms or bool(target_terms & observed_terms))

    def strong_context(self, item: LandscapeBasis) -> bool:
        key = (item.dimension, self.concept_key(item.value))
        cached = self.strong_context_cache.get(key)
        if cached is not None:
            return cached
        relevant_ids = [
            paper_id
            for paper_id in item.paper_ids
            if paper_id in self.records
            and self.record_anchor(self.records[paper_id])
        ]
        result = len(relevant_ids) >= MIN_SUPPORT_PAPERS
        if not result:
            for paper_id in relevant_ids:
                record = self.records[paper_id]
                if any(self.item_matches_basis(item.dimension, item.value, claim) for claim in self.field_items(record, item.dimension)):
                    if self.record_anchor(record, require_method_and_problem=True):
                        result = True
                        break
                basis_terms = self.tokens(item.value)
                if any(basis_terms & self.item_tokens_for(claim) for claim in (*record.limitations, *record.future_work)):
                    result = True
                    break
        self.strong_context_cache[key] = result
        return result

    def candidate_relevant(self, basis: Sequence[LandscapeBasis]) -> bool:
        key = tuple((item.dimension, self.concept_key(item.value)) for item in basis)
        cached = self.relevance_cache.get(key)
        if cached is not None:
            return cached
        if not basis:
            result = True
        else:
            anchored = [item for item in basis if self.basis_matches(item)]
            if not anchored:
                result = len(basis) <= 1 and any(self.strong_context(item) for item in basis)
            else:
                result = all(self.strong_context(item) for item in basis if item not in anchored)
        self.relevance_cache[key] = result
        return result

    def has_idea_context(self, paper_ids: Iterable[str]) -> bool:
        if not self.idea_context_terms:
            return True
        return any(
            paper_id in self.records
            and bool(self.idea_context_terms & self.records_by_id[paper_id].context_terms)
            for paper_id in paper_ids
        )

    def semantic_valid(self, claim_text: str, evidence_type: str, item: EvidenceItem) -> bool:
        key = (self.canonical(claim_text), evidence_type, id(item))
        cached = self.semantic_cache.get(key)
        if cached is None:
            cached = validate_evidence_semantics(
                claim_text=claim_text,
                evidence_type=evidence_type,
                evidence=item,
            )
            self.semantic_cache[key] = cached
        return cached

    def observed_combination(self, basis: Sequence[LandscapeBasis]) -> bool:
        if not basis:
            return False
        return self.observed_combination_keys(
            tuple((item.dimension, self.concept_key(item.value)) for item in basis)
        )

    def observed_combination_keys(
        self,
        combination_key: Sequence[tuple[str, str]],
    ) -> bool:
        if not combination_key:
            return False
        return any(
            all(
                value in values.get(dimension, frozenset())
                for dimension, value in combination_key
            )
            for _, values in self.paper_dimension_values
        )


def _tokens_uncached(text: str, *, canonical: str | None = None) -> set[str]:
    normalized = canonical if canonical is not None else canonical_evidence_key(text)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if token not in _STOPWORDS
    }


def _idea_text(idea: ResearchIdea) -> str:
    return " ".join(
        (
            idea.original_text,
            *idea.problem,
            *idea.population,
            *idea.intervention_or_method,
            *idea.comparison,
            *idea.outcomes,
            *idea.domain,
            *idea.constraints,
            *idea.keywords,
        )
    )


# ---------------------------------------------------------------------------
# Candidate generator
# ---------------------------------------------------------------------------


class GapCandidateGenerator:
    """Generate candidates grounded in the deterministic literature landscape."""

    def __init__(self, *, max_candidates: int = MAX_CANDIDATES) -> None:
        if not 1 <= max_candidates <= 50:
            raise ValueError("max_candidates must be between 1 and 50")

        self.max_candidates = max_candidates
        self.notices: list[str] = []
        self._metrics: dict[str, int] = {
            "candidate_hypotheses_generated": 0,
            "candidate_hypotheses_after_pruning": 0,
        }

    def metrics_snapshot(self) -> dict[str, int]:
        return dict(self._metrics)

    def generate(
        self,
        idea: ResearchIdea,
        landscape: LiteratureLandscape,
        evidence: Sequence[PaperEvidence],
    ) -> list[GapCandidate]:
        self.notices = []
        self._metrics = {
            "candidate_hypotheses_generated": 0,
            "candidate_hypotheses_after_pruning": 0,
        }

        records = {
            item.paper_id: item
            for item in evidence
        }

        if not records or landscape.total_papers == 0:
            self.notices.append(
                "insufficient_evidence: no landscape-backed evidence is available"
            )
            return []

        context = _GenerationContext(
            idea=idea,
            landscape=landscape,
            records=records,
        )

        candidates: list[GapCandidate] = []

        candidates.extend(
            self._conflict_candidates(
                landscape,
                records,
                idea,
                context,
            )
        )
        candidates.extend(
            self._repeated_candidates(
                landscape,
                records,
                idea,
                "limitation",
                context,
            )
        )
        candidates.extend(
            self._repeated_candidates(
                landscape,
                records,
                idea,
                "future_work",
                context,
            )
        )
        candidates.extend(
            self._combination_candidates(
                idea,
                landscape,
                records,
                context,
            )
        )
        candidates.extend(
            self._narrow_setting_candidates(
                idea,
                landscape,
                records,
                context,
            )
        )
        candidates.extend(
            self._comparison_candidates(
                idea,
                landscape,
                records,
                context,
            )
        )

        self._metrics["candidate_hypotheses_generated"] = len(candidates)
        candidates = consolidate_candidates(candidates)
        # Keep the generator's complete deterministic hypothesis set
        # inspectable; dominance pruning is applied immediately before
        # verification, where it can reduce expensive work without hiding
        # valid landscape hypotheses from callers.
        self._metrics["candidate_hypotheses_after_pruning"] = len(candidates)

        if len(candidates) > self.max_candidates:
            self.notices.append(
                f"candidate_limit: retained first "
                f"{self.max_candidates} deterministic candidates"
            )

        return candidates[: self.max_candidates]

    # ------------------------------------------------------------------
    # Conflicting findings
    # ------------------------------------------------------------------

    def _conflict_candidates(
        self,
        landscape: LiteratureLandscape,
        records: dict[str, PaperEvidence],
        idea: ResearchIdea,
        context: _GenerationContext,
    ) -> list[GapCandidate]:
        result: list[GapCandidate] = []

        for conflict in landscape.conflicts:
            if not context.is_concrete(conflict.topic):
                self.notices.append(
                    f"invalid_entity: skipped conflict candidate "
                    f"for {conflict.topic}"
                )
                continue

            support = self._collect_evidence(
                conflict.paper_ids,
                "finding",
                conflict.topic,
                records,
                role="direct_support",
                claim_text=conflict.topic,
                context=context,
            )

            candidate = self._build(
                title=(
                    f"Conditions behind conflicting findings on "
                    f"{conflict.topic}"
                ),
                description=(
                    "Comparable retrieved studies report opposing findings. "
                    "The conditions responsible for the disagreement remain "
                    "a verification hypothesis."
                ),
                category="contradiction",
                rationale=conflict.reason,
                pattern_type="comparable_conflict",
                support=support,
                basis=[
                    LandscapeBasis(
                        dimension="conflict",
                        value=conflict.topic,
                        count=len(conflict.paper_ids),
                        total=landscape.total_papers,
                        prevalence=(
                            len(conflict.paper_ids)
                            / max(1, landscape.total_papers)
                        ),
                        paper_ids=list(conflict.paper_ids),
                    )
                ],
                idea=idea,
                records=records,
                context=context,
            )

            if candidate:
                result.append(candidate)

        return result

    # ------------------------------------------------------------------
    # Repeated author-stated limitations / future work
    # ------------------------------------------------------------------

    def _repeated_candidates(
        self,
        landscape: LiteratureLandscape,
        records: dict[str, PaperEvidence],
        idea: ResearchIdea,
        dimension: str,
        context: _GenerationContext,
    ) -> list[GapCandidate]:
        frequencies = [
            item
            for item in landscape.frequencies
            if item.dimension == dimension
        ]

        grouped: dict[str, list] = defaultdict(list)

        for item in frequencies:
            key = context.concept_key(item.value)

            if key:
                grouped[key].append(item)

        result: list[GapCandidate] = []

        for key, items in grouped.items():
            if not context.is_concrete(key):
                self.notices.append(
                    f"invalid_entity: skipped repeated "
                    f"{dimension} candidate for {key}"
                )
                continue

            paper_ids = list(
                dict.fromkeys(
                    paper_id
                    for item in items
                    for paper_id in item.paper_ids
                    if paper_id in records
                )
            )

            if len(paper_ids) < MIN_SUPPORT_PAPERS:
                continue

            missing_field = (
                "limitations"
                if dimension == "limitation"
                else "future_work"
            )

            available = (
                landscape.total_papers
                - landscape.missing_field_counts.get(
                    missing_field,
                    landscape.total_papers,
                )
            )

            if available < MIN_COVERAGE_PAPERS:
                self.notices.append(
                    f"coverage_limited: skipped repeated "
                    f"{dimension} candidate for {key}"
                )
                continue

            value = items[0].value
            support = self._collect_evidence(
                paper_ids,
                dimension,
                value,
                records,
                role="direct_support",
                claim_text=value,
                context=context,
            )

            candidate = self._build(
                title=(
                    f"Repeated author-stated limitation: {value}"
                    if dimension == "limitation"
                    else f"Repeated future-work direction: {value}"
                ),
                description=(
                    f"{value} recurs across {len(paper_ids)} analyzed "
                    "papers. This supports targeted follow-up but does not "
                    "establish global absence."
                ),
                category=dimension,
                rationale=(
                    "The direction recurs in explicit structured evidence "
                    "from multiple analyzed papers and therefore warrants "
                    "targeted verification."
                ),
                pattern_type=_PATTERN_ALIASES[dimension],
                support=support,
                basis=[
                    self._basis(item, landscape.total_papers)
                    for item in items
                ],
                idea=idea,
                records=records,
                context=context,
            )

            if candidate:
                result.append(candidate)

        return result

    # ------------------------------------------------------------------
    # Missing observed combinations
    # ------------------------------------------------------------------

    def _combination_candidates(
        self,
        idea: ResearchIdea,
        landscape: LiteratureLandscape,
        records: dict[str, PaperEvidence],
        context: _GenerationContext,
    ) -> list[GapCandidate]:
        frequencies_by_dimension = {
            dimension: sorted(
                (
                    item
                    for item in landscape.frequencies
                    if item.dimension == dimension
                    and item.count >= 1
                    and context.is_concrete(item.value)
                ),
                key=lambda item: (
                    -int(context.basis_matches(self._basis(item, landscape.total_papers))),
                    -item.count,
                    item.value.casefold(),
                ),
            )[:_MAX_VALUES_PER_DIMENSION]
            for dimension in _COMBINATION_DIMENSIONS
        }

        available_dimensions = [
            dimension
            for dimension in _COMBINATION_DIMENSIONS
            if frequencies_by_dimension[dimension]
        ]
        if (
            "method_family" in available_dimensions
            and "method" in available_dimensions
        ):
            # The family is the comparison-oriented normalized identity used
            # by the landscape. Do not emit a second candidate for its raw
            # method representation.
            available_dimensions.remove("method")

        idea_dimensions = {
            dimension
            for dimension in available_dimensions
            if any(
                context.basis_matches(item)
                for item in frequencies_by_dimension[dimension]
            )
        }

        # Try informative higher-order combinations first, then retain
        # pairwise candidates as useful lower-order fallbacks.
        dimension_sets = [
            selected
            for size in range(
                min(5, len(available_dimensions)),
                1,
                -1,
            )
            for selected in combinations(available_dimensions, size)
            if len(set(selected) & idea_dimensions) >= min(2, size)
        ]

        ranked_candidates: list[tuple[tuple[object, ...], GapCandidate]] = []
        seen_combinations: set[tuple[tuple[str, str], ...]] = set()
        candidate_budget = self.max_candidates * 8

        for dimensions in dimension_sets:
            if {
                "method",
                "method_family",
            }.issubset(dimensions):
                # These two fields are alternative representations of the
                # same extracted method claim, not an informative joint
                # hypothesis.
                continue
            value_lists = [frequencies_by_dimension[dimension] for dimension in dimensions]

            # The cartesian product is deliberately bounded per dimension.
            # Only combinations structurally anchored in the decomposed idea
            # are allowed through the relevance check below.
            for selected_items in _product(*value_lists):
                combination_key = tuple(
                    (item.dimension, context.concept_key(item.value))
                    for item in selected_items
                )

                if combination_key in seen_combinations:
                    continue

                seen_combinations.add(combination_key)

                # This is the correctness guard: membership is evaluated on
                # one paper's structured dimensions, not independent counts.
                if context.observed_combination_keys(combination_key):
                    continue

                # Validate against the prepared feature records before
                # constructing Pydantic basis models. This is logically the
                # same relevance decision used by _build(), but most rejected
                # products never need a model allocation.
                if not context.candidate_relevant(selected_items):
                    continue

                basis = [
                    self._basis(item, landscape.total_papers)
                    for item in selected_items
                ]

                support = [
                    evidence_item
                    for item in selected_items
                    for evidence_item in self._collect_basis_evidence(
                        item,
                        records,
                        role="contextual_support",
                        context=context,
                    )
                ]

                title, description = _combination_text_for_basis(basis)
                candidate = self._build(
                    title=title,
                    description=description,
                    category="combination",
                    rationale=(
                        "Each component is represented in the analyzed "
                        "landscape, but this complete structured "
                        "combination was not observed within one analyzed "
                        "paper. Targeted verification is required."
                    ),
                    pattern_type="combination_gap",
                    support=support,
                    basis=basis,
                    idea=idea,
                    records=records,
                    context=context,
                    already_validated=True,
                )

                if candidate:
                    idea_matches = sum(
                        context.basis_matches(item)
                        for item in basis
                    )
                    rank = (
                        -len(basis),
                        -idea_matches,
                        -sum(item.count for item in basis),
                        tuple(
                            (item.dimension, item.value.casefold())
                            for item in basis
                        ),
                    )
                    ranked_candidates.append((rank, candidate))

                    if len(ranked_candidates) >= candidate_budget:
                        break

            if len(ranked_candidates) >= candidate_budget:
                break

        return [
            candidate
            for _, candidate in sorted(
                ranked_candidates,
                key=lambda item: item[0],
            )
        ]

    # ------------------------------------------------------------------
    # Narrow dataset / validation setting
    # ------------------------------------------------------------------

    def _narrow_setting_candidates(
        self,
        idea: ResearchIdea,
        landscape: LiteratureLandscape,
        records: dict[str, PaperEvidence],
        context: _GenerationContext,
    ) -> list[GapCandidate]:
        available_datasets = (
            landscape.total_papers
            - landscape.missing_field_counts.get(
                "datasets",
                landscape.total_papers,
            )
        )

        if available_datasets < MIN_COVERAGE_PAPERS:
            self.notices.append(
                "coverage_limited: skipped dataset-setting candidates "
                "because dataset evidence is sparse"
            )
            return []

        settings = [
            item
            for item in landscape.frequencies
            if item.dimension == "dataset_type"
            and item.count >= MIN_SUPPORT_PAPERS
            and context.is_concrete(item.value)
        ]

        if not settings:
            return []

        dominant = settings[0]

        # Do not call a dataset type narrow when multiple alternative types
        # are themselves repeatedly represented.
        alternatives = [
            item
            for item in settings[1:]
            if item.count >= MIN_SUPPORT_PAPERS
        ]

        if alternatives:
            return []

        coverage = dominant.count / max(
            1,
            available_datasets,
        )

        if coverage < 0.60:
            return []

        support = self._collect_evidence(
            dominant.paper_ids,
            "dataset_type",
            dominant.value,
            records,
            role="contextual_support",
            claim_text=dominant.value,
            context=context,
        )

        candidate = self._build(
            title=f"Validation beyond {dominant.value} data",
            description=(
                f"The analyzed evidence is concentrated in "
                f"{dominant.value} data. Evaluation in substantially "
                "different data or validation settings remains a candidate "
                "question requiring targeted verification."
            ),
            category="dataset",
            rationale=(
                "One generic dataset characteristic dominates the available "
                "structured dataset evidence while alternative validation "
                "settings are not repeatedly represented."
            ),
            pattern_type="narrow_dataset_setting",
            support=support,
            basis=[
                self._basis(
                    dominant,
                    landscape.total_papers,
                )
            ],
            idea=idea,
            records=records,
            context=context,
        )

        return [candidate] if candidate else []

    # ------------------------------------------------------------------
    # Missing matched comparison
    # ------------------------------------------------------------------

    def _comparison_candidates(
        self,
        idea: ResearchIdea,
        landscape: LiteratureLandscape,
        records: dict[str, PaperEvidence],
        context: _GenerationContext,
    ) -> list[GapCandidate]:
        available_baselines = (
            landscape.total_papers
            - landscape.missing_field_counts.get(
                "baselines",
                landscape.total_papers,
            )
        )

        if available_baselines < MIN_COVERAGE_PAPERS:
            return []

        methods = [
            item
            for item in landscape.frequencies
            if item.dimension == "method_family"
            and item.count >= MIN_SUPPORT_PAPERS
            and context.is_concrete(item.value)
        ]

        if len(methods) < 2:
            return []

        observed_pairs = _observed_comparison_pairs(records, context=context)
        result: list[GapCandidate] = []

        for left, right in combinations(methods, 2):
            left_key = context.concept_key(left.value)
            right_key = context.concept_key(right.value)

            if not left_key or not right_key or left_key == right_key:
                continue

            pair = frozenset((left_key, right_key))

            if pair in observed_pairs:
                continue

            if not _comparison_pair_is_justified(
                idea,
                left.value,
                right.value,
                records,
                context=context,
            ):
                continue

            basis = [
                self._basis(left, landscape.total_papers),
                self._basis(right, landscape.total_papers),
            ]

            if not context.candidate_relevant(basis):
                continue

            if not (
                _has_idea_context(
                    left.paper_ids,
                    idea,
                    records,
                    context=context,
                )
                and _has_idea_context(
                    right.paper_ids,
                    idea,
                    records,
                    context=context,
                )
            ):
                continue

            support = (
                self._collect_basis_evidence(
                    left,
                    records,
                    role="contextual_support",
                    context=context,
                )
                + self._collect_basis_evidence(
                    right,
                    records,
                    role="contextual_support",
                    context=context,
                )
            )

            left_supported = _support_contains_family(
                support,
                left.value,
                context=context,
            )
            right_supported = _support_contains_family(
                support,
                right.value,
                context=context,
            )

            if not (left_supported and right_supported):
                continue

            candidate = self._build(
                title=(
                    f"Matched comparison of {left.value} "
                    f"and {right.value}"
                ),
                description=(
                    "Both method families are repeatedly represented in the "
                    "analyzed evidence, but no explicit matched comparison "
                    "between them was observed in the structured comparison "
                    "evidence."
                ),
                category="comparison",
                rationale=(
                    "The candidate is anchored by an explicit comparison "
                    "request or compatible structured comparison evidence, "
                    "while no observed matched comparison covers both "
                    "method families."
                ),
                pattern_type="missing_comparison",
                support=support,
                basis=basis,
                idea=idea,
                records=records,
                context=context,
                already_validated=True,
            )

            if candidate:
                result.append(candidate)

        return result

    # ------------------------------------------------------------------
    # Shared construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _basis(
        item,
        total: int,
    ) -> LandscapeBasis:
        return LandscapeBasis(
            dimension=item.dimension,
            value=item.value,
            count=item.count,
            total=total,
            prevalence=item.prevalence,
            paper_ids=list(item.paper_ids),
        )

    @classmethod
    def _collect_basis_evidence(
        cls,
        item,
        records: dict[str, PaperEvidence],
        *,
        role: str,
        context: _GenerationContext,
    ) -> list[GapEvidence]:
        return cls._collect_evidence(
            item.paper_ids,
            item.dimension,
            item.value,
            records,
            role=role,
            claim_text=item.value,
            context=context,
        )

    @staticmethod
    def _collect_evidence(
        paper_ids: Iterable[str],
        evidence_type: str,
        value: str,
        records: dict[str, PaperEvidence],
        *,
        role: str,
        claim_text: str,
        context: _GenerationContext,
    ) -> list[GapEvidence]:
        result: list[GapEvidence] = []

        for paper_id in dict.fromkeys(paper_ids):
            record = records.get(paper_id)

            if record is None:
                continue

            items = context.field_items(record, evidence_type)

            items = [
                item
                for item in items
                if context.is_concrete(item.value)
            ]

            matched = [
                item
                for item in items
                if _item_matches_basis(
                    evidence_type,
                    value,
                    item,
                    context=context,
                )
            ]

            # Derived dataset-type labels do not necessarily occur literally
            # in the source claim. The original dataset/setting evidence may
            # therefore be retained as contextual provenance.
            if (
                not matched
                and evidence_type == "dataset_type"
                and items
            ):
                matched = items[:1]

            for item in matched[:2]:
                if not context.semantic_valid(claim_text, evidence_type, item):
                    continue

                result.append(
                    GapEvidence(
                        paper_id=paper_id,
                        evidence_type=evidence_type,
                        value=item.value,
                        evidence_text=item.evidence_text,
                        study_type=record.study_type,
                        role=role,
                    )
                )

        return _unique_evidence(result, context=context)

    @staticmethod
    def _build(
        *,
        title: str,
        description: str,
        category: str,
        rationale: str,
        pattern_type: GapPattern,
        support: list[GapEvidence],
        basis: list[LandscapeBasis],
        idea: ResearchIdea,
        records: dict[str, PaperEvidence],
        context: _GenerationContext,
        already_validated: bool = False,
    ) -> GapCandidate | None:
        support = _unique_evidence(support, context=context)

        support_ids = list(
            dict.fromkeys(
                item.paper_id
                for item in support
            )
        )

        if len(support_ids) < MIN_SUPPORT_PAPERS:
            return None

        if not already_validated and not _candidate_is_relevant(
            idea,
            basis,
            records,
            context=context,
        ):
            return None

        candidate_terms = _tokens(
            f"{title} {description}"
        )
        idea_terms = context.idea_terms

        relevance = (
            len(candidate_terms & idea_terms)
            / max(1, len(candidate_terms))
        )

        idea_anchor_score = (
            sum(
                context.basis_matches(item)
                for item in basis
            )
            / max(1, len(basis))
        )

        return GapCandidate(
            title=title,
            description=description,
            category=category,
            rationale=rationale,
            pattern_type=pattern_type,
            supporting_paper_ids=support_ids,
            supporting_evidence=support,
            landscape_basis=basis,
            confidence=1.0,
            idea_relevance=min(
                1.0,
                relevance,
            ),
            idea_anchor_score=min(
                1.0,
                idea_anchor_score,
            ),
        )


# ---------------------------------------------------------------------------
# Landscape/evidence matching
# ---------------------------------------------------------------------------


def _field_items(
    record: PaperEvidence,
    dimension: str,
) -> list[EvidenceItem]:
    mapping = {
        "problem": (
            [record.research_objective]
            if record.research_objective
            else []
        ),
        "population_or_setting": record.population_or_setting,
        "method": record.method_or_intervention,
        "method_family": record.method_or_intervention,
        "constraint": record.constraints,
        "limitation": record.limitations,
        "future_work": record.future_work,
        "finding": record.main_findings,
        "outcome": record.main_findings,
        "comparison": record.comparison_or_baseline,
        "baseline": record.comparison_or_baseline,
        "dataset": record.datasets,
        # Derived dataset characteristics retain dataset-role provenance;
        # population/setting claims must not be relabeled as dataset types.
        "dataset_type": record.datasets,
    }

    return list(
        mapping.get(
            dimension,
            [],
        )
    )


def _item_matches_basis(
    dimension: str,
    basis_value: str,
    item: EvidenceItem,
    *,
    context: _GenerationContext | None = None,
) -> bool:
    if context is not None:
        return context.item_matches_basis(dimension, basis_value, item)

    if dimension == "problem":
        return (
            normalize_problem(basis_value)
            == normalize_problem(item.value)
        )

    if dimension in {
        "method",
        "method_family",
        "baseline",
        "comparison",
    }:
        target = normalize_method_family(
            basis_value
        )
        observed = normalize_method_family(
            item.value
        )

        return bool(
            target
            and observed
            and target == observed
        )

    if dimension == "constraint":
        target = normalize_constraint(
            basis_value
        )
        observed = normalize_constraint(
            item.value
        )

        return bool(
            target
            and observed
            and target == observed
        )

    if dimension == "dataset":
        target = normalize_dataset(
            basis_value
        )
        observed = normalize_dataset(
            item.value
        )

        return bool(
            target
            and observed
            and target == observed
        )

    target_terms = _tokens(
        basis_value
    )
    observed_terms = _tokens(
        f"{item.value} {item.evidence_text}"
    )

    if not target_terms:
        return False

    return target_terms <= observed_terms or bool(
        target_terms & observed_terms
    )


def _observed_pairs(
    landscape: LiteratureLandscape,
    left_dimension: str,
    right_dimension: str,
) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()

    for combination in landscape.combinations:
        left = combination.dimensions.get(
            left_dimension
        )
        right = combination.dimensions.get(
            right_dimension
        )

        if not left or not right:
            continue

        result.add(
            (
                _concept_key(left),
                _concept_key(right),
            )
        )

    return result


def _paper_dimension_values(
    paper,
    dimension: str,
) -> set[str]:
    field_name = _COMBINATION_FIELDS.get(dimension)
    if field_name is None:
        return set()

    return {
        _concept_key(value)
        for value in getattr(paper, field_name, [])
        if is_concrete_entity(value)
    }


def _combination_observed_in_papers(
    landscape: LiteratureLandscape,
    basis: Sequence[LandscapeBasis],
) -> bool:
    """Return whether every basis value occurs in the same paper.

    `LandscapeBasis.count` deliberately is not used here: independent global
    frequencies cannot establish a joint observation.
    """

    if not basis:
        return False

    return any(
        all(
            _concept_key(item.value)
            in _paper_dimension_values(paper, item.dimension)
            for item in basis
        )
        for paper in landscape.papers
    )


def _combination_text_for_basis(
    basis: Sequence[LandscapeBasis],
) -> tuple[str, str]:
    if len(basis) == 2:
        left, right = basis
        return _combination_text(
            left.dimension,
            left.value,
            right.dimension,
            right.value,
        )

    values = [item.value for item in basis]
    joined = " + ".join(values)
    return (
        f"Combination of {joined}",
        (
            "The analyzed landscape contains each component, but this "
            "complete structured combination was not observed within one "
            "paper."
        ),
    )


def _combination_text(
    left_dimension: str,
    left_value: str,
    right_dimension: str,
    right_value: str,
) -> tuple[str, str]:
    pair = (
        left_dimension,
        right_dimension,
    )

    if pair == (
        "problem",
        "method_family",
    ):
        return (
            f"{right_value} for {left_value}",
            (
                f"The analyzed evidence contains {left_value} and "
                f"{right_value}, but their joint occurrence was not "
                "observed in the structured literature landscape."
            ),
        )

    if pair == (
        "method_family",
        "population_or_setting",
    ):
        return (
            f"{left_value} in {right_value}",
            (
                f"The analyzed evidence contains {left_value} and "
                f"{right_value} separately, but not their joint "
                "occurrence."
            ),
        )

    if pair in {
        (
            "method_family",
            "dataset",
        ),
        (
            "method_family",
            "dataset_type",
        ),
    }:
        return (
            f"{left_value} evaluated with {right_value}",
            (
                f"The analyzed landscape contains {left_value} and "
                f"{right_value} data separately, but their joint "
                "evaluation was not observed."
            ),
        )

    if pair == (
        "method_family",
        "constraint",
    ):
        return (
            f"{left_value} under {right_value}",
            (
                f"The analyzed landscape contains {left_value} and "
                f"{right_value} separately, but not their joint "
                "experimental occurrence."
            ),
        )

    return (
        f"Combination of {left_value} and {right_value}",
        (
            f"The analyzed landscape contains {left_value} and "
            f"{right_value} separately, but not their joint occurrence."
        ),
    )


# ---------------------------------------------------------------------------
# Comparison observation
# ---------------------------------------------------------------------------


def _observed_comparison_pairs(
    records: dict[str, PaperEvidence],
    *,
    context: _GenerationContext | None = None,
) -> set[frozenset[str]]:
    observed: set[frozenset[str]] = set()

    for record in records.values():
        if context is not None:
            prepared = context.records_by_id[record.paper_id]
            methods = prepared.method_families
            baselines = prepared.baseline_families
        else:
            methods = {
                normalize_method_family(item.value)
                for item in record.method_or_intervention
                if is_concrete_entity(item.value)
            }
            methods.discard("")

            baselines = {
                normalize_method_family(item.value)
                for item in record.comparison_or_baseline
                if is_concrete_entity(item.value)
            }
            baselines.discard("")

        for method in methods:
            for baseline in baselines:
                if method == baseline:
                    continue

                observed.add(
                    frozenset(
                        (
                            _concept_key(method),
                            _concept_key(baseline),
                        )
                    )
                )

        # A neutral head-to-head paper may represent all compared methods as
        # primary methods. Only infer that relationship when the source text
        # explicitly contains comparison language.
        if (
            len(methods) >= 2
            and _record_indicates_comparison(record)
        ):
            for left, right in combinations(
                sorted(methods),
                2,
            ):
                observed.add(
                    frozenset(
                        (
                            _concept_key(left),
                            _concept_key(right),
                        )
                    )
                )

    return observed


def _comparison_pair_is_justified(
    idea: ResearchIdea,
    left: str,
    right: str,
    records: dict[str, PaperEvidence],
    *,
    context: _GenerationContext | None = None,
) -> bool:
    """Require a positive structural reason to consider a missing comparison."""

    left_key = context._normalize_basis(left, "method_family") if context else normalize_method_family(left)
    right_key = context._normalize_basis(right, "method_family") if context else normalize_method_family(right)

    # An explicit comparison facet is the strongest generic signal. It must
    # mention both alternatives; independent mention in the idea is not enough.
    for requested in idea.comparison:
        text = f"{requested}"
        if (
            _method_text_contains_family(text, left_key, context=context)
            and _method_text_contains_family(text, right_key, context=context)
        ):
            return True

    # A structured method/baseline relationship in the same paper is positive
    # comparison evidence even when the pair is represented in separate fields.
    for record in records.values():
        if context:
            prepared = context.records_by_id[record.paper_id]
            methods = prepared.method_families
            baselines = prepared.baseline_families
        else:
            methods = [
                normalize_method_family(item.value)
                for item in record.method_or_intervention
                if is_concrete_entity(item.value)
            ]
            baselines = [
                normalize_method_family(item.value)
                for item in record.comparison_or_baseline
                if is_concrete_entity(item.value)
            ]
        if (
            (left_key in methods and right_key in baselines)
            or (right_key in methods and left_key in baselines)
        ):
            return True

        # Some extractors retain a single comparison claim containing both
        # alternatives. Require comparison language in that claim instead of
        # interpreting co-occurrence as a relationship.
        for item in record.comparison_or_baseline:
            text = f"{item.value} {item.evidence_text}"
            if (
                _COMPARISON_LANGUAGE.search(text)
                and _method_text_contains_family(text, left_key, context=context)
                and _method_text_contains_family(text, right_key, context=context)
            ):
                return True

    return False


def _method_text_contains_family(
    text: str,
    family: str,
    *,
    context: _GenerationContext | None = None,
) -> bool:
    """Match an already normalized method family by conservative phrase terms."""

    if context is not None:
        return bool(family and context.tokens(text) >= context.tokens(family))
    return bool(family and _tokens(text) >= _tokens(family))


def _record_indicates_comparison(
    record: PaperEvidence,
) -> bool:
    source = [
        record.title,
    ]

    if record.research_objective:
        source.extend(
            (
                record.research_objective.value,
                record.research_objective.evidence_text,
            )
        )

    source.extend(
        item.evidence_text
        for item in record.main_findings
    )

    return bool(
        _COMPARISON_LANGUAGE.search(
            " ".join(source)
        )
    )


def _support_contains_family(
    support: Sequence[GapEvidence],
    family: str,
    *,
    context: _GenerationContext | None = None,
) -> bool:
    target = context._normalize_basis(family, "method_family") if context else normalize_method_family(family)

    return any(
        (context._normalize_basis(item.value, "method_family") if context else normalize_method_family(item.value)) == target
        for item in support
        if (context.is_concrete(item.value) if context else is_concrete_entity(item.value))
    )


# ---------------------------------------------------------------------------
# Candidate relevance to the original idea
# ---------------------------------------------------------------------------


def _candidate_is_relevant(
    idea: ResearchIdea,
    basis: Sequence[LandscapeBasis],
    records: dict[str, PaperEvidence],
    *,
    context: _GenerationContext | None = None,
) -> bool:
    """Require a candidate to remain anchored to the user's research idea."""

    if context is not None:
        return context.candidate_relevant(basis)

    if not basis:
        return True

    anchored = [
        item
        for item in basis
        if _basis_matches_idea(
            item,
            idea,
        )
    ]

    if not anchored:
        # Multi-facet hypotheses with no direct idea anchor are too easy to
        # generate accidentally from incidental landscape observations.
        if len(basis) > 1:
            return False

        return _basis_has_relevant_context(
            basis,
            idea,
            records,
        )

    for item in basis:
        if item in anchored:
            continue

        if not _basis_has_strong_context(
            item,
            idea,
            records,
        ):
            return False

    return True


def _basis_matches_idea(
    item: LandscapeBasis,
    idea: ResearchIdea,
    *,
    context: _GenerationContext | None = None,
) -> bool:
    if context is not None:
        return context.basis_matches(item)

    value = item.value

    if item.dimension in {
        "method",
        "method_family",
        "baseline",
    }:
        target = normalize_method_family(
            value
        )

        candidates = [
            *idea.intervention_or_method,
            *idea.comparison,
        ]

        return bool(
            target
            and any(
                normalize_method_family(candidate)
                == target
                for candidate in candidates
            )
        )

    if item.dimension == "problem":
        target = normalize_problem(
            value
        )

        return bool(
            target
            and any(
                normalize_problem(candidate)
                == target
                for candidate in idea.problem
            )
        )

    if item.dimension == "constraint":
        target = normalize_constraint(
            value
        )

        return bool(
            target
            and any(
                normalize_constraint(candidate)
                == target
                for candidate in idea.constraints
            )
        )

    if item.dimension == "dataset":
        target = normalize_dataset(
            value
        )

        idea_values = [
            *idea.keywords,
            *idea.domain,
            *idea.population,
        ]

        return bool(
            target
            and any(
                normalize_dataset(candidate)
                == target
                for candidate in idea_values
            )
        )

    facet_terms = _tokens(
        value
    )

    return bool(
        facet_terms
        and facet_terms
        <= _idea_terms(idea)
    )


def _basis_has_relevant_context(
    basis: Sequence[LandscapeBasis],
    idea: ResearchIdea,
    records: dict[str, PaperEvidence],
) -> bool:
    return any(
        _basis_has_strong_context(
            item,
            idea,
            records,
        )
        for item in basis
    )


def _basis_has_strong_context(
    item: LandscapeBasis,
    idea: ResearchIdea,
    records: dict[str, PaperEvidence],
) -> bool:
    relevant_ids = [
        paper_id
        for paper_id in item.paper_ids
        if paper_id in records
        and _record_has_idea_anchor(
            records[paper_id],
            idea,
        )
    ]

    if len(relevant_ids) >= MIN_SUPPORT_PAPERS:
        return True

    for paper_id in relevant_ids:
        record = records[
            paper_id
        ]

        field_items = _field_items(
            record,
            item.dimension,
        )

        if any(
            _item_matches_basis(
                item.dimension,
                item.value,
                claim,
            )
            for claim in field_items
        ):
            if _record_has_idea_anchor(
                record,
                idea,
                require_method_and_problem=True,
            ):
                return True

        # Explicit unresolved author statements may justify a one-paper
        # connection even when the landscape frequency itself is sparse.
        basis_terms = _tokens(
            item.value
        )

        for claim in [
            *record.limitations,
            *record.future_work,
        ]:
            claim_terms = _tokens(
                f"{claim.value} {claim.evidence_text}"
            )

            if basis_terms and basis_terms & claim_terms:
                return True

    return False


def _record_has_idea_anchor(
    record: PaperEvidence,
    idea: ResearchIdea,
    *,
    require_method_and_problem: bool = False,
) -> bool:
    problems = _idea_problem_values(
        idea
    )
    methods = [
        value
        for value in idea.intervention_or_method
        if is_concrete_entity(value)
    ]

    objective = record.research_objective

    problem_match = bool(
        problems
        and objective
        and any(
            normalize_problem(problem)
            == normalize_problem(objective.value)
            for problem in problems
        )
    )

    idea_method_families = {
        normalize_method_family(value)
        for value in methods
        if normalize_method_family(value)
    }

    record_method_families = {
        normalize_method_family(item.value)
        for item in record.method_or_intervention
        if normalize_method_family(item.value)
    }

    method_match = bool(
        idea_method_families
        & record_method_families
    )

    if require_method_and_problem:
        if problems and methods:
            return (
                problem_match
                and method_match
            )

        return False

    context_match = _record_context_overlap(
        record,
        idea,
    )

    available_signals = []

    if problems:
        available_signals.append(
            problem_match
        )

    if methods:
        available_signals.append(
            method_match
        )

    if (
        idea.population
        or idea.domain
    ):
        available_signals.append(
            context_match
        )

    if not available_signals:
        return bool(
            _tokens(_record_text(record))
            & _idea_terms(idea)
        )

    return any(
        available_signals
    )


def _idea_problem_values(
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


def _idea_terms(
    idea: ResearchIdea,
) -> set[str]:
    return _tokens(
        " ".join(
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
    )


def _record_text(
    record: PaperEvidence,
) -> str:
    values = [
        record.title,
    ]

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

    for field_name in fields:
        raw = getattr(
            record,
            field_name,
        )

        items = (
            raw
            if isinstance(raw, list)
            else [raw]
            if raw
            else []
        )

        values.extend(
            f"{item.value} {item.evidence_text}"
            for item in items
        )

    return " ".join(
        values
    )


def _record_context_overlap(
    record: PaperEvidence,
    idea: ResearchIdea,
) -> bool:
    context_terms = _tokens(
        " ".join(
            [
                *idea.problem,
                *idea.domain,
                *idea.population,
            ]
        )
    )

    if not context_terms:
        return True

    objective = (
        f"{record.research_objective.value} "
        f"{record.research_objective.evidence_text}"
        if record.research_objective
        else ""
    )

    record_text = " ".join(
        [
            objective,
            *(
                f"{item.value} {item.evidence_text}"
                for item in record.population_or_setting
            ),
        ]
    )

    return bool(
        context_terms
        & _tokens(record_text)
    )


def _has_idea_context(
    paper_ids: Iterable[str],
    idea: ResearchIdea,
    records: dict[str, PaperEvidence],
    *,
    context: _GenerationContext | None = None,
) -> bool:
    if context is not None:
        return context.has_idea_context(paper_ids)

    context_terms = _tokens(
        " ".join(
            [
                *idea.problem,
                *idea.domain,
                *idea.population,
            ]
        )
    )

    if not context_terms:
        return True

    return any(
        paper_id in records
        and _record_context_overlap(
            records[paper_id],
            idea,
        )
        for paper_id in paper_ids
    )


# ---------------------------------------------------------------------------
# Candidate consolidation
# ---------------------------------------------------------------------------


def _unique_evidence(
    items: Iterable[GapEvidence],
    *,
    context: _GenerationContext | None = None,
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
            context.canonical(item.value) if context else canonical_evidence_key(item.value),
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


def consolidate_candidates(
    candidates: Sequence[GapCandidate],
) -> list[GapCandidate]:
    """Merge deterministic near-duplicates before verification."""

    merged: list[GapCandidate] = []

    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.pattern_type,
            item.title.casefold(),
            item.id,
        ),
    ):
        duplicate = next(
            (
                item
                for item in merged
                if _same_candidate(
                    item,
                    candidate,
                )
            ),
            None,
        )

        if duplicate is None:
            merged.append(
                candidate
            )
            continue

        preferred = _stronger(
            candidate,
            duplicate,
        )

        other = (
            duplicate
            if preferred is candidate
            else candidate
        )

        preferred.supporting_paper_ids = list(
            dict.fromkeys(
                (
                    *preferred.supporting_paper_ids,
                    *other.supporting_paper_ids,
                )
            )
        )

        preferred.supporting_evidence = _unique_evidence(
            (
                *preferred.supporting_evidence,
                *other.supporting_evidence,
            )
        )

        preferred.landscape_basis = _unique_basis(
            (
                *preferred.landscape_basis,
                *other.landscape_basis,
            )
        )

        merged[
            merged.index(
                duplicate
            )
        ] = preferred

    return sorted(
        merged,
        key=lambda item: (
            -len(item.supporting_paper_ids),
            item.pattern_type,
            item.title.casefold(),
        ),
    )


def _candidate_components(
    candidate: GapCandidate,
) -> frozenset[tuple[str, str]]:
    return frozenset(
        (
            item.dimension,
            _concept_key(item.value),
        )
        for item in candidate.landscape_basis
        if _concept_key(item.value)
    )


def candidate_priority(
    candidate: GapCandidate,
    idea: ResearchIdea,
) -> tuple[object, ...]:
    """Return a deterministic, vocabulary-independent verification priority."""

    basis = candidate.landscape_basis
    idea_matches = sum(_basis_matches_idea(item, idea) for item in basis)
    idea_anchor_score = candidate.idea_anchor_score
    if idea_anchor_score <= 0.0 and basis:
        idea_anchor_score = idea_matches / len(basis)
    idea_specificity = sum(
        (1.0 - item.prevalence)
        for item in basis
        if _basis_matches_idea(item, idea)
    )
    specificity = sum(
        1.0 - item.prevalence
        for item in basis
    ) / max(1, len(basis))
    generic_components = sum(
        not is_concrete_entity(item.value)
        for item in basis
    )

    return (
        -idea_anchor_score,
        -idea_specificity,
        generic_components,
        -specificity,
        -len(basis),
        -idea_matches,
        -len(candidate.supporting_paper_ids),
        candidate.pattern_type,
        candidate.id,
    )


def candidate_anchor_score(
    candidate: GapCandidate,
    idea: ResearchIdea,
) -> float:
    """Return the deterministic structured anchor score for a candidate."""

    if candidate.idea_anchor_score > 0.0:
        return candidate.idea_anchor_score

    basis = candidate.landscape_basis
    if not basis:
        return 0.0

    return sum(
        _basis_matches_idea(item, idea)
        for item in basis
    ) / len(basis)


def prune_redundant_candidates(
    candidates: Sequence[GapCandidate],
    idea: ResearchIdea,
    *,
    overlap_threshold: float = 0.75,
) -> tuple[list[GapCandidate], int]:
    """Remove nested/high-overlap hypotheses before expensive verification."""

    ordered = sorted(
        candidates,
        key=lambda item: candidate_priority(item, idea),
    )
    kept: list[GapCandidate] = []

    for candidate in ordered:
        components = _candidate_components(candidate)
        if not components:
            kept.append(candidate)
            continue

        anchor_score = candidate.idea_anchor_score
        if anchor_score <= 0.0:
            anchor_score = sum(
                _basis_matches_idea(item, idea)
                for item in candidate.landscape_basis
            ) / max(1, len(candidate.landscape_basis))

        # A multi-component combination that has no structured relationship
        # to the original idea is an incidental landscape observation, not a
        # candidate worth paying for at verification time.
        if len(components) > 1 and anchor_score <= 0.0:
            continue

        candidate_matches = sum(
            _basis_matches_idea(item, idea)
            for item in candidate.landscape_basis
        )
        dominated = False

        for stronger in kept:
            if (
                stronger.category != candidate.category
                or stronger.pattern_type != candidate.pattern_type
            ):
                continue

            stronger_components = _candidate_components(stronger)
            if not stronger_components:
                continue

            overlap = len(components & stronger_components) / max(
                1,
                len(components | stronger_components),
            )
            stronger_matches = sum(
                _basis_matches_idea(item, idea)
                for item in stronger.landscape_basis
            )

            nested = (
                components < stronger_components
                and stronger_matches >= candidate_matches
            )
            if nested or overlap >= overlap_threshold:
                dominated = True
                break

        if not dominated:
            kept.append(candidate)

    return kept, len(candidates) - len(kept)


def _same_candidate(
    left: GapCandidate,
    right: GapCandidate,
) -> bool:
    if (
        left.category != right.category
        or left.pattern_type != right.pattern_type
    ):
        return False

    left_basis = {
        (item.dimension, _concept_key(item.value))
        for item in left.landscape_basis
    }
    right_basis = {
        (item.dimension, _concept_key(item.value))
        for item in right.landscape_basis
    }

    # Distinct structured combinations must remain distinct even when they
    # draw support from the same papers. Shared provenance alone is not
    # semantic duplication.
    if left_basis or right_basis:
        return left_basis == right_basis

    left_terms = _tokens(
        f"{left.title} {left.description}"
    )
    right_terms = _tokens(
        f"{right.title} {right.description}"
    )

    if not left_terms or not right_terms:
        return False

    overlap = (
        len(
            left_terms
            & right_terms
        )
        / max(
            1,
            min(
                len(left_terms),
                len(right_terms),
            ),
        )
    )

    shared_support = len(
        set(left.supporting_paper_ids)
        & set(right.supporting_paper_ids)
    )

    return (
        overlap >= 0.60
        or shared_support >= 2
    )


def _stronger(
    left: GapCandidate,
    right: GapCandidate,
) -> GapCandidate:
    left_key = (
        len(
            left.supporting_paper_ids
        ),
        sum(
            item.role
            == "direct_support"
            for item in left.supporting_evidence
        ),
        len(
            _tokens(left.title)
        ),
    )

    right_key = (
        len(
            right.supporting_paper_ids
        ),
        sum(
            item.role
            == "direct_support"
            for item in right.supporting_evidence
        ),
        len(
            _tokens(right.title)
        ),
    )

    return (
        left
        if left_key > right_key
        else right
    )


def _unique_basis(
    items: Iterable[LandscapeBasis],
) -> list[LandscapeBasis]:
    result: list[LandscapeBasis] = []
    seen: set[
        tuple[
            str,
            str,
        ]
    ] = set()

    for item in items:
        key = (
            item.dimension,
            _concept_key(
                item.value
            ),
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


__all__ = [
    "GapCandidateGenerator",
    "candidate_priority",
    "consolidate_candidates",
    "is_concrete_entity",
    "prune_redundant_candidates",
    "validate_evidence_semantics",
]
