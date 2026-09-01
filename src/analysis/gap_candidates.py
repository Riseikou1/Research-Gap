"""Deterministic, landscape-grounded Milestone-6 candidate generation.

Candidate gaps are derived only from structured evidence and observations
already present in LiteratureLandscape. No domain-specific scientific entities
or named methods/datasets are hard-coded here.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from itertools import combinations

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
    "a",
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

# Only combinations for which Milestone 5 actually records joint observations
# may produce an "absent combination" hypothesis.
_COMBINATION_PAIRS = (
    ("problem", "method_family"),
    ("method_family", "population_or_setting"),
    ("method_family", "dataset"),
    ("method_family", "dataset_type"),
    ("method_family", "constraint"),
)

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
    """Create a conservative comparison key.

    Do not rewrite domain words such as "field" into "realworld". Scientific
    meaning is domain-dependent and this layer should not guess it.
    """

    return " ".join(
        token
        for token in canonical_evidence_key(text).split()
        if token not in _STOPWORDS
    )


def is_concrete_entity(value: str) -> bool:
    """Reject placeholders while preserving arbitrary scientific entities."""

    key = _concept_key(value)

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

        if normalize_constraint(evidence_text) != "limited labeled data":
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

    def generate(
        self,
        idea: ResearchIdea,
        landscape: LiteratureLandscape,
        evidence: Sequence[PaperEvidence],
    ) -> list[GapCandidate]:
        self.notices = []

        records = {
            item.paper_id: item
            for item in evidence
        }

        if not records or landscape.total_papers == 0:
            self.notices.append(
                "insufficient_evidence: no landscape-backed evidence is available"
            )
            return []

        candidates: list[GapCandidate] = []

        candidates.extend(
            self._conflict_candidates(
                landscape,
                records,
                idea,
            )
        )
        candidates.extend(
            self._repeated_candidates(
                landscape,
                records,
                idea,
                "limitation",
            )
        )
        candidates.extend(
            self._repeated_candidates(
                landscape,
                records,
                idea,
                "future_work",
            )
        )
        candidates.extend(
            self._combination_candidates(
                idea,
                landscape,
                records,
            )
        )
        candidates.extend(
            self._narrow_setting_candidates(
                idea,
                landscape,
                records,
            )
        )
        candidates.extend(
            self._comparison_candidates(
                idea,
                landscape,
                records,
            )
        )

        candidates = consolidate_candidates(candidates)

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
    ) -> list[GapCandidate]:
        result: list[GapCandidate] = []

        for conflict in landscape.conflicts:
            if not is_concrete_entity(conflict.topic):
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
    ) -> list[GapCandidate]:
        frequencies = [
            item
            for item in landscape.frequencies
            if item.dimension == dimension
        ]

        grouped: dict[str, list] = defaultdict(list)

        for item in frequencies:
            key = _concept_key(item.value)

            if key:
                grouped[key].append(item)

        result: list[GapCandidate] = []

        for key, items in grouped.items():
            if not is_concrete_entity(key):
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
    ) -> list[GapCandidate]:
        result: list[GapCandidate] = []

        frequencies_by_dimension = {
            dimension: [
                item
                for item in landscape.frequencies
                if item.dimension == dimension
                and item.count >= 1
                and is_concrete_entity(item.value)
            ]
            for dimension_pair in _COMBINATION_PAIRS
            for dimension in dimension_pair
        }

        for left_dimension, right_dimension in _COMBINATION_PAIRS:
            left_items = frequencies_by_dimension.get(
                left_dimension,
                [],
            )
            right_items = frequencies_by_dimension.get(
                right_dimension,
                [],
            )

            if not left_items or not right_items:
                continue

            observed = _observed_pairs(
                landscape,
                left_dimension,
                right_dimension,
            )

            for left in left_items:
                for right in right_items:
                    pair_key = (
                        _concept_key(left.value),
                        _concept_key(right.value),
                    )

                    if pair_key in observed:
                        continue

                    basis = [
                        self._basis(left, landscape.total_papers),
                        self._basis(right, landscape.total_papers),
                    ]

                    if not _candidate_is_relevant(
                        idea,
                        basis,
                        records,
                    ):
                        continue

                    support = (
                        self._collect_basis_evidence(
                            left,
                            records,
                            role="contextual_support",
                        )
                        + self._collect_basis_evidence(
                            right,
                            records,
                            role="contextual_support",
                        )
                    )

                    title, description = _combination_text(
                        left_dimension,
                        left.value,
                        right_dimension,
                        right.value,
                    )

                    candidate = self._build(
                        title=title,
                        description=description,
                        category="combination",
                        rationale=(
                            "Both components occur in the analyzed "
                            "landscape, but their joint occurrence was not "
                            "observed in the structured combinations. "
                            "Targeted verification is required."
                        ),
                        pattern_type="combination_gap",
                        support=support,
                        basis=basis,
                        idea=idea,
                        records=records,
                    )

                    if candidate:
                        result.append(candidate)

        return result

    # ------------------------------------------------------------------
    # Narrow dataset / validation setting
    # ------------------------------------------------------------------

    def _narrow_setting_candidates(
        self,
        idea: ResearchIdea,
        landscape: LiteratureLandscape,
        records: dict[str, PaperEvidence],
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
            and is_concrete_entity(item.value)
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
            and is_concrete_entity(item.value)
        ]

        if len(methods) < 2:
            return []

        observed_pairs = _observed_comparison_pairs(records)
        result: list[GapCandidate] = []

        for left, right in combinations(methods, 2):
            left_key = _concept_key(left.value)
            right_key = _concept_key(right.value)

            if not left_key or not right_key or left_key == right_key:
                continue

            pair = frozenset((left_key, right_key))

            if pair in observed_pairs:
                continue

            basis = [
                self._basis(left, landscape.total_papers),
                self._basis(right, landscape.total_papers),
            ]

            if not _candidate_is_relevant(
                idea,
                basis,
                records,
            ):
                continue

            if not (
                _has_idea_context(
                    left.paper_ids,
                    idea,
                    records,
                )
                and _has_idea_context(
                    right.paper_ids,
                    idea,
                    records,
                )
            ):
                continue

            support = (
                self._collect_basis_evidence(
                    left,
                    records,
                    role="contextual_support",
                )
                + self._collect_basis_evidence(
                    right,
                    records,
                    role="contextual_support",
                )
            )

            left_supported = _support_contains_family(
                support,
                left.value,
            )
            right_supported = _support_contains_family(
                support,
                right.value,
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
                    "The candidate is based on repeated independent presence "
                    "of both method families and absence of an observed "
                    "explicit comparison between them."
                ),
                pattern_type="missing_comparison",
                support=support,
                basis=basis,
                idea=idea,
                records=records,
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
    ) -> list[GapEvidence]:
        return cls._collect_evidence(
            item.paper_ids,
            item.dimension,
            item.value,
            records,
            role=role,
            claim_text=item.value,
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
    ) -> list[GapEvidence]:
        result: list[GapEvidence] = []

        for paper_id in dict.fromkeys(paper_ids):
            record = records.get(paper_id)

            if record is None:
                continue

            items = _field_items(
                record,
                evidence_type,
            )

            items = [
                item
                for item in items
                if is_concrete_entity(item.value)
            ]

            matched = [
                item
                for item in items
                if _item_matches_basis(
                    evidence_type,
                    value,
                    item,
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
                if not validate_evidence_semantics(
                    claim_text=claim_text,
                    evidence_type=evidence_type,
                    evidence=item,
                ):
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

        return _unique_evidence(result)

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
    ) -> GapCandidate | None:
        support = _unique_evidence(support)

        support_ids = list(
            dict.fromkeys(
                item.paper_id
                for item in support
            )
        )

        if len(support_ids) < MIN_SUPPORT_PAPERS:
            return None

        if not _candidate_is_relevant(
            idea,
            basis,
            records,
        ):
            return None

        candidate_terms = _tokens(
            f"{title} {description}"
        )
        idea_terms = _idea_terms(idea)

        relevance = (
            len(candidate_terms & idea_terms)
            / max(1, len(candidate_terms))
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
        "dataset_type": [
            *record.datasets,
            *record.population_or_setting,
        ],
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
) -> bool:
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
) -> set[frozenset[str]]:
    observed: set[frozenset[str]] = set()

    for record in records.values():
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
) -> bool:
    target = normalize_method_family(
        family
    )

    return any(
        normalize_method_family(item.value)
        == target
        for item in support
        if is_concrete_entity(item.value)
    )


# ---------------------------------------------------------------------------
# Candidate relevance to the original idea
# ---------------------------------------------------------------------------


def _candidate_is_relevant(
    idea: ResearchIdea,
    basis: Sequence[LandscapeBasis],
    records: dict[str, PaperEvidence],
) -> bool:
    """Require a candidate to remain anchored to the user's research idea."""

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
) -> bool:
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


def _same_candidate(
    left: GapCandidate,
    right: GapCandidate,
) -> bool:
    if (
        left.category != right.category
        or left.pattern_type != right.pattern_type
    ):
        return False

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
    "consolidate_candidates",
    "is_concrete_entity",
    "validate_evidence_semantics",
]