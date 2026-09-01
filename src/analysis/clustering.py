"""Deterministic frequency, combination, coverage, and conflict analysis."""

from __future__ import annotations

import re
from collections import OrderedDict
from itertools import product
from typing import Iterable, Sequence

from src.extraction.evidence import PaperEvidence
from src.models.landscape import (
    CombinationPattern,
    EvidenceConflict,
    FeatureFrequency,
    LiteratureLandscape,
    PaperFeatures,
)
from src.models.paper import Paper

from .comparison import to_paper_features


_DIMENSIONS = (
    ("problem", "problems"),
    ("population_or_setting", "populations_or_settings"),
    ("method", "methods"),
    ("method_family", "method_families"),
    ("dataset", "datasets"),
    ("dataset_type", "dataset_types"),
    ("baseline", "baselines"),
    ("performance_metric", "performance_metrics"),
    ("efficiency_metric", "efficiency_metrics"),
    ("study_type", "study_type"),
    ("outcome", "outcomes"),
    ("constraint", "constraints"),
    ("limitation", "limitations"),
    ("future_work", "future_work"),
)

_MISSING_FIELDS = (
    ("research_objective", "problems"),
    ("population_or_setting", "populations_or_settings"),
    ("methods", "methods"),
    ("datasets", "datasets"),
    ("sample_size", None),
    ("baselines", "baselines"),
    ("evaluation_metrics", "metrics"),
    ("outcomes", "outcomes"),
    ("constraints", "constraints"),
    ("limitations", "limitations"),
    ("future_work", "future_work"),
)

# These are structural scientific combinations, not domain-specific concepts.
_COMBINATION_SPECS = (
    (
        ("problem", "problems"),
        ("method_family", "method_families"),
    ),
    (
        ("method_family", "method_families"),
        ("population_or_setting", "populations_or_settings"),
    ),
    (
        ("method_family", "method_families"),
        ("dataset", "datasets"),
    ),
    (
        ("method_family", "method_families"),
        ("dataset_type", "dataset_types"),
    ),
    (
        ("method_family", "method_families"),
        ("baseline", "baselines"),
    ),
    (
        ("method_family", "method_families"),
        ("constraint", "constraints"),
    ),
    (
        ("problem", "problems"),
        ("method_family", "method_families"),
        ("population_or_setting", "populations_or_settings"),
    ),
    (
        ("problem", "problems"),
        ("method_family", "method_families"),
        ("dataset", "datasets"),
    ),
    (
        ("problem", "problems"),
        ("method_family", "method_families"),
        ("constraint", "constraints"),
    ),
)

_SENTINELS = frozenset({
    "",
    "other",
    "unknown",
    "unspecified",
    "none",
    "not specified",
    "not reported",
})

# Reviews and surveys may mention many methods/datasets without experimentally
# testing the combinations. Do not turn those mentions into observed empirical
# combinations.
_COMBINATION_STUDY_TYPES = frozenset({
    "empirical",
    "methodological",
})

# Conflict detection must use directionally meaningful comparative language.
# Generic words such as "higher" and "lower" are intentionally excluded:
# higher mortality can be worse while lower error can be better.
_POSITIVE_FINDING = re.compile(
    r"\b(?:"
    r"outperform(?:s|ed|ing)?|"
    r"perform(?:s|ed)?\s+better\s+than|"
    r"better\s+than|"
    r"superior\s+to|"
    r"significantly\s+better\s+than|"
    r"significantly\s+outperform(?:s|ed)?|"
    r"improv(?:e|es|ed|ing)\s+(?:over|upon|compared\s+with|compared\s+to)|"
    r"significant(?:ly)?\s+improv(?:e|es|ed|ement)"
    r")\b",
    re.I,
)

_NEGATIVE_FINDING = re.compile(
    r"\b(?:"
    r"underperform(?:s|ed|ing)?|"
    r"perform(?:s|ed)?\s+worse\s+than|"
    r"worse\s+than|"
    r"inferior\s+to|"
    r"does\s+not\s+outperform|"
    r"did\s+not\s+outperform|"
    r"fails?\s+to\s+outperform|"
    r"failed\s+to\s+outperform|"
    r"does\s+not\s+improve|"
    r"did\s+not\s+improve|"
    r"fails?\s+to\s+improve|"
    r"failed\s+to\s+improve|"
    r"no\s+significant\s+improvement|"
    r"no\s+improvement"
    r")\b",
    re.I,
)


class LandscapeAnalyzer:
    """Build a deterministic literature landscape from structured evidence."""

    def analyze(
        self,
        evidence: Sequence[PaperEvidence],
        papers: Sequence[Paper] | None = None,
    ) -> LiteratureLandscape:
        features = to_paper_features(evidence, papers)
        total = len(features)

        return LiteratureLandscape(
            total_papers=total,
            papers=features,
            frequencies=_frequencies(features, total),
            combinations=_combinations(features, total),
            missing_field_counts=_missing_counts(evidence, features),
            conflicts=_conflicts(features),
        )


def _is_usable_value(value: str) -> bool:
    return bool(value and value.casefold().strip() not in _SENTINELS)


def _field_values(
    feature: PaperFeatures,
    field_name: str,
) -> list[str]:
    raw = getattr(feature, field_name)

    if isinstance(raw, str):
        values = [raw]
    else:
        values = list(raw)

    return [
        value
        for value in dict.fromkeys(values)
        if _is_usable_value(value)
    ]


def _frequencies(
    features: Sequence[PaperFeatures],
    total: int,
) -> list[FeatureFrequency]:
    grouped: OrderedDict[tuple[str, str], list[str]] = OrderedDict()

    for feature in features:
        for dimension, field_name in _DIMENSIONS:
            for value in _field_values(feature, field_name):
                key = (dimension, value)
                grouped.setdefault(key, [])

                if feature.paper_id not in grouped[key]:
                    grouped[key].append(feature.paper_id)

    result = [
        FeatureFrequency(
            dimension=dimension,
            value=value,
            count=len(paper_ids),
            prevalence=len(paper_ids) / total if total else 0.0,
            paper_ids=paper_ids,
        )
        for (dimension, value), paper_ids in grouped.items()
    ]

    return sorted(
        result,
        key=lambda item: (
            -item.count,
            item.dimension,
            item.value,
        ),
    )


def _combinations(
    features: Sequence[PaperFeatures],
    total: int,
) -> list[CombinationPattern]:
    grouped: OrderedDict[
        tuple[tuple[str, str], ...],
        list[str],
    ] = OrderedDict()

    for feature in features:
        if feature.study_type not in _COMBINATION_STUDY_TYPES:
            continue

        for spec in _COMBINATION_SPECS:
            value_sets = [
                _field_values(feature, field_name)
                for _, field_name in spec
            ]

            if any(not values for values in value_sets):
                continue

            for selected_values in product(*value_sets):
                key = tuple(
                    (dimension, value)
                    for (dimension, _), value in zip(
                        spec,
                        selected_values,
                    )
                )

                grouped.setdefault(key, [])

                if feature.paper_id not in grouped[key]:
                    grouped[key].append(feature.paper_id)

    result = [
        CombinationPattern(
            dimensions=dict(key),
            count=len(paper_ids),
            prevalence=len(paper_ids) / total if total else 0.0,
            paper_ids=paper_ids,
        )
        for key, paper_ids in grouped.items()
    ]

    return sorted(
        result,
        key=lambda item: (
            -item.count,
            tuple(item.dimensions.items()),
        ),
    )


def _missing_counts(
    evidence: Sequence[PaperEvidence],
    features: Sequence[PaperFeatures],
) -> dict[str, int]:
    counts: OrderedDict[str, int] = OrderedDict()

    for field_name, feature_field in _MISSING_FIELDS:
        if feature_field is None:
            counts[field_name] = sum(
                record.sample_size is None
                for record in evidence
            )
            continue

        counts[field_name] = sum(
            not getattr(feature, feature_field)
            for feature in features
        )

    return dict(counts)


def _overlap(
    left: Iterable[str],
    right: Iterable[str],
) -> bool:
    left_values = {
        value
        for value in left
        if _is_usable_value(value)
    }
    right_values = {
        value
        for value in right
        if _is_usable_value(value)
    }

    return bool(left_values & right_values)


def _optional_context_matches(
    left: Sequence[str],
    right: Sequence[str],
) -> bool:
    """Require overlap when both studies report the contextual dimension.

    Missing context does not establish comparability, so asymmetric coverage
    remains conservatively incomparable.
    """

    left_values = [
        value
        for value in left
        if _is_usable_value(value)
    ]
    right_values = [
        value
        for value in right
        if _is_usable_value(value)
    ]

    if bool(left_values) != bool(right_values):
        return False

    if not left_values:
        return True

    return _overlap(left_values, right_values)


def _contexts_comparable(
    left: PaperFeatures,
    right: PaperFeatures,
) -> bool:
    """Require explicitly comparable experimental contexts."""

    if left.study_type not in _COMBINATION_STUDY_TYPES:
        return False

    if right.study_type not in _COMBINATION_STUDY_TYPES:
        return False

    if not _overlap(left.problems, right.problems):
        return False

    if not _overlap(left.method_families, right.method_families):
        return False

    if not _overlap(left.performance_metrics, right.performance_metrics):
        return False

    for left_values, right_values in (
        (left.datasets, right.datasets),
        (left.populations_or_settings, right.populations_or_settings),
        (left.baselines, right.baselines),
        (left.constraints, right.constraints),
    ):
        if not _optional_context_matches(left_values, right_values):
            return False

    return True


def _finding_polarity(
    finding: str,
) -> int | None:
    """Return only explicit comparative polarity.

    +1 = explicit favorable comparison
    -1 = explicit unfavorable/non-improving comparison
    None = direction cannot be safely inferred
    """

    # Check negative first because phrases such as "does not outperform"
    # contain the positive word "outperform".
    if _NEGATIVE_FINDING.search(finding):
        return -1

    if _POSITIVE_FINDING.search(finding):
        return 1

    return None


def _finding_polarities(
    findings: Sequence[str],
) -> set[int]:
    return {
        polarity
        for finding in findings
        if (polarity := _finding_polarity(finding)) is not None
    }


def _conflicts(
    features: Sequence[PaperFeatures],
) -> list[EvidenceConflict]:
    conflicts: list[EvidenceConflict] = []

    for index, left in enumerate(features):
        for right in features[index + 1:]:
            if not _contexts_comparable(left, right):
                continue

            left_polarity = _finding_polarities(left.outcomes)
            right_polarity = _finding_polarities(right.outcomes)

            if not left_polarity or not right_polarity:
                continue

            has_opposition = (
                1 in left_polarity and -1 in right_polarity
            ) or (
                -1 in left_polarity and 1 in right_polarity
            )

            if not has_opposition:
                continue

            topic = _conflict_topic(left, right)

            if not topic:
                continue

            conflicts.append(
                EvidenceConflict(
                    paper_ids=[
                        left.paper_id,
                        right.paper_id,
                    ],
                    topic=topic,
                    status="comparable_conflict",
                    reason=(
                        "Comparable empirical studies contain explicitly "
                        "opposing comparative findings under matching "
                        "problem, method-family, metric, and reported context."
                    ),
                )
            )

    return conflicts


def _conflict_topic(
    left: PaperFeatures,
    right: PaperFeatures,
) -> str:
    shared_problems = [
        value
        for value in left.problems
        if value in right.problems and _is_usable_value(value)
    ]

    if shared_problems:
        return shared_problems[0]

    shared_methods = [
        value
        for value in left.method_families
        if value in right.method_families and _is_usable_value(value)
    ]

    return shared_methods[0] if shared_methods else ""