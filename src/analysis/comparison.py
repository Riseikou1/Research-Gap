"""Generic field normalization from structured paper evidence to landscape features."""

from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

from src.extraction.evidence import EvidenceItem, PaperEvidence, canonical_evidence_key
from src.models.landscape import PaperFeatures
from src.models.paper import Paper


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

_VAGUE_VALUES = {
    "",
    "other",
    "unknown",
    "unspecified",
    "none",
    "not specified",
    "not reported",
}

_GENERIC_METHOD_VALUES = {
    "method",
    "methods",
    "model",
    "models",
    "approach",
    "approaches",
    "technique",
    "techniques",
    "algorithm",
    "algorithms",
    "architecture",
    "architectures",
    "framework",
    "frameworks",
    "proposed method",
    "proposed model",
    "our method",
    "our model",
}

_GENERIC_DATASET_PATTERN = re.compile(
    r"^(?:"
    r"dataset|datasets|data|benchmark|benchmark datasets?|"
    r"public datasets?|private datasets?|custom datasets?|"
    r"proprietary datasets?|several datasets?|multiple datasets?|"
    r"various datasets?|different datasets?|several public datasets?|"
    r"multiple public datasets?|widely used datasets?"
    r")$",
    re.I,
)

_SENTENCE_LIKE_PATTERN = re.compile(
    r"\b(?:we|this paper|this study|our work|the authors|"
    r"propose|proposes|proposed|develop|developed|evaluate|evaluated|"
    r"investigate|investigated|achieve|achieved|outperform|outperformed|"
    r"using|used to)\b",
    re.I,
)


def _clean(value: str) -> str:
    return canonical_evidence_key(value)


def _is_concrete_phrase(value: str, *, max_words: int = 12) -> bool:
    normalized = _clean(value)

    if not normalized or normalized in _VAGUE_VALUES:
        return False

    words = normalized.split()

    if not 1 <= len(words) <= max_words:
        return False

    if _SENTENCE_LIKE_PATTERN.search(normalized):
        return False

    return any(character.isalpha() for character in normalized)


# ---------------------------------------------------------------------------
# Problem / objective normalization
# ---------------------------------------------------------------------------

_REVIEW_PATTERN = re.compile(
    r"\b(?:systematic review|scoping review|literature review|review|"
    r"survey|meta analysis)\b",
    re.I,
)

def normalize_problem(value: str) -> str:
    """Canonicalize representation while preserving the user's terminology."""

    normalized = _clean(value)

    if not normalized:
        return ""

    return normalized if _is_concrete_phrase(normalized) else ""


# ---------------------------------------------------------------------------
# Method normalization
# ---------------------------------------------------------------------------

def normalize_method(value: str) -> str:
    normalized = _clean(value)

    if not normalized or normalized in _GENERIC_METHOD_VALUES:
        return ""

    normalized = re.sub(
        r"\b(?:model|models|architecture|architectures)$",
        "",
        normalized,
    ).strip()

    return normalized if normalized not in _GENERIC_METHOD_VALUES else ""


def normalize_method_family(value: str) -> str:
    """Return the supplied method phrase without assigning a family."""

    return normalize_method(value)


# ---------------------------------------------------------------------------
# Dataset normalization
# ---------------------------------------------------------------------------

_DATASET_SUFFIX_PATTERN = re.compile(
    r"\s+(?:dataset|datasets|corpus|cohort)$",
    re.I,
)


def normalize_dataset(value: str) -> str:
    normalized = _clean(value)

    if not normalized:
        return ""

    if _GENERIC_DATASET_PATTERN.fullmatch(normalized):
        return ""

    normalized = _DATASET_SUFFIX_PATTERN.sub("", normalized).strip()

    if not normalized or _GENERIC_DATASET_PATTERN.fullmatch(normalized):
        return ""

    return normalized if _is_concrete_phrase(normalized, max_words=12) else ""


# ---------------------------------------------------------------------------
# Metric normalization
# ---------------------------------------------------------------------------

def normalize_metric(value: str) -> str:
    normalized = _clean(value)
    return normalized if _is_concrete_phrase(normalized, max_words=6) else ""


def metric_kind(value: str) -> str | None:
    normalized = normalize_metric(value)

    if not normalized:
        return None

    return "performance"


# ---------------------------------------------------------------------------
# Constraint normalization
# ---------------------------------------------------------------------------

def normalize_constraint(value: str) -> str:
    normalized = _clean(value)

    if not normalized or normalized in _VAGUE_VALUES:
        return ""

    return normalized if _is_concrete_phrase(normalized, max_words=12) else ""


# ---------------------------------------------------------------------------
# Dataset characteristics
# ---------------------------------------------------------------------------

def dataset_types(record: PaperEvidence) -> list[str]:
    """Return no inferred categories; dataset characteristics need structured evidence."""

    return []


# ---------------------------------------------------------------------------
# Study-type normalization
# ---------------------------------------------------------------------------

def classify_study_type(record: PaperEvidence) -> str:
    text = " ".join(
        [
            record.title,
            record.research_objective.evidence_text if record.research_objective else "",
            *(item.evidence_text for item in record.main_findings),
        ]
    )

    if re.search(r"\bsurvey\b", text, re.I):
        return "survey"

    if _REVIEW_PATTERN.search(text):
        return "review"

    return (
        record.study_type
        if record.study_type in {
            "empirical",
            "review",
            "survey",
            "methodological",
        }
        else "other"
    )


# ---------------------------------------------------------------------------
# Evidence → PaperFeatures
# ---------------------------------------------------------------------------

def _values(items: Iterable[EvidenceItem], normalizer: Callable[[str], str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in items:
        value = normalizer(item.value)

        if value and value not in seen:
            seen.add(value)
            result.append(value)

    return result


def _single_value(item: EvidenceItem | None, normalizer: Callable[[str], str]) -> list[str]:
    if item is None:
        return []

    value = normalizer(item.value)
    return [value] if value else []


def to_paper_features(
    evidence: Sequence[PaperEvidence],
    papers: Sequence[Paper] | None = None,
) -> list[PaperFeatures]:
    relevance = {
        paper.id: paper.final_score
        for paper in papers or []
    }

    result: list[PaperFeatures] = []

    for record in evidence:
        methods = _values(
            record.method_or_intervention,
            normalize_method,
        )

        families: list[str] = []

        for method in methods:
            family = normalize_method_family(method)

            if family and family not in families:
                families.append(family)

        performance_metrics = _values(
            (
                item
                for item in record.evaluation_metrics
                if metric_kind(item.value) == "performance"
            ),
            normalize_metric,
        )

        efficiency_metrics = _values(
            (
                item
                for item in record.evaluation_metrics
                if metric_kind(item.value) == "efficiency"
            ),
            normalize_metric,
        )

        result.append(
            PaperFeatures(
                paper_id=record.paper_id,
                title=record.title,
                relevance_score=relevance.get(record.paper_id),
                problems=_single_value(record.research_objective, normalize_problem),
                populations_or_settings=_values(record.population_or_setting, _clean),
                methods=methods,
                method_families=families,
                datasets=_values(record.datasets, normalize_dataset),
                dataset_types=dataset_types(record),
                baselines=_values(record.comparison_or_baseline, normalize_method),
                metrics=list(dict.fromkeys((*performance_metrics, *efficiency_metrics))),
                performance_metrics=performance_metrics,
                efficiency_metrics=efficiency_metrics,
                outcomes=_values(record.main_findings, _clean),
                constraints=_values(record.constraints, normalize_constraint),
                limitations=_values(record.limitations, _clean),
                future_work=_values(record.future_work, _clean),
                study_type=classify_study_type(record),
            )
        )

    return result


def normalize_feature(value: str, dimension: str) -> str:
    normalizers = {
        "problem": normalize_problem,
        "method": normalize_method,
        "method_family": normalize_method_family,
        "dataset": normalize_dataset,
        "metric": normalize_metric,
        "constraint": normalize_constraint,
    }

    return normalizers.get(dimension, _clean)(value)


method_family = normalize_method_family
normalize_method_family_value = normalize_method_family
