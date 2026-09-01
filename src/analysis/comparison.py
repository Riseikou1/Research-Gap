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

_PROBLEM_RULES = (
    (re.compile(r"\banomal(?:y|ies)\s+detection\b", re.I), "anomaly detection"),
    (re.compile(r"\bdomain\s+adaptation\b", re.I), "domain adaptation"),
    (re.compile(r"\bcausal\s+(?:inference|estimation)\b", re.I), "causal inference"),
    (re.compile(r"\bquestion answering\b|\bqa\b", re.I), "question answering"),
    (re.compile(r"\bclassif(?:y|ies|ied|ication|ying)\b", re.I), "classification"),
    (re.compile(r"\bdetect(?:s|ed|ion|ing)?\b|\brecognition\b|\bidentification\b", re.I), "detection"),
    (re.compile(r"\bsegment(?:s|ed|ation|ing)?\b", re.I), "segmentation"),
    (re.compile(r"\bforecast(?:s|ed|ing)?\b", re.I), "forecasting"),
    (re.compile(r"\bpredict(?:s|ed|ion|ing|ive)?\b", re.I), "prediction"),
    (re.compile(r"\bregression\b", re.I), "regression"),
    (re.compile(r"\bcluster(?:s|ed|ing)?\b", re.I), "clustering"),
    (re.compile(r"\brank(?:s|ed|ing)?\b", re.I), "ranking"),
    (re.compile(r"\brecommend(?:s|ed|ation|ing|er)?\b", re.I), "recommendation"),
    (re.compile(r"\bsummari[sz](?:e|es|ed|ation|ing)\b", re.I), "summarization"),
    (re.compile(r"\btranslat(?:e|es|ed|ion|ing)\b", re.I), "translation"),
    (re.compile(r"\bretriev(?:al|e|es|ed|ing)\b", re.I), "retrieval"),
    (re.compile(r"\bgenerat(?:e|es|ed|ion|ing|ive)\b", re.I), "generation"),
    (re.compile(r"\boptim(?:ize|izes|ized|ization|izing)\b", re.I), "optimization"),
    (re.compile(r"\bestimat(?:e|es|ed|ion|ing)\b", re.I), "estimation"),
)


def _specific_problem(normalized: str, pattern: re.Pattern[str], label: str) -> str:
    """Preserve meaningful qualifiers around a generic task family."""

    matched = pattern.search(normalized)

    if not matched:
        return normalized

    if normalized == matched.group(0):
        return label

    replaced = (
        normalized[: matched.start()]
        + label
        + normalized[matched.end() :]
    )

    replaced = " ".join(replaced.split())

    if _is_concrete_phrase(replaced, max_words=12):
        return replaced

    return label


def normalize_problem(value: str) -> str:
    """Normalize task wording without discarding meaningful scientific context."""

    normalized = _clean(value)

    if not normalized:
        return ""

    if _REVIEW_PATTERN.search(normalized):
        return "review"

    for pattern, label in _PROBLEM_RULES:
        if pattern.search(normalized):
            return _specific_problem(normalized, pattern, label)

    return normalized if _is_concrete_phrase(normalized) else ""


# ---------------------------------------------------------------------------
# Method normalization
# ---------------------------------------------------------------------------

_METHOD_FAMILY_RULES = (
    (re.compile(r"\bvision\s+transformer\b", re.I), "vision transformer"),
    (re.compile(r"\btransformer\b", re.I), "transformer"),
    (
        re.compile(r"\bconvolutional\s+neural\s+network\b|\bcnn\b", re.I),
        "convolutional neural network",
    ),
    (
        re.compile(r"\bgraph\s+neural\s+network\b|\bgnn\b", re.I),
        "graph neural network",
    ),
    (
        re.compile(r"\brecurrent\s+neural\s+network\b|\brnn\b", re.I),
        "recurrent neural network",
    ),
    (
        re.compile(r"\bgenerative\s+adversarial\s+network\b|\bgan\b", re.I),
        "generative adversarial network",
    ),
    (re.compile(r"\breinforcement\s+learning\b", re.I), "reinforcement learning"),
    (re.compile(r"\bself\s+supervised\s+learning\b", re.I), "self-supervised learning"),
    (re.compile(r"\bsemi\s+supervised\s+learning\b", re.I), "semi-supervised learning"),
    (re.compile(r"\bunsupervised\s+learning\b", re.I), "unsupervised learning"),
    (re.compile(r"\bsupervised\s+learning\b", re.I), "supervised learning"),
    (re.compile(r"\bfederated\s+learning\b", re.I), "federated learning"),
    (re.compile(r"\btransfer\s+learning\b", re.I), "transfer learning"),
    (re.compile(r"\bmeta\s+learning\b", re.I), "meta-learning"),
    (re.compile(r"\bcontrastive\s+learning\b", re.I), "contrastive learning"),
    (re.compile(r"\bfew\s+shot\s+learning\b", re.I), "few-shot learning"),
    (re.compile(r"\bzero\s+shot\s+learning\b", re.I), "zero-shot learning"),
    (re.compile(r"\bdomain\s+adaptation\b", re.I), "domain adaptation"),
    (re.compile(r"\bfine\s+tuning\b", re.I), "fine-tuning"),
    (
        re.compile(r"\bretrieval\s+augmented\s+generation\b", re.I),
        "retrieval-augmented generation",
    ),
    (
        re.compile(
            r"\brandomized\s+controlled\s+trial\b|"
            r"\brandomised\s+controlled\s+trial\b",
            re.I,
        ),
        "randomized controlled trial",
    ),
    (
        re.compile(r"\bdifference\s+in\s+differences\b", re.I),
        "difference-in-differences",
    ),
)


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
    normalized = normalize_method(value)

    if not normalized:
        return ""

    for pattern, family in _METHOD_FAMILY_RULES:
        if pattern.search(normalized):
            return family

    return normalized if _is_concrete_phrase(normalized, max_words=8) else ""


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

_PERFORMANCE_METRIC_RULES = (
    (re.compile(r"\baccuracy\b", re.I), "accuracy"),
    (re.compile(r"\bprecision\b", re.I), "precision"),
    (re.compile(r"\brecall\b", re.I), "recall"),
    (re.compile(r"\bf1(?:\s+score)?\b", re.I), "f1"),
    (re.compile(r"\bmean\s+average\s+precision\b|\bmap\b", re.I), "map"),
    (re.compile(r"\bmean\s+average\s+recall\b|\bmar\b", re.I), "mar"),
    (re.compile(r"\bauc\b|\barea\s+under\s+the\s+curve\b", re.I), "auc"),
    (re.compile(r"\bspecificity\b", re.I), "specificity"),
    (re.compile(r"\bsensitivity\b", re.I), "sensitivity"),
    (re.compile(r"\biou\b|\bintersection\s+over\s+union\b", re.I), "iou"),
    (re.compile(r"\bbleu\b", re.I), "bleu"),
    (re.compile(r"\brouge\b", re.I), "rouge"),
    (re.compile(r"\bndcg\b", re.I), "ndcg"),
    (re.compile(r"\bmrr\b|\bmean\s+reciprocal\s+rank\b", re.I), "mrr"),
    (re.compile(r"\bmean\s+absolute\s+error\b|\bmae\b", re.I), "mae"),
    (re.compile(r"\bmean\s+squared\s+error\b|\bmse\b", re.I), "mse"),
    (re.compile(r"\broot\s+mean\s+squared\s+error\b|\brmse\b", re.I), "rmse"),
    (re.compile(r"\br\s+squared\b|\br2\b", re.I), "r2"),
)

_EFFICIENCY_METRIC_RULES = (
    (re.compile(r"\binference\s+(?:time|latency)\b", re.I), "inference time"),
    (re.compile(r"\btraining\s+time\b", re.I), "training time"),
    (
        re.compile(
            r"\bcomputational\s+(?:cost|complexity|time|overhead)\b|"
            r"\bcomputation(?:al)?\s+cost\b",
            re.I,
        ),
        "computational cost",
    ),
    (
        re.compile(r"\bcommunication\s+(?:cost|costs|overhead)\b", re.I),
        "communication cost",
    ),
    (
        re.compile(r"\bparameter\s+count\b|\btrainable\s+parameters?\b", re.I),
        "parameter count",
    ),
    (re.compile(r"\bmemory\s+(?:usage|footprint)\b", re.I), "memory usage"),
    (re.compile(r"\bflops?\b|\bfloating\s+point\s+operations\b", re.I), "flops"),
)


def normalize_metric(value: str) -> str:
    normalized = _clean(value)

    for pattern, label in (*_PERFORMANCE_METRIC_RULES, *_EFFICIENCY_METRIC_RULES):
        if pattern.search(normalized):
            return label

    return normalized if _is_concrete_phrase(normalized, max_words=6) else ""


def metric_kind(value: str) -> str | None:
    normalized = normalize_metric(value)

    if not normalized:
        return None

    if any(label == normalized for _, label in _EFFICIENCY_METRIC_RULES):
        return "efficiency"

    return "performance"


# ---------------------------------------------------------------------------
# Constraint normalization
# ---------------------------------------------------------------------------

_LIMITED_LABEL_PATTERN = re.compile(
    r"\b(?:"
    r"few\s+shot|"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+shot|"
    r"low\s+data|data\s+scarcity|label\s+scarcity|annotation\s+scarcity|"
    r"label\s+budget|annotation\s+budget|"
    r"(?:limited|small|scarce)\s+(?:labeled|labelled|annotated)\s+"
    r"(?:training\s+)?(?:data|dataset|datasets|samples?|examples?|images?)|"
    r"limited\s+(?:annotations?|annotated\s+data|training\s+data)|"
    r"(?:data|label)\s+efficient\s+(?:training|learning|setting|classification)|"
    r"(?:only|using|with)\s+(?:only\s+)?\d+(?:\.\d+)?"
    r"(?:\s*%|\s+percent)?\s+(?:of\s+)?(?:the\s+)?"
    r"(?:labeled|labelled|annotated)\s+(?:training\s+)?"
    r"(?:data|samples?|examples?|images?)|"
    r"(?:\d+|n)\s+(?:labeled|labelled|annotated)\s+"
    r"(?:training\s+)?(?:samples?|examples?|images?)\s+per\s+class"
    r")\b",
    re.I,
)

_NEGATIVE_GENERALIZATION_PATTERN = re.compile(
    r"\b(?:poor|limited|weak|reduced|degraded|insufficient)\s+generalization\b|"
    r"\bgeneralization\s+(?:problem|problems|challenge|challenges|"
    r"limitation|limitations|failure|failures|gap|gaps)\b",
    re.I,
)


def normalize_constraint(value: str) -> str:
    normalized = _clean(value)

    if not normalized or normalized in _VAGUE_VALUES:
        return ""

    if _LIMITED_LABEL_PATTERN.search(normalized):
        return "limited labeled data"

    if _NEGATIVE_GENERALIZATION_PATTERN.search(normalized):
        return "poor generalization"

    return normalized if _is_concrete_phrase(normalized, max_words=12) else ""


# ---------------------------------------------------------------------------
# Dataset characteristics
# ---------------------------------------------------------------------------

_DATASET_TYPE_RULES = (
    (
        re.compile(r"\b(?:public|open access|openly available|benchmark)\b", re.I),
        "public/benchmark",
    ),
    (
        re.compile(r"\b(?:private|proprietary|institutional)\b", re.I),
        "private/proprietary",
    ),
    (
        re.compile(r"\bsynthetic\b|\bsimulat(?:ed|ion)\b", re.I),
        "synthetic",
    ),
    (
        re.compile(
            r"\b(?:real world|real life|in the wild|field collected|"
            r"field data|natural environment)\b",
            re.I,
        ),
        "real-world",
    ),
    (re.compile(r"\blongitudinal\b", re.I), "longitudinal"),
    (re.compile(r"\bmultimodal\b|\bmulti modal\b", re.I), "multimodal"),
    (
        re.compile(r"\b(?:multi center|multicenter|multi site|multisite)\b", re.I),
        "multi-site",
    ),
)


def dataset_types(record: PaperEvidence) -> list[str]:
    values = [
        *(item.value for item in record.datasets),
        *(item.evidence_text for item in record.datasets),
        *(item.value for item in record.population_or_setting),
        *(item.evidence_text for item in record.population_or_setting),
    ]

    text = " ".join(values)

    return list(
        dict.fromkeys(
            label
            for pattern, label in _DATASET_TYPE_RULES
            if pattern.search(text)
        )
    )


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