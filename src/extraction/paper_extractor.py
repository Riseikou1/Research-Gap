"""OpenAI Structured Outputs backend for paper evidence extraction."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import openai_api_key, openai_extraction_model
from src.models.paper import Paper

from .evidence import EvidenceItem, LimitationEvidence, PaperEvidence, StudyType, canonical_evidence_key


LOGGER = logging.getLogger(__name__)


class PaperExtractionError(RuntimeError):
    """Raised when a paper cannot be converted into structured evidence."""


class _LimitationClaim(EvidenceItem):
    author_stated: bool


class _MethodClaim(EvidenceItem):
    role: Literal["primary", "supporting", "comparison"]


class _ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    research_objective: EvidenceItem | None = Field(
        default=None,
        description="The main research problem, objective, or question explicitly investigated by the paper.",
    )
    population_or_setting: list[EvidenceItem] = Field(
        default_factory=list,
        description="Explicit populations, application domains, environments, or experimental settings.",
    )
    method_or_intervention: list[_MethodClaim] = Field(
        default_factory=list,
        description=(
            "Methods, models, algorithms, interventions, or architectures actually studied. "
            "Each claim must be marked primary, supporting, or comparison."
        ),
    )
    comparison_or_baseline: list[EvidenceItem] = Field(
        default_factory=list,
        description="Methods or systems explicitly compared with or evaluated against the focal method.",
    )
    datasets: list[EvidenceItem] = Field(
        default_factory=list,
        description="Named datasets or sufficiently specific dataset descriptions.",
    )
    sample_size: EvidenceItem | None = Field(
        default=None,
        description="An explicit numerical sample count such as images, participants, documents, records, or examples.",
    )
    evaluation_metrics: list[EvidenceItem] = Field(
        default_factory=list,
        description="Explicit evaluation metrics such as accuracy, F1, AUC, BLEU, ROUGE, or NDCG.",
    )
    main_findings: list[EvidenceItem] = Field(
        default_factory=list,
        description="Major empirical findings or conclusions explicitly reported by the authors.",
    )
    constraints: list[EvidenceItem] = Field(
        default_factory=list,
        description=(
            "Explicit experimental, data, resource, deployment, generalization, or evaluation constraints. "
            "The value must describe the constraint itself, never merely name a method or model."
        ),
    )
    limitations: list[_LimitationClaim] = Field(
        default_factory=list,
        description="Limitations or weaknesses explicitly attributed to the paper by its authors.",
    )
    future_work: list[EvidenceItem] = Field(
        default_factory=list,
        description="Concrete future research directions explicitly proposed by the authors.",
    )
    study_type: StudyType = Field(
        default="other",
        description="Study classification based only on the supplied title and abstract.",
    )
    extraction_confidence: float = Field(ge=0.0, le=1.0)


_INSTRUCTIONS = """
Extract structured research evidence ONLY from the supplied title and abstract.

Do not use outside knowledge.
Do not infer missing experimental details.
Do not invent evidence.

For every claim:
- value must be concise;
- evidence_text must be copied directly from the title or abstract;
- source must be exactly "title" or "abstract";
- confidence describes extraction confidence, not scientific truth.

METHODS

Extract focal methods, models, algorithms, architectures, training strategies,
or interventions that the paper actually studies.

Each method claim must have one role:

primary:
The method is a focal or genuinely co-equal studied approach.

supporting:
An implementation component or procedure such as preprocessing, augmentation,
normalization, filtering, scheduling, or hyperparameter tuning.

comparison:
A method explicitly evaluated against a distinct focal approach.

Fine-tuning, full fine-tuning, parameter-efficient fine-tuning, pre-training,
transfer learning, few-shot learning, and similar strategies MAY themselves be
scientific methods. Do not automatically classify them as implementation
details merely because they describe a training procedure.

Preserve informative named approaches. Prefer a specific complete focal method
over a vague parent description when both refer to the same approach.

Do not put background or related-work methods into the studied method list.

COMPARISONS

Extract methods, models, systems, or approaches explicitly compared,
benchmarked, evaluated against, tested against, or used as baselines.

The source does not need to use the literal word "baseline".

Do not create comparison evidence merely because another method appears in
background, motivation, prior work, or a generic criticism.

For a neutral head-to-head study where several approaches are genuinely
studied equally, they may all be primary methods instead of forcing one to
become a baseline.

DATASETS

Extract named datasets or sufficiently specific dataset descriptions.

Do not create dataset entities from vague phrases such as:
- benchmark datasets
- several public datasets
- widely used datasets

unless the data can actually be identified.

SAMPLE SIZE

Extract an explicit numerical number of images, samples, participants,
documents, records, instances, examples, or another clearly defined sample
unit.

Do not infer sample size from:
- number of classes;
- percentages alone;
- accuracy values;
- unrelated numerical quantities.

METRICS

Extract only explicitly stated evaluation metric names.

FINDINGS

Extract major reported results or conclusions.

Preserve important numerical results when explicitly stated.
Avoid duplicate paraphrases of the same finding.

CONSTRAINTS

A constraint must describe an explicit condition under which the study,
training, evaluation, deployment, or data collection occurs.

Examples include:
- few-shot or limited-label training;
- non-IID data;
- communication constraints;
- computational or memory limits;
- class imbalance;
- privacy constraints;
- deployment or field-condition constraints;
- domain-shift or generalization constraints.

Do NOT convert a method name into a constraint.

For example:

"Data-efficient Image Transformer (DeiT)"

is a method name by itself.

It does NOT prove that the study experimentally uses limited labeled data.

Limited-label evidence requires an explicit condition such as:
- five-shot training;
- a small labeled training dataset;
- 10% labeled samples;
- a stated label or annotation budget.

When both a method and an experimental constraint are present, extract them as
separate claims.

LIMITATIONS

Extract only limitations, weaknesses, shortcomings, unresolved problems, or
constraints that the authors explicitly attribute to the study.

Do not generate your own criticism.

FUTURE WORK

Extract only a concrete future research direction.

Do NOT extract generic statements such as:
- future work is discussed;
- further research is needed;
- future directions are presented.

The source must say what should actually be investigated, evaluated,
developed, extended, compared, tested, deployed, or validated.

STUDY TYPE

empirical:
The paper tests or evaluates methods using data or experiments.

review:
A literature review.

survey:
A survey of research, systems, or methods.

methodological:
Primarily a methodological or framework contribution whose empirical status
is unclear.

otherwise:
other.
""".strip()


_GENERIC_DATASET_VALUES = {
    "dataset",
    "datasets",
    "benchmark dataset",
    "benchmark datasets",
    "widely used dataset",
    "widely used datasets",
    "widely used benchmark dataset",
    "widely used benchmark datasets",
    "public dataset",
    "public datasets",
}

_GENERIC_DATASET_KEYS = {
    canonical_evidence_key(value)
    for value in _GENERIC_DATASET_VALUES
}

_EXPLICIT_COMPARISON_PATTERN = re.compile(
    r"\b(?:compar(?:e|ed|ing|ison)|benchmark(?:ed|s)?|"
    r"evaluat(?:e|ed|ing)|test(?:ed|s|ing)?|outperform(?:ed|s|ing)?|"
    r"against|versus|vs\.?|baseline|relative to)\b",
    re.IGNORECASE,
)

_BACKGROUND_COMPARISON_PATTERN = re.compile(
    r"\b(?:background|related work|prior work|conventional|traditional|"
    r"existing approaches?|limitations?|slow|expensive|challenging|"
    r"shortcomings?)\b",
    re.IGNORECASE,
)

# These are implementation details only when a more informative focal method
# exists. Deliberately DO NOT include fine-tuning, pre-training, transfer
# learning, optimization, few-shot learning, etc. Those may themselves be the
# scientific intervention being studied.
_GENERIC_METHOD_DETAIL_PATTERN = re.compile(
    r"\b(?:implementation detail|preprocessing|pre-processing|"
    r"data augmentation|normalization|filtering|"
    r"learning[- ]rate scheduling|hyperparameter tuning|"
    r"postprocessing|post-processing)\b",
    re.IGNORECASE,
)

_SAMPLE_SIZE_PATTERN = re.compile(
    r"\b(?:dataset|subset|test set|training set|validation set|sample|cohort)"
    r"[^.!?]{0,40}?\b(?:of|with|containing|consisting of)?\s*"
    r"(\d[\d,]*)\s+"
    r"(images?|samples?|participants?|patients?|documents?|records?|instances?|examples?)\b",
    re.IGNORECASE,
)

_FUTURE_ACTION_PATTERN = re.compile(
    r"\b(?:assess|adapt|analy[sz]e|apply|benchmark|collect|compare|"
    r"conduct|develop|deploy|design|evaluate|examine|extend|explore|"
    r"improve|implement|investigate|measure|study|test|validate)\w*\b",
    re.IGNORECASE,
)


class PaperExtractor:
    """Extract evidence for papers using OpenAI Structured Outputs."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        model: str | None = None,
        max_output_tokens: int = 2400,
        evidence_limit: int = 10,
    ) -> None:
        if max_output_tokens <= 0 or evidence_limit < 0:
            raise ValueError("max_output_tokens must be positive and evidence_limit non-negative")

        self.model = model or openai_extraction_model()
        self.max_output_tokens = max_output_tokens
        self.evidence_limit = evidence_limit
        self.failures: list[PaperExtractionError] = []

        if client is not None:
            self.client = client
            return

        key = api_key or openai_api_key()
        if not key:
            raise PaperExtractionError("OPENAI_API_KEY is required for evidence extraction.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PaperExtractionError("The OpenAI package is required for evidence extraction.") from exc

        self.client = OpenAI(api_key=key)

    def extract(self, paper: Paper) -> PaperEvidence:
        title = paper.title.strip()
        abstract = paper.abstract.strip() if paper.abstract else None

        if not title:
            raise PaperExtractionError("Paper title is required for evidence extraction.")

        source = f"Title:\n{title}"
        if abstract:
            source += f"\n\nAbstract:\n{abstract}"

        try:
            response = self.client.responses.parse(
                model=self.model,
                reasoning={"effort": "low"},
                store=False,
                max_output_tokens=self.max_output_tokens,
                instructions=_INSTRUCTIONS,
                input=source,
                text_format=_ExtractionResult,
            )

            payload = getattr(response, "output_parsed", None)
            if not isinstance(payload, _ExtractionResult):
                raise PaperExtractionError("OpenAI returned no parsed evidence payload.")

            return _to_evidence(paper, payload)

        except PaperExtractionError:
            raise
        except (ValidationError, TypeError, ValueError) as exc:
            raise PaperExtractionError(f"Invalid evidence response: {exc}") from exc
        except Exception as exc:
            raise PaperExtractionError(f"Evidence extraction failed: {exc}") from exc

    def extract_many(
        self,
        papers: Sequence[Paper],
        limit: int | None = None,
    ) -> list[PaperEvidence]:
        self.failures = []

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        selected = list(papers)[: self.evidence_limit if limit is None else limit]
        results: list[PaperEvidence] = []

        for paper in selected:
            try:
                results.append(self.extract(paper))
            except PaperExtractionError as exc:
                self.failures.append(PaperExtractionError(f"{paper.id}: {exc}"))

        return results


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _is_supported(item: EvidenceItem, paper: Paper) -> bool:
    """Require evidence_text to exist verbatim in its declared source."""

    source_text = paper.title if item.source == "title" else paper.abstract or ""
    supported = _normalize(item.evidence_text) in _normalize(source_text)

    if not supported:
        LOGGER.debug(
            "Rejected unsupported evidence for paper %s: %r",
            paper.id,
            item.evidence_text,
        )

    return supported


def _clean_items(
    items: Sequence[EvidenceItem],
    paper: Paper,
    *,
    drop_generic_datasets: bool = False,
) -> list[EvidenceItem]:
    cleaned: list[EvidenceItem] = []
    seen: set[str] = set()

    for item in items:
        if not _is_supported(item, paper):
            continue

        key = canonical_evidence_key(item.value)

        if drop_generic_datasets and key in _GENERIC_DATASET_KEYS:
            continue

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(item)

    return cleaned


def _clean_single(
    item: EvidenceItem | None,
    paper: Paper,
) -> EvidenceItem | None:
    if item is None or not _is_supported(item, paper):
        return None

    return item


def _is_explicit_comparison(item: EvidenceItem) -> bool:
    """Reject comparison claims that are clearly background-only."""

    evidence = item.evidence_text

    if _EXPLICIT_COMPARISON_PATTERN.search(evidence):
        return True

    if _BACKGROUND_COMPARISON_PATTERN.search(evidence):
        return False

    # The structured extractor has explicitly assigned this value to a
    # comparison field. Keep ambiguous but supported claims rather than
    # pretending deterministic regex can fully understand the sentence.
    return True


def _clean_comparisons(
    items: Sequence[EvidenceItem],
    paper: Paper,
) -> list[EvidenceItem]:
    return _clean_items(
        [item for item in items if _is_explicit_comparison(item)],
        paper,
    )


def _remove_generic_primary_methods(
    items: list[EvidenceItem],
) -> list[EvidenceItem]:
    """Drop obvious implementation details when real focal methods exist."""

    if len(items) < 2:
        return items

    generic = [
        item
        for item in items
        if _GENERIC_METHOD_DETAIL_PATTERN.search(item.value)
    ]

    if not generic or len(generic) == len(items):
        return items

    generic_keys = {
        canonical_evidence_key(item.value)
        for item in generic
    }

    return [
        item
        for item in items
        if canonical_evidence_key(item.value) not in generic_keys
    ]


def _clean_methods(
    items: Sequence[_MethodClaim],
    paper: Paper,
) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    primary = _remove_generic_primary_methods(
        _clean_items(
            [item for item in items if item.role == "primary"],
            paper,
        )
    )

    comparisons = _clean_comparisons(
        [item for item in items if item.role == "comparison"],
        paper,
    )

    # Supporting claims intentionally disappear from PaperEvidence.methods.
    # They remain implementation details, not focal scientific methods.
    return primary, comparisons


def _clean_future_work(
    items: Sequence[EvidenceItem],
    paper: Paper,
) -> list[EvidenceItem]:
    supported = _clean_items(items, paper)

    return [
        item
        for item in supported
        if _FUTURE_ACTION_PATTERN.search(item.evidence_text)
    ]


def _same_role_entity(
    left: str,
    right: str,
) -> bool:
    """Return true only for the same normalized extracted entity.

    Do not use architecture-name blacklists here. If the extractor puts the
    exact same entity in both a method role and a constraint role, the method
    role wins because a bare scientific method is not itself a constraint.
    """

    return canonical_evidence_key(left) == canonical_evidence_key(right)


def _clean_constraints(
    items: Sequence[EvidenceItem],
    paper: Paper,
    *,
    methods: Sequence[EvidenceItem],
    comparisons: Sequence[EvidenceItem],
) -> list[EvidenceItem]:
    """Validate constraints without maintaining a model-name blacklist.

    This deliberately avoids rules such as "ViT is a method, BERT is a method,
    LoRA is a method..." because Research GAP is meant to work across domains.

    Instead:
    1. evidence must be verbatim-supported;
    2. exact duplicate scientific entities already classified as methods or
       comparisons cannot also survive as bare constraints.

    Semantic constraint-specific rules are handled downstream where the
    research idea is known, such as explicit low-label matching.
    """

    supported = _clean_items(items, paper)
    scientific_entities = [*methods, *comparisons]

    return [
        item
        for item in supported
        if not any(
            _same_role_entity(item.value, entity.value)
            for entity in scientific_entities
        )
    ]


def _fallback_sample_size(
    paper: Paper,
) -> EvidenceItem | None:
    if not paper.abstract:
        return None

    match = _SAMPLE_SIZE_PATTERN.search(paper.abstract)
    if not match:
        return None

    count, unit = match.groups()

    return EvidenceItem(
        value=f"{count} {unit}",
        evidence_text=match.group(0),
        source="abstract",
        confidence=0.95,
    )


def _unique_items(
    items: Sequence[EvidenceItem],
) -> list[EvidenceItem]:
    result: list[EvidenceItem] = []
    seen: set[str] = set()

    for item in items:
        key = canonical_evidence_key(item.value)

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def _to_evidence(
    paper: Paper,
    payload: _ExtractionResult,
) -> PaperEvidence:
    limitations = [
        LimitationEvidence(
            value=item.value,
            evidence_text=item.evidence_text,
            source=item.source,
            confidence=item.confidence,
        )
        for item in payload.limitations
        if item.author_stated and _is_supported(item, paper)
    ]

    methods, role_comparisons = _clean_methods(
        payload.method_or_intervention,
        paper,
    )

    explicit_comparisons = _clean_comparisons(
        payload.comparison_or_baseline,
        paper,
    )

    primary_keys = {
        canonical_evidence_key(item.value)
        for item in methods
    }

    comparisons = _unique_items([
        item
        for item in [*role_comparisons, *explicit_comparisons]
        if canonical_evidence_key(item.value) not in primary_keys
    ])

    constraints = _clean_constraints(
        payload.constraints,
        paper,
        methods=methods,
        comparisons=comparisons,
    )

    sample_size = _clean_single(
        payload.sample_size,
        paper,
    )

    if sample_size is None:
        sample_size = _fallback_sample_size(paper)

    return PaperEvidence(
        paper_id=paper.id,
        title=paper.title,
        study_type=payload.study_type,
        research_objective=_clean_single(
            payload.research_objective,
            paper,
        ),
        population_or_setting=_clean_items(
            payload.population_or_setting,
            paper,
        ),
        method_or_intervention=methods,
        comparison_or_baseline=comparisons,
        datasets=_clean_items(
            payload.datasets,
            paper,
            drop_generic_datasets=True,
        ),
        sample_size=sample_size,
        evaluation_metrics=_clean_items(
            payload.evaluation_metrics,
            paper,
        ),
        main_findings=_clean_items(
            payload.main_findings,
            paper,
        ),
        constraints=constraints,
        limitations=limitations,
        future_work=_clean_future_work(
            payload.future_work,
            paper,
        ),
        extraction_confidence=payload.extraction_confidence,
    )
