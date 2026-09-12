"""OpenAI Structured Outputs backend for paper evidence extraction."""

from __future__ import annotations

import os
import hashlib
import logging
import re
from time import perf_counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import cache_dir, openai_api_key, openai_extraction_model
from src.models.paper import Paper

from .evidence import (
    EvidenceItem,
    EvidenceSource,
    LimitationEvidence,
    PaperCoverageRecord,
    PaperEvidence,
    StudyType,
    canonical_evidence_key,
)
from .evidence import ExtractionCoverage
from .document import PaperDocument, PaperSection, SectionType, build_extraction_context
from .full_text import PARSER_VERSION, FullTextClient
from .store import EvidenceStore


LOGGER = logging.getLogger(__name__)


# Increment when the structured extraction contract or its compatibility
# assumptions change. Old cache rows remain harmless misses after a bump.
EVIDENCE_SCHEMA_VERSION = 8


class PaperExtractionError(RuntimeError):
    """Raised when a paper cannot be converted into structured evidence."""


class _LimitationClaim(EvidenceItem):
    author_stated: bool


class _MethodClaim(EvidenceItem):
    role: Literal["primary", "supporting", "comparison"]


class _ExtractionClaim(BaseModel):
    """Typed provider transport before provenance-dependent validation.

    OpenAI Structured Outputs needs one JSON schema for abstract and full-text
    requests. Section fields therefore remain nullable at this boundary. A
    single bounded normalization pass below removes impossible section
    provenance before the strict ``EvidenceItem`` domain model is constructed.
    """

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    value: str = Field(min_length=1)
    canonical_value: str | None = None
    evidence_text: str = Field(min_length=1)
    source: EvidenceSource
    confidence: float = Field(ge=0.0, le=1.0)
    section_type: SectionType | None = Field(
        default=None,
        description="Only for source=full_text. Must be null for title or abstract evidence.",
    )
    section_heading: str | None = Field(
        default=None,
        description="Only for source=full_text. Must be null for title or abstract evidence.",
    )
    section_id: str | None = Field(
        default=None,
        description="Only for source=full_text. Must be null for title or abstract evidence.",
    )


class _ExtractionLimitationClaim(_ExtractionClaim):
    author_stated: bool


class _ExtractionMethodClaim(_ExtractionClaim):
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
    data_or_modality: list[EvidenceItem] = Field(
        default_factory=list,
        description=(
            "Input data, measurements, signals, sensing modalities, source data, "
            "or input representations explicitly used by the study."
        ),
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


class _ExtractionTransportResult(BaseModel):
    """Provider payload whose members are normalized into `_ExtractionResult`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    research_objective: _ExtractionClaim | None = None
    population_or_setting: list[_ExtractionClaim] = Field(default_factory=list)
    method_or_intervention: list[_ExtractionMethodClaim] = Field(default_factory=list)
    comparison_or_baseline: list[_ExtractionClaim] = Field(
        default_factory=list,
        description="Methods or systems explicitly compared with the focal method.",
    )
    data_or_modality: list[_ExtractionClaim] = Field(default_factory=list)
    datasets: list[_ExtractionClaim] = Field(
        default_factory=list,
        description="Named datasets or source-explicit, identifiable dataset descriptions; not merely input requirements or an application setting.",
    )
    sample_size: _ExtractionClaim | None = Field(
        default=None,
        description="An explicit numerical sample count with a defined unit, such as participants, records, documents, or examples.",
    )
    evaluation_metrics: list[_ExtractionClaim] = Field(
        default_factory=list,
        description="Only explicitly stated evaluation metric names; outcome statements belong in main_findings.",
    )
    main_findings: list[_ExtractionClaim] = Field(default_factory=list)
    constraints: list[_ExtractionClaim] = Field(default_factory=list)
    limitations: list[_ExtractionLimitationClaim] = Field(default_factory=list)
    future_work: list[_ExtractionClaim] = Field(default_factory=list)
    study_type: StudyType = "other"
    extraction_confidence: float = Field(ge=0.0, le=1.0)


class _BatchPaperExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    paper_id: str = Field(min_length=1)
    evidence: _ExtractionTransportResult


class _BatchExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    papers: list[_BatchPaperExtraction]


_INSTRUCTIONS = """
Extract structured research evidence ONLY from the supplied title and abstract.

Do not use outside knowledge.
Do not infer missing experimental details.
Do not invent evidence.

For every claim:
- value must be concise;
- canonical_value must be a concise, facet-scoped normalized semantic concept
  for the claim, not a summary of the whole sentence;
- equivalent surface forms must use the same canonical_value, while distinct
  concepts and meaningful qualifiers must remain distinct;
- evidence_text must be copied directly from the title or abstract;
- source must be exactly "title" or "abstract";
- section_type, section_heading, and section_id must be absent or null for
  title and abstract evidence; these fields are valid only for a claim copied
  from an actual supplied full-text section;
- confidence describes extraction confidence, not scientific truth.

CANONICAL CONCEPTS

For every extracted claim, provide canonical_value using only the supplied
title and abstract. Canonical values are generic semantic identities for
matching and grouping, not a domain taxonomy. Preserve the claim's important
qualifiers in value and evidence_text, but do not copy unrelated facet context
into canonical_value. For example, a data identity should describe the core
data or modality rather than its source/target role, label status, or sample
count when those are separate claims. A problem identity should describe the
core task and subject rather than repeating the method, modality, sampling
regime, or other facet represented elsewhere. A constraint identity should
describe the core restriction rather than the full experiment sentence.
Use a short noun phrase, not a multi-clause summary.

DATA OR MODALITY

Extract input data, measurements, signals, sensing modalities, source data,
or input representations explicitly used by the study into data_or_modality.
Do not infer a modality merely because it is common for the method or domain.
Do not put a modality in method_or_intervention or domain.

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

Natural-language requirements, an application setting, or another input
description is not by itself a dataset. Put it in data_or_modality or
population_or_setting when appropriate unless the source identifies it as a
dataset or data collection.

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

An outcome such as "reduced hallucination" or "improved generalization" is a
finding, not a metric name. Extract a metric only when the source explicitly
names what was measured, such as Hallucination Rate, accuracy, F1, or latency.

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


_BATCH_INSTRUCTIONS = _INSTRUCTIONS + """

The input contains several papers. Return exactly one `papers` item for each
supplied paper identifier when possible. The paper_id must be copied exactly
from the input. Never use evidence from one paper in another paper's item.
If a paper cannot be extracted reliably, omit only that paper so it can be
retried individually; valid sibling items must still be returned.
"""

_FULL_TEXT_INSTRUCTIONS = _INSTRUCTIONS.replace(
    "ONLY from the supplied title and abstract",
    "ONLY from the supplied title, abstract, and bounded full-text sections",
).replace(
    'source must be exactly "title" or "abstract";',
    'source must be "title", "abstract", or "full_text"; for full_text, copy the section id, original heading, and one listed semantic section type;',
) + """

Full-text blocks are explicitly labeled with a stable section id, semantic types,
and original heading. For full-text claims, section_id and section_heading must be
copied exactly and section_type must be one of that block's listed types. Do not
claim that an unextracted field is absent from the paper.
"""


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
        max_workers: int = max(1, (os.cpu_count() or 2) // 2),
        cache_path: str | Path | None = None,
        cache_database_url: str | None = None,
        batch_size: int = 1,
        max_batch_input_chars: int = 24000,
        full_text_client: FullTextClient | None = None,
        max_full_text_context_chars: int = 30000,
    ) -> None:
        if (
            max_output_tokens <= 0
            or evidence_limit < 0
            or max_workers <= 0
            or batch_size <= 0
            or max_batch_input_chars <= 0
            or max_full_text_context_chars <= 0
        ):
            raise ValueError(
                "max_output_tokens must be positive, evidence_limit non-negative, "
                "max_workers and batch limits must be positive"
            )

        self.model = model or openai_extraction_model()
        self.max_output_tokens = max_output_tokens
        self.evidence_limit = evidence_limit
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_batch_input_chars = max_batch_input_chars
        self.full_text_client = full_text_client
        self.max_full_text_context_chars = max_full_text_context_chars
        self.failures: list[PaperExtractionError] = []
        self.coverage_records: list[PaperCoverageRecord] = []
        self._attempt_documents: dict[str, PaperDocument] = {}
        self._cache_lock = RLock()
        self._cache: dict[tuple[str, str, str, str], PaperEvidence] = {}
        self._inflight: dict[tuple[str, str, str, str], Future[PaperEvidence]] = {}
        self._metrics: dict[str, int] = {
            "evidence_requested": 0,
            "memory_cache_hits": 0,
            "persistent_cache_hits": 0,
            "new_evidence_extractions": 0,
            "openai_extraction_requests": 0,
            "extraction_batch_requests": 0,
            "extraction_batch_members": 0,
            "extraction_fallback_requests": 0,
            "extraction_repair_attempts": 0,
            "extraction_repaired_claims": 0,
            "extraction_failed_members": 0,
            "evidence_inflight_hits": 0,
            "full_text_extraction_fallback_requests": 0,
        }
        self._timings: dict[str, float] = {
            "initial_evidence_extraction_api_wait": 0.0,
        }

        if client is not None:
            self.client = client
            # Unit-test fakes stay isolated by default. Supplying cache_path
            # explicitly enables the same persistent behavior for them.
            self.evidence_store = EvidenceStore(cache_path, database_url=cache_database_url)
            return

        key = api_key or openai_api_key()
        if not key:
            raise PaperExtractionError("OPENAI_API_KEY is required for evidence extraction.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise PaperExtractionError("The OpenAI package is required for evidence extraction.") from exc

        self.client = OpenAI(api_key=key)
        self.evidence_store = EvidenceStore(
            cache_path if cache_path is not None else cache_dir() / "research_gap.sqlite3",
            database_url=cache_database_url,
        )

    def _cache_key(
        self,
        paper: Paper,
        model: str,
        document: PaperDocument | None = None,
    ) -> tuple[str, str, str, str]:
        content_hash = EvidenceStore.content_hash(paper)
        if self.full_text_client is not None:
            full_text_identity = "\0".join(
                [PARSER_VERSION, *(item.url for item in paper.full_text_locations)]
            )
            if document is not None:
                full_text_identity += "\0" + document.model_dump_json()
            content_hash = hashlib.sha256(
                f"{content_hash}\0{full_text_identity}".encode("utf-8")
            ).hexdigest()
        return (
            paper.id,
            content_hash,
            model,
            str(EVIDENCE_SCHEMA_VERSION),
        )

    def metrics_snapshot(self) -> dict[str, int]:
        """Return thread-safe cumulative work counters."""

        with self._cache_lock:
            return dict(self._metrics)

    def timings_snapshot(self) -> dict[str, float]:
        with self._cache_lock:
            return dict(self._timings)

    def get_or_extract(self, paper: Paper) -> PaperEvidence:
        """Shared evidence access point for every pipeline stage."""

        return self.extract(paper)

    def get_many_or_extract(
        self,
        papers: Sequence[Paper],
        limit: int | None = None,
    ) -> list[PaperEvidence]:
        """Get or extract papers in deterministic input order."""

        return self.extract_many(papers, limit=limit)

    def _load_document(self, paper: Paper) -> PaperDocument:
        """Resolve one paper independently; acquisition errors become coverage."""

        if self.full_text_client is None:
            document = PaperDocument(paper_id=paper.id, status="not_attempted")
            with self._cache_lock:
                self._attempt_documents[paper.id] = document
            return document
        try:
            document = self.full_text_client.load(paper)
        except Exception as exc:
            LOGGER.info("full-text enrichment failed paper=%s error=%s", paper.id, exc)
            document = PaperDocument(
                paper_id=paper.id, status="fetch_failed",
                notices=[f"full-text enrichment failed: {str(exc)[:240]}"],
            )
        with self._cache_lock:
            self._attempt_documents[paper.id] = document
        return document

    def _extract_uncached(
        self,
        paper: Paper,
        document: PaperDocument | None = None,
        *,
        use_full_text: bool = True,
        fallback_explanation: str | None = None,
    ) -> PaperEvidence:
        title = paper.title.strip()
        abstract = paper.abstract.strip() if paper.abstract else None

        if not title:
            raise PaperExtractionError("Paper title is required for evidence extraction.")

        source = f"Title:\n{title}"
        if abstract:
            source += f"\n\nAbstract:\n{abstract}"

        document = document or self._load_document(paper)
        inspected_sections: list[PaperSection] = []
        context_truncated = False
        instructions = _INSTRUCTIONS
        if use_full_text and document.status == "usable":
            context, inspected_sections, context_truncated = build_extraction_context(
                document, max_chars=self.max_full_text_context_chars,
            )
            if context:
                source += f"\n\n{context}"
                instructions = _FULL_TEXT_INSTRUCTIONS

        try:
            with self._cache_lock:
                self._metrics["openai_extraction_requests"] += 1
            started = perf_counter()
            response = self.client.responses.parse(
                # Count provider requests, rather than papers, for work
                # accounting. The existing one-paper request contract stays
                # unchanged and remains the reliable fallback.
                model=self.model,
                reasoning={"effort": "low"},
                store=False,
                max_output_tokens=self.max_output_tokens,
                instructions=instructions,
                input=source,
                text_format=_ExtractionTransportResult,
            )
            with self._cache_lock:
                self._timings["initial_evidence_extraction_api_wait"] += (
                    perf_counter() - started
                )

            payload = getattr(response, "output_parsed", None)
            if not isinstance(payload, (_ExtractionTransportResult, _ExtractionResult)):
                raise PaperExtractionError("OpenAI returned no parsed evidence payload.")

            strict_payload, repaired = _normalize_extraction_payload(
                payload,
                paper,
                full_text_allowed=bool(inspected_sections),
            )
            if repaired:
                with self._cache_lock:
                    self._metrics["extraction_repair_attempts"] += 1
                    self._metrics["extraction_repaired_claims"] += repaired
                LOGGER.info(
                    "normalized extraction provenance paper=%s repaired_claims=%d",
                    paper.id,
                    repaired,
                )

            return _to_evidence(
                paper, strict_payload, document=document,
                inspected_sections=inspected_sections,
                context_truncated=context_truncated,
                full_text_requested=self.full_text_client is not None,
                full_text_used=bool(inspected_sections),
                fallback_explanation=(
                    fallback_explanation
                    or _fallback_explanation(document, paper)
                ),
            )

        except PaperExtractionError:
            raise
        except (ValidationError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "invalid structured evidence response paper=%s error_type=%s error=%s",
                paper.id,
                type(exc).__name__,
                exc,
            )
            raise PaperExtractionError(f"Invalid evidence response: {exc}") from exc
        except Exception as exc:
            LOGGER.warning(
                "evidence provider request failed paper=%s error_type=%s error=%s",
                paper.id,
                type(exc).__name__,
                exc,
            )
            raise PaperExtractionError(f"Evidence extraction failed: {exc}") from exc

    def _extract_uncached_batch(
        self,
        papers: Sequence[Paper],
    ) -> dict[str, PaperEvidence]:
        """Extract a bounded batch and map results by unambiguous cache keys."""

        inputs: list[str] = []
        request_ids: dict[str, tuple[str, Paper]] = {}
        for paper in papers:
            cache_key = self._cache_key(paper, self.model)
            request_id = f"{paper.id}::{cache_key[1]}"
            title = paper.title.strip()
            if not title:
                raise PaperExtractionError(
                    f"{paper.id}: Paper title is required for evidence extraction."
                )
            source = f"Paper ID: {request_id}\nTitle:\n{title}"
            if paper.abstract:
                source += f"\n\nAbstract:\n{paper.abstract.strip()}"
            inputs.append(source)
            request_ids[request_id] = (cache_key[1], paper)

        try:
            with self._cache_lock:
                self._metrics["openai_extraction_requests"] += 1
                self._metrics["extraction_batch_requests"] += 1
                self._metrics["extraction_batch_members"] += len(papers)
            started = perf_counter()
            response = self.client.responses.parse(
                model=self.model,
                reasoning={"effort": "low"},
                store=False,
                max_output_tokens=self.max_output_tokens * len(papers),
                instructions=_BATCH_INSTRUCTIONS,
                input="\n\n--- NEXT PAPER ---\n\n".join(inputs),
                text_format=_BatchExtractionResult,
            )
            with self._cache_lock:
                self._timings["initial_evidence_extraction_api_wait"] += (
                    perf_counter() - started
                )
            payload = getattr(response, "output_parsed", None)
            if not isinstance(payload, _BatchExtractionResult):
                raise PaperExtractionError(
                    "OpenAI returned no parsed batch evidence payload."
                )
        except PaperExtractionError:
            raise
        except (ValidationError, TypeError, ValueError) as exc:
            LOGGER.warning(
                "invalid structured evidence batch members=%d error_type=%s error=%s",
                len(papers),
                type(exc).__name__,
                exc,
            )
            raise PaperExtractionError(f"Invalid batch evidence response: {exc}") from exc
        except Exception as exc:
            LOGGER.warning(
                "evidence batch provider request failed members=%d error_type=%s error=%s",
                len(papers),
                type(exc).__name__,
                exc,
            )
            raise PaperExtractionError(f"Batch evidence extraction failed: {exc}") from exc

        result: dict[str, PaperEvidence] = {}
        seen_request_ids: set[str] = set()
        for item in payload.papers:
            mapping = request_ids.get(item.paper_id)
            if mapping is None or item.paper_id in seen_request_ids:
                continue
            seen_request_ids.add(item.paper_id)
            content_hash, paper = mapping
            try:
                strict_payload, repaired = _normalize_extraction_payload(
                    item.evidence,
                    paper,
                    full_text_allowed=False,
                )
                if repaired:
                    with self._cache_lock:
                        self._metrics["extraction_repair_attempts"] += 1
                        self._metrics["extraction_repaired_claims"] += repaired
                    LOGGER.info(
                        "normalized batch extraction provenance paper=%s repaired_claims=%d",
                        paper.id,
                        repaired,
                    )
                result[content_hash] = _to_evidence(paper, strict_payload)
            except (PaperExtractionError, ValidationError, TypeError, ValueError) as exc:
                # Only this member is invalid; its sibling results remain
                # eligible for completion and caching.
                with self._cache_lock:
                    self._metrics["extraction_failed_members"] += 1
                LOGGER.warning(
                    "invalid structured evidence batch member paper=%s error_type=%s error=%s",
                    paper.id,
                    type(exc).__name__,
                    exc,
                )
                continue
        return result

    def _batch_item_input(self, paper: Paper) -> str:
        request_id = f"{paper.id}::{EvidenceStore.content_hash(paper)}"
        title = paper.title.strip()
        source = f"Paper ID: {request_id}\nTitle:\n{title}"
        if paper.abstract:
            source += f"\n\nAbstract:\n{paper.abstract.strip()}"
        return source

    def _pack_batches(
        self,
        owners: Sequence[tuple[Paper, tuple[str, str, str, str], Future[PaperEvidence]]],
    ) -> list[list[tuple[Paper, tuple[str, str, str, str], Future[PaperEvidence]]]]:
        """Pack by both configured member count and estimated input size."""

        batches: list[list[tuple[Paper, tuple[str, str, str, str], Future[PaperEvidence]]]] = []
        current: list[tuple[Paper, tuple[str, str, str, str], Future[PaperEvidence]]] = []
        current_chars = 0
        separator_chars = len("\n\n--- NEXT PAPER ---\n\n")

        for owner in owners:
            item_chars = len(self._batch_item_input(owner[0]))
            proposed = current_chars + (separator_chars if current else 0) + item_chars
            if current and (
                len(current) >= self.batch_size
                or proposed > self.max_batch_input_chars
            ):
                batches.append(current)
                current = []
                current_chars = 0

            current.append(owner)
            current_chars += item_chars + (separator_chars if len(current) > 1 else 0)

        if current:
            batches.append(current)
        return batches

    def _reserve(
        self,
        paper: Paper,
        document: PaperDocument | None = None,
    ) -> tuple[tuple[str, str, str, str], PaperEvidence | None, Future[PaperEvidence] | None, bool]:
        cache_key = self._cache_key(paper, self.model, document)
        content_hash = cache_key[1]
        with self._cache_lock:
            self._metrics["evidence_requested"] += 1
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._metrics["memory_cache_hits"] += 1
                return cache_key, cached, None, False

            persistent = self.evidence_store.get(
                paper_id=paper.id,
                content_hash=content_hash,
                model=self.model,
                schema_version=EVIDENCE_SCHEMA_VERSION,
            )
            if persistent is not None:
                self._cache[cache_key] = persistent
                self._metrics["persistent_cache_hits"] += 1
                return cache_key, persistent, None, False

            pending = self._inflight.get(cache_key)
            if pending is not None:
                self._metrics["evidence_inflight_hits"] += 1
                return cache_key, None, pending, False

            pending = Future()
            self._inflight[cache_key] = pending
            return cache_key, None, pending, True

    def _finish_success(
        self,
        cache_key: tuple[str, str, str, str],
        result: PaperEvidence,
        pending: Future[PaperEvidence],
    ) -> None:
        with self._cache_lock:
            self._cache[cache_key] = result
            self._metrics["new_evidence_extractions"] += 1
            try:
                self.evidence_store.put(
                    result,
                    content_hash=cache_key[1],
                    model=self.model,
                    schema_version=EVIDENCE_SCHEMA_VERSION,
                )
            except Exception as exc:
                LOGGER.warning(
                    "evidence cache write failed paper=%s error=%s",
                    result.paper_id,
                    exc,
                )
            self._inflight.pop(cache_key, None)
        pending.set_result(result)

    def _finish_failure(
        self,
        cache_key: tuple[str, str, str, str],
        error: BaseException,
        pending: Future[PaperEvidence],
    ) -> None:
        with self._cache_lock:
            self._inflight.pop(cache_key, None)
        pending.set_exception(error)

    def extract(self, paper: Paper) -> PaperEvidence:
        """Extract one paper, reusing completed or in-progress work safely."""

        document = self._load_document(paper)
        cache_key, cached, pending, is_owner = self._reserve(paper, document)
        if cached is not None:
            return cached

        if not is_owner:
            assert pending is not None
            return pending.result()

        try:
            result = self._extract_uncached(paper, document)
        except PaperExtractionError as full_text_error:
            if document.status == "usable" and self.full_text_client is not None:
                try:
                    with self._cache_lock:
                        self._metrics["full_text_extraction_fallback_requests"] += 1
                    result = self._extract_uncached(
                        paper,
                        document,
                        use_full_text=False,
                        fallback_explanation=(
                            "Full-text structured extraction failed; "
                            + (
                                "abstract fallback succeeded."
                                if paper.abstract
                                else "title metadata fallback succeeded."
                            )
                        ),
                    )
                except BaseException as fallback_error:
                    assert pending is not None
                    self._finish_failure(cache_key, fallback_error, pending)
                    raise fallback_error from full_text_error
            else:
                assert pending is not None
                self._finish_failure(cache_key, full_text_error, pending)
                raise
        except BaseException as exc:
            assert pending is not None
            self._finish_failure(cache_key, exc, pending)
            raise
        assert pending is not None
        self._finish_success(cache_key, result, pending)
        return result

    def extract_many(
        self,
        papers: Sequence[Paper],
        limit: int | None = None,
    ) -> list[PaperEvidence]:
        self.failures = []
        self.coverage_records = []
        with self._cache_lock:
            self._attempt_documents = {}

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        selected = list(papers)[: self.evidence_limit if limit is None else limit]

        if self.full_text_client is None and self.batch_size > 1 and len(selected) > 1:
            results = self._extract_many_batched(selected)
            self._finalize_coverage_records(selected, results)
            return results

        results: list[PaperEvidence] = []

        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(selected) or 1),
            thread_name_prefix="paper-extraction",
        ) as executor:
            futures = [executor.submit(self.extract, paper) for paper in selected]

            for paper, future in zip(selected, futures):
                try:
                    results.append(future.result())
                except PaperExtractionError as exc:
                    self.failures.append(PaperExtractionError(f"{paper.id}: {exc}"))

        self._finalize_coverage_records(selected, results)
        return results

    def _finalize_coverage_records(
        self,
        requested: Sequence[Paper],
        evidence: Sequence[PaperEvidence],
    ) -> None:
        by_id = {record.paper_id: record for record in evidence}
        records: list[PaperCoverageRecord] = []
        for paper in requested:
            record = by_id.get(paper.id)
            document = self._attempt_documents.get(paper.id)
            attempted_section_types: list[SectionType] = []
            attempted_truncated = False
            if document is not None and document.status == "usable":
                _context, attempted_sections, attempted_truncated = (
                    build_extraction_context(
                        document,
                        max_chars=self.max_full_text_context_chars,
                    )
                )
                attempted_section_types = list(dict.fromkeys(
                    section_type
                    for section in attempted_sections
                    for section_type in section.section_types
                ))
            if record is not None and record.coverage is not None:
                coverage = record.coverage
                records.append(PaperCoverageRecord(
                    paper_id=paper.id,
                    title=paper.title,
                    aliases=_paper_aliases(paper),
                    full_text_requested=coverage.full_text_requested,
                    full_text_attempted=coverage.full_text_attempted,
                    final_evidence_level=coverage.source_level,
                    full_text_status=coverage.full_text_status,
                    full_text_extraction_succeeded=coverage.full_text_extraction_succeeded,
                    full_text_source_format=coverage.full_text_source_format,
                    truncated=bool(coverage.truncated or attempted_truncated),
                    inspected_section_types=(
                        attempted_section_types
                        if coverage.full_text_attempted
                        else list(coverage.inspected_section_types)
                    ),
                    fallback_explanation=coverage.fallback_explanation,
                    final_state="success",
                ))
                continue

            status = document.status if document is not None else "not_attempted"
            attempted = bool(
                document
                and (document.source_url or status in {"fetch_failed", "parse_failed", "usable"})
            )
            failure = next(
                (item for item in self.failures if str(item).startswith(f"{paper.id}:")),
                None,
            )
            records.append(PaperCoverageRecord(
                paper_id=paper.id,
                title=paper.title,
                aliases=_paper_aliases(paper),
                full_text_requested=self.full_text_client is not None,
                full_text_attempted=attempted,
                final_evidence_level="none",
                full_text_status=status,
                full_text_extraction_succeeded=False,
                full_text_source_format=document.source_format if document else None,
                truncated=bool(
                    document and document.truncated
                    or attempted_truncated
                ),
                inspected_section_types=attempted_section_types,
                fallback_explanation=_final_failure_explanation(status, bool(paper.abstract)),
                final_state="failure",
                failure_category=_failure_category(failure),
            ))
        self.coverage_records = records

    def _extract_many_batched(
        self,
        selected: Sequence[Paper],
    ) -> list[PaperEvidence]:
        entries: list[PaperEvidence | Future[PaperEvidence]] = []
        owners: list[tuple[Paper, tuple[str, str, str, str], Future[PaperEvidence]]] = []

        for paper in selected:
            cache_key, cached, pending, is_owner = self._reserve(paper)
            if cached is not None:
                entries.append(cached)
            else:
                assert pending is not None
                entries.append(pending)
                if is_owner:
                    owners.append((paper, cache_key, pending))

        batches = self._pack_batches(owners)
        if batches:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(batches)),
                thread_name_prefix="paper-extraction-batch",
            ) as executor:
                futures = [executor.submit(self._run_batch, batch) for batch in batches]
                for future in futures:
                    future.result()

        results: list[PaperEvidence] = []
        for paper, entry in zip(selected, entries):
            try:
                result = entry.result() if isinstance(entry, Future) else entry
                results.append(result)
            except PaperExtractionError as exc:
                self.failures.append(PaperExtractionError(f"{paper.id}: {exc}"))
        return results

    def _run_batch(
        self,
        batch: Sequence[tuple[Paper, tuple[str, str, str, str], Future[PaperEvidence]]],
    ) -> None:
        papers = [item[0] for item in batch]
        try:
            extracted = self._extract_uncached_batch(papers)
        except BaseException:
            extracted = {}

        for paper, cache_key, pending in batch:
            result = extracted.get(cache_key[1])
            if result is not None:
                self._finish_success(cache_key, result, pending)
                continue

            # A failed batch member is retried alone. Other valid members have
            # already been completed and cached, so they are never re-extracted.
            try:
                with self._cache_lock:
                    self._metrics["extraction_fallback_requests"] += 1
                result = self._extract_uncached(paper)
            except BaseException as exc:
                self._finish_failure(cache_key, exc, pending)
            else:
                self._finish_success(cache_key, result, pending)


_LIST_CLAIM_FIELDS = (
    "population_or_setting",
    "method_or_intervention",
    "comparison_or_baseline",
    "data_or_modality",
    "datasets",
    "evaluation_metrics",
    "main_findings",
    "constraints",
    "limitations",
    "future_work",
)


def _normalize_extraction_payload(
    payload: _ExtractionTransportResult | _ExtractionResult,
    paper: Paper,
    *,
    full_text_allowed: bool,
) -> tuple[_ExtractionResult, int]:
    """Normalize impossible optional provenance once, then validate strictly.

    The repair is deliberately narrow: it clears section-only properties from
    title/abstract evidence. If an abstract-only request is mislabeled as
    full-text, it is reassigned only when its verbatim evidence can be located
    in the supplied title or abstract; otherwise that one claim is dropped.
    Values, evidence text, roles, confidence, and paper identity are untouched.
    """

    raw = payload.model_dump(mode="python")
    repaired = 0

    def normalize_claim(claim: dict[str, object]) -> dict[str, object] | None:
        nonlocal repaired
        source = claim.get("source")
        section_fields = ("section_type", "section_heading", "section_id")

        if source == "full_text" and not full_text_allowed:
            evidence_text = claim.get("evidence_text")
            if not isinstance(evidence_text, str):
                repaired += 1
                return None
            normalized_evidence = _normalize(evidence_text)
            if normalized_evidence and normalized_evidence in _normalize(paper.title):
                claim["source"] = "title"
            elif normalized_evidence and normalized_evidence in _normalize(paper.abstract or ""):
                claim["source"] = "abstract"
            else:
                repaired += 1
                return None
            repaired += 1
            source = claim["source"]

        if source in {"title", "abstract"}:
            if any(claim.get(field) is not None for field in section_fields):
                repaired += 1
            for field in section_fields:
                claim[field] = None
        return claim

    for field_name in _LIST_CLAIM_FIELDS:
        value = raw.get(field_name)
        if not isinstance(value, list):
            continue
        normalized_items: list[dict[str, object]] = []
        for item in value:
            if isinstance(item, dict):
                normalized = normalize_claim(item)
                if normalized is not None:
                    normalized_items.append(normalized)
        raw[field_name] = normalized_items

    for field_name in ("research_objective", "sample_size"):
        item = raw.get(field_name)
        if isinstance(item, dict):
            raw[field_name] = normalize_claim(item)

    return _ExtractionResult.model_validate(raw), repaired


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _is_supported(
    item: EvidenceItem,
    paper: Paper,
    document: PaperDocument | None = None,
    inspected_section_ids: set[str] | None = None,
) -> bool:
    """Require evidence_text to exist verbatim in its declared source."""

    if item.source == "title":
        source_text = paper.title
    elif item.source == "abstract":
        source_text = paper.abstract or ""
    else:
        section = document.section(item.section_id or "") if document else None
        source_text = section.text if section is not None else ""
        if (
            section is None
            or inspected_section_ids is not None
            and section.id not in inspected_section_ids
            or item.section_heading != section.heading
            or item.section_type not in section.section_types
        ):
            source_text = ""
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
    document: PaperDocument | None = None,
    inspected_section_ids: set[str] | None = None,
) -> list[EvidenceItem]:
    cleaned: list[EvidenceItem] = []
    seen: set[str] = set()

    for item in items:
        if not _is_supported(item, paper, document, inspected_section_ids):
            continue

        key = canonical_evidence_key(item.canonical_value or item.value)

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
    document: PaperDocument | None = None,
    inspected_section_ids: set[str] | None = None,
) -> EvidenceItem | None:
    if item is None or not _is_supported(item, paper, document, inspected_section_ids):
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
    document: PaperDocument | None = None,
    inspected_section_ids: set[str] | None = None,
) -> list[EvidenceItem]:
    return _clean_items(
        [item for item in items if _is_explicit_comparison(item)],
        paper, document=document, inspected_section_ids=inspected_section_ids,
    )


def _remove_generic_primary_methods(
    items: list[EvidenceItem],
) -> list[EvidenceItem]:
    """Drop obvious implementation details when real focal methods exist."""

    if len(items) < 2:
        return items

    generic = [
        item for item in items if _GENERIC_METHOD_DETAIL_PATTERN.search(item.value)
    ]

    if not generic or len(generic) == len(items):
        return items

    generic_keys = {
        canonical_evidence_key(item.value)
        for item in generic
    }

    return [item for item in items if canonical_evidence_key(item.value) not in generic_keys]


def _clean_methods(
    items: Sequence[_MethodClaim],
    paper: Paper,
    document: PaperDocument | None = None,
    inspected_section_ids: set[str] | None = None,
) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    primary = _remove_generic_primary_methods(
        _clean_items(
            [item for item in items if item.role == "primary"],
            paper, document=document, inspected_section_ids=inspected_section_ids,
        )
    )

    comparisons = _clean_comparisons(
        [item for item in items if item.role == "comparison"],
        paper, document, inspected_section_ids,
    )

    # Supporting claims intentionally disappear from PaperEvidence.methods.
    # They remain implementation details, not focal scientific methods.
    return primary, comparisons


def _clean_future_work(
    items: Sequence[EvidenceItem],
    paper: Paper,
    document: PaperDocument | None = None,
    inspected_section_ids: set[str] | None = None,
) -> list[EvidenceItem]:
    supported = _clean_items(items, paper, document=document, inspected_section_ids=inspected_section_ids)

    return [item for item in supported if _FUTURE_ACTION_PATTERN.search(item.evidence_text)]


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
    document: PaperDocument | None = None,
    inspected_section_ids: set[str] | None = None,
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

    Simply, filters out invalid constraint collisions. For example, 
    
        method     : LoRa, 
        constraint : LoRa  
    
    gets rejected.
    """

    supported = _clean_items(items, paper, document=document, inspected_section_ids=inspected_section_ids)
    scientific_entities = [*methods, *comparisons]

    return [
        item for item in supported if not any(
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
    *,
    document: PaperDocument | None = None,
    inspected_sections: Sequence[PaperSection] = (),
    context_truncated: bool = False,
    full_text_requested: bool = False,
    full_text_used: bool = False,
    fallback_explanation: str | None = None,
) -> PaperEvidence:
    inspected_section_ids = {section.id for section in inspected_sections}
    limitations = [
        LimitationEvidence(
            value=item.value,
            canonical_value=item.canonical_value,
            evidence_text=item.evidence_text,
            source=item.source,
            confidence=item.confidence,
            section_type=item.section_type,
            section_heading=item.section_heading,
            section_id=item.section_id,
        )
        for item in payload.limitations
        if item.author_stated and _is_supported(item, paper, document, inspected_section_ids)
    ]

    methods, role_comparisons = _clean_methods(
        payload.method_or_intervention,
        paper, document, inspected_section_ids,
    )

    explicit_comparisons = _clean_comparisons(
        payload.comparison_or_baseline,
        paper, document, inspected_section_ids,
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
        document=document,
        inspected_section_ids=inspected_section_ids,
    )

    sample_size = _clean_single(
        payload.sample_size,
        paper,
        document,
        inspected_section_ids,
    )

    if sample_size is None:
        sample_size = _fallback_sample_size(paper)

    result = PaperEvidence(
        paper_id=paper.id,
        title=paper.title,
        study_type=payload.study_type,
        research_objective=_clean_single(
            payload.research_objective,
            paper, document, inspected_section_ids,
        ),
        population_or_setting=_clean_items(
            payload.population_or_setting,
            paper, document=document, inspected_section_ids=inspected_section_ids,
        ),
        method_or_intervention=methods,
        comparison_or_baseline=comparisons,
        data_or_modality=_clean_items(
            payload.data_or_modality,
            paper, document=document, inspected_section_ids=inspected_section_ids,
        ),
        datasets=_clean_items(
            payload.datasets,
            paper, document=document, inspected_section_ids=inspected_section_ids,
            drop_generic_datasets=True,
        ),
        sample_size=sample_size,
        evaluation_metrics=_clean_items(
            payload.evaluation_metrics,
            paper, document=document, inspected_section_ids=inspected_section_ids,
        ),
        main_findings=_clean_items(
            payload.main_findings,
            paper, document=document, inspected_section_ids=inspected_section_ids,
        ),
        constraints=constraints,
        limitations=limitations,
        future_work=_clean_future_work(
            payload.future_work,
            paper, document, inspected_section_ids,
        ),
        extraction_confidence=payload.extraction_confidence,
    )
    evidence_items = [
        item
        for field_name in (
            "research_objective", "population_or_setting", "method_or_intervention",
            "comparison_or_baseline", "data_or_modality", "datasets", "sample_size",
            "evaluation_metrics", "main_findings", "constraints", "limitations", "future_work",
        )
        for item in (
            getattr(result, field_name)
            if isinstance(getattr(result, field_name), list)
            else [getattr(result, field_name)]
        )
        if item is not None
    ]
    has_full_text_evidence = any(item.source == "full_text" for item in evidence_items)
    if has_full_text_evidence:
        source_level = "full_text"
    elif full_text_requested and paper.abstract:
        source_level = "abstract_fallback"
    elif paper.abstract:
        source_level = "abstract"
    else:
        source_level = "metadata_only"
    attempted = bool(
        full_text_requested
        and document
        and (
            document.source_url
            or document.status in {"fetch_failed", "parse_failed", "usable"}
        )
    )
    if source_level == "abstract_fallback" and fallback_explanation is None:
        fallback_explanation = (
            "Full text was inspected, but validated evidence came from the abstract."
        )
    result.coverage = ExtractionCoverage(
        source_level=source_level,
        full_text_status=document.status if document else "not_attempted",
        full_text_requested=full_text_requested,
        full_text_attempted=attempted,
        # A successful provider call is not a successful full-text analysis
        # unless at least one exactly validated full-text claim survives.
        full_text_extraction_succeeded=has_full_text_evidence,
        full_text_source_format=document.source_format if document and attempted else None,
        inspected_section_types=list(dict.fromkeys(
            role for section in inspected_sections for role in section.section_types
        )),
        structure_available=bool(full_text_used and document and document.structure_available),
        truncated=bool(
            context_truncated
            or full_text_used and document and document.truncated
            or any(section.truncated for section in inspected_sections)
        ),
        fallback_explanation=fallback_explanation,
        notices=list(document.notices) if document else [],
    )
    return result


def _fallback_explanation(document: PaperDocument, paper: Paper) -> str | None:
    if document.status == "usable":
        return None
    fallback = "abstract" if paper.abstract else "title metadata"
    if document.status == "fetch_failed":
        return f"Full-text fetch failed; {fallback} fallback succeeded."
    if document.status == "parse_failed":
        return f"Full-text parsing failed; {fallback} fallback succeeded."
    if document.status == "unavailable":
        return f"Open full text was unavailable; {fallback} fallback succeeded."
    return None


def _final_failure_explanation(status: str, has_abstract: bool) -> str:
    source = "abstract" if has_abstract else "title metadata"
    if status == "fetch_failed":
        return f"Full-text fetch failed and the {source} fallback did not produce validated evidence."
    if status == "parse_failed":
        return f"Full-text parsing failed and the {source} fallback did not produce validated evidence."
    if status == "usable":
        return f"Full text was inspected, but neither it nor the {source} fallback produced validated evidence."
    if status == "unavailable":
        return f"Open full text was unavailable and the {source} fallback did not produce validated evidence."
    return "Structured extraction did not produce validated evidence."


def _failure_category(error: object | None) -> str:
    message = str(error or "").casefold()
    if (
        "invalid evidence response" in message
        or "validation" in message
        or "no parsed evidence payload" in message
    ):
        return "model_schema_evidence_validation"
    return "provider_failure"


def _paper_aliases(paper: Paper) -> list[str]:
    return list(dict.fromkeys([
        *paper.openalex_aliases,
        *paper.doi_aliases,
    ]))
