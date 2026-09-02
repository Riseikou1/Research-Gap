"""OpenAI Structured Outputs backend for paper evidence extraction."""

from __future__ import annotations

import os
import logging
import re
from time import perf_counter
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import openai_api_key, openai_extraction_model
from src.config import CACHE_DIR
from src.models.paper import Paper

from .evidence import EvidenceItem, LimitationEvidence, PaperEvidence, StudyType, canonical_evidence_key
from .store import EvidenceStore


LOGGER = logging.getLogger(__name__)


# Increment when the structured extraction contract or its compatibility
# assumptions change. Old cache rows remain harmless misses after a bump.
EVIDENCE_SCHEMA_VERSION = 4


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


class _BatchPaperExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    paper_id: str = Field(min_length=1)
    evidence: _ExtractionResult


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


_BATCH_INSTRUCTIONS = _INSTRUCTIONS + """

The input contains several papers. Return exactly one `papers` item for each
supplied paper identifier when possible. The paper_id must be copied exactly
from the input. Never use evidence from one paper in another paper's item.
If a paper cannot be extracted reliably, omit only that paper so it can be
retried individually; valid sibling items must still be returned.
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
        batch_size: int = 1,
        max_batch_input_chars: int = 24000,
    ) -> None:
        if (
            max_output_tokens <= 0
            or evidence_limit < 0
            or max_workers <= 0
            or batch_size <= 0
            or max_batch_input_chars <= 0
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
        self.failures: list[PaperExtractionError] = []
        self._cache_lock = RLock()
        self._cache: dict[tuple[str, str, str, str], PaperEvidence] = {}
        self._inflight: dict[tuple[str, str, str, str], Future[PaperEvidence]] = {}
        self._metrics: dict[str, int] = {
            "evidence_requested": 0,
            "memory_cache_hits": 0,
            "persistent_cache_hits": 0,
            "new_evidence_extractions": 0,
            "openai_extraction_requests": 0,
        }
        self._timings: dict[str, float] = {
            "initial_evidence_extraction_api_wait": 0.0,
        }

        if client is not None:
            self.client = client
            # Unit-test fakes stay isolated by default. Supplying cache_path
            # explicitly enables the same persistent behavior for them.
            self.evidence_store = EvidenceStore(cache_path)
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
            cache_path if cache_path is not None else CACHE_DIR / "research_gap.sqlite3"
        )

    @staticmethod
    def _cache_key(
        paper: Paper,
        model: str,
    ) -> tuple[str, str, str, str]:
        return (
            paper.id,
            EvidenceStore.content_hash(paper),
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

    def _extract_uncached(self, paper: Paper) -> PaperEvidence:
        title = paper.title.strip()
        abstract = paper.abstract.strip() if paper.abstract else None

        if not title:
            raise PaperExtractionError("Paper title is required for evidence extraction.")

        source = f"Title:\n{title}"
        if abstract:
            source += f"\n\nAbstract:\n{abstract}"

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
                instructions=_INSTRUCTIONS,
                input=source,
                text_format=_ExtractionResult,
            )
            with self._cache_lock:
                self._timings["initial_evidence_extraction_api_wait"] += (
                    perf_counter() - started
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
            raise PaperExtractionError(f"Invalid batch evidence response: {exc}") from exc
        except Exception as exc:
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
                result[content_hash] = _to_evidence(paper, item.evidence)
            except (PaperExtractionError, TypeError, ValueError):
                # Only this member is invalid; its sibling results remain
                # eligible for completion and caching.
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
    ) -> tuple[tuple[str, str, str, str], PaperEvidence | None, Future[PaperEvidence] | None, bool]:
        cache_key = self._cache_key(paper, self.model)
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

        cache_key, cached, pending, is_owner = self._reserve(paper)
        if cached is not None:
            return cached

        if not is_owner:
            assert pending is not None
            return pending.result()

        try:
            result = self._extract_uncached(paper)
        except BaseException as exc:
            assert pending is not None
            self._finish_failure(cache_key, exc, pending)
            raise
        else:
            assert pending is not None
            self._finish_success(cache_key, result, pending)
            return result

    def extract_many(
        self,
        papers: Sequence[Paper],
        limit: int | None = None,
    ) -> list[PaperEvidence]:
        self.failures = []

        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        selected = list(papers)[: self.evidence_limit if limit is None else limit]

        if self.batch_size > 1 and len(selected) > 1:
            return self._extract_many_batched(selected)

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

        return results

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
                result = self._extract_uncached(paper)
            except BaseException as exc:
                self._finish_failure(cache_key, exc, pending)
            else:
                self._finish_success(cache_key, result, pending)


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
            canonical_value=item.canonical_value,
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
        data_or_modality=_clean_items(
            payload.data_or_modality,
            paper,
        ),
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
