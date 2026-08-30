"""OpenAI Structured Outputs backend for paper evidence extraction."""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import openai_api_key, openai_extraction_model
from src.models.paper import Paper

from .evidence import EvidenceItem, LimitationEvidence, PaperEvidence


class PaperExtractionError(RuntimeError):
    """Raised when a paper cannot be converted into structured evidence."""


class _Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    value: str = Field(min_length=1)
    evidence_text: str | None = None
    source: str
    confidence: float = Field(ge=0.0, le=1.0)


class _Limitation(_Claim):
    author_stated: bool


class _ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    research_objective: _Claim | None = None
    population_or_setting: list[_Claim] = Field(default_factory=list)
    method_or_intervention: list[_Claim] = Field(default_factory=list)
    comparison_or_baseline: list[_Claim] = Field(default_factory=list)
    datasets: list[_Claim] = Field(default_factory=list)
    sample_size: _Claim | None = None
    evaluation_metrics: list[_Claim] = Field(default_factory=list)
    main_findings: list[_Claim] = Field(default_factory=list)
    limitations: list[_Limitation] = Field(default_factory=list)
    future_work: list[_Claim] = Field(default_factory=list)
    extraction_confidence: float = Field(ge=0.0, le=1.0)


_INSTRUCTIONS = """
Extract structured evidence only from the supplied paper title and abstract.
Do not use outside knowledge or infer missing experimental details. Do not
invent datasets, sample sizes, baselines, metrics, findings, limitations, or
future work. Return null or an empty list when information is absent.

Every claim must include supporting text copied from the title or abstract and
source must be exactly title or abstract. A limitation may be author_stated
only when the text explicitly presents it as a limitation, weakness,
constraint, shortcoming, or equivalent. Generic criticism is not a limitation.
Future work must be an explicit direction proposed by the authors. Sample size
must be explicit. Keep claims concise and preserve stated numerical results.
""".strip()


class PaperExtractor:
    """Extract evidence for papers using OpenAI Structured Outputs."""

    def __init__(self, *, client: Any | None = None, api_key: str | None = None,
                 model: str | None = None, max_output_tokens: int = 2400,
                 evidence_limit: int = 10) -> None:
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
        source = f"Title:\n{title}\n\nAbstract:\n{abstract or '[not available]'}"
        try:
            response = self.client.responses.parse(
                model=self.model, reasoning={"effort": "low"}, store=False,
                max_output_tokens=self.max_output_tokens, instructions=_INSTRUCTIONS,
                input=source, text_format=_ExtractionResult,
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

    def extract_many(self, papers: Sequence[Paper], limit: int | None = None) -> list[PaperEvidence]:
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


def _claim(value: _Claim) -> EvidenceItem:
    return EvidenceItem(value=value.value, evidence_text=value.evidence_text,
                        source=value.source, confidence=value.confidence)


def _to_evidence(paper: Paper, payload: _ExtractionResult) -> PaperEvidence:
    def claims(values: list[_Claim]) -> list[EvidenceItem]:
        return [_claim(item) for item in values if item.source in ("title", "abstract")]

    limitations = [
        LimitationEvidence(**_claim(item).model_dump(), author_stated=True)
        for item in payload.limitations
        if item.author_stated and item.source in ("title", "abstract")
    ]
    return PaperEvidence(
        paper_id=paper.id, title=paper.title,
        research_objective=_claim(payload.research_objective) if payload.research_objective and payload.research_objective.source in ("title", "abstract") else None,
        population_or_setting=claims(payload.population_or_setting),
        method_or_intervention=claims(payload.method_or_intervention),
        comparison_or_baseline=claims(payload.comparison_or_baseline),
        datasets=claims(payload.datasets), sample_size=_claim(payload.sample_size) if payload.sample_size and payload.sample_size.source in ("title", "abstract") else None,
        evaluation_metrics=claims(payload.evaluation_metrics), main_findings=claims(payload.main_findings),
        limitations=limitations, future_work=claims(payload.future_work),
        extraction_confidence=payload.extraction_confidence,
    )
