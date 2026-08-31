"""OpenAI Structured Outputs backend for paper evidence extraction."""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import openai_api_key, openai_extraction_model
from src.models.paper import Paper

from .evidence import EvidenceItem, LimitationEvidence, PaperEvidence


class PaperExtractionError(RuntimeError):
    """Raised when a paper cannot be converted into structured evidence."""


class _LimitationClaim(EvidenceItem):
    author_stated: bool


class _ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    research_objective: EvidenceItem | None = None
    population_or_setting: list[EvidenceItem] = Field(default_factory=list)
    method_or_intervention: list[EvidenceItem] = Field(default_factory=list)
    comparison_or_baseline: list[EvidenceItem] = Field(default_factory=list)
    datasets: list[EvidenceItem] = Field(default_factory=list)
    sample_size: EvidenceItem | None = None
    evaluation_metrics: list[EvidenceItem] = Field(default_factory=list)
    main_findings: list[EvidenceItem] = Field(default_factory=list)
    limitations: list[_LimitationClaim] = Field(default_factory=list)
    future_work: list[EvidenceItem] = Field(default_factory=list)
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
        
        source = f"Title:\n{title}"
        if abstract:
            source += f"\n\nAbstract:\n{abstract}"        
            
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
    

def _to_evidence(paper: Paper, payload: _ExtractionResult) -> PaperEvidence:
    limitations = [
        LimitationEvidence(
            value=item.value,
            evidence_text=item.evidence_text,
            source=item.source,
            confidence=item.confidence,
        ) for item in payload.limitations if item.author_stated
    ]

    return PaperEvidence(
        paper_id=paper.id,
        title=paper.title,
        research_objective=payload.research_objective,
        population_or_setting=payload.population_or_setting,
        method_or_intervention=payload.method_or_intervention,
        comparison_or_baseline=payload.comparison_or_baseline,
        datasets=payload.datasets,
        sample_size=payload.sample_size,
        evaluation_metrics=payload.evaluation_metrics,
        main_findings=payload.main_findings,
        limitations=limitations,
        future_work=payload.future_work,
        extraction_confidence=payload.extraction_confidence,
    )
