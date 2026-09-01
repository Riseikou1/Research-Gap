"""Structured evidence extraction for ranked papers."""

from .evidence import (
    EvidenceItem,
    LimitationEvidence,
    PaperEvidence,
    StudyType,
    canonical_evidence_key,
)
from .paper_extractor import PaperExtractionError, PaperExtractor

__all__ = [
    "EvidenceItem",
    "LimitationEvidence",
    "PaperEvidence",
    "StudyType",
    "canonical_evidence_key",
    "PaperExtractionError",
    "PaperExtractor",
]
