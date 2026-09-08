"""Structured evidence extraction for ranked papers."""

from .evidence import (
    EvidenceItem,
    ExtractionCoverage,
    LimitationEvidence,
    PaperEvidence,
    StudyType,
    canonical_evidence_key,
)
from .paper_extractor import PaperExtractionError, PaperExtractor
from .document import PaperDocument, PaperSection
from .full_text import FullTextClient
from .store import EvidenceStore

__all__ = [
    "EvidenceItem",
    "ExtractionCoverage",
    "PaperDocument",
    "PaperSection",
    "FullTextClient",
    "LimitationEvidence",
    "PaperEvidence",
    "StudyType",
    "canonical_evidence_key",
    "PaperExtractionError",
    "PaperExtractor",
    "EvidenceStore",
]
