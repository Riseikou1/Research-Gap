"""Cross-paper literature analysis and research-gap verification."""

from .clustering import LandscapeAnalyzer
from .comparison import method_family, normalize_feature, to_paper_features
from .gap_candidates import (
    GapCandidateGenerator,
    candidate_priority,
    consolidate_candidates,
    is_concrete_entity,
    prune_redundant_candidates,
    validate_evidence_semantics,
)
from .models import (
    GapAssessmentLabel,
    GapCandidate,
    GapCategory,
    GapEvidence,
    GapPattern,
    GapVerification,
    IdeaAssessment,
    LandscapeBasis,
    VerificationFailure,
    VerificationQuery,
)
from .verification import (
    GapVerifier,
    build_idea_verification_queries,
    build_verification_queries,
)


__all__ = [
    "GapAssessmentLabel",
    "GapCandidate",
    "GapCandidateGenerator",
    "candidate_priority",
    "GapCategory",
    "GapEvidence",
    "GapPattern",
    "GapVerification",
    "GapVerifier",
    "IdeaAssessment",
    "LandscapeAnalyzer",
    "LandscapeBasis",
    "VerificationFailure",
    "VerificationQuery",
    "build_idea_verification_queries",
    "build_verification_queries",
    "consolidate_candidates",
    "is_concrete_entity",
    "method_family",
    "normalize_feature",
    "to_paper_features",
    "prune_redundant_candidates",
    "validate_evidence_semantics",
]
