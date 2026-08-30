"""Configuration for the Research GAP pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# ------------------------------------------------------------------
# API credentials
# ------------------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO")


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6-luna",
)

OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)


def openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY") or None


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL") or "gpt-5.6-luna"


def openai_extraction_model() -> str:
    return os.getenv("OPENAI_EXTRACTION_MODEL") or openai_model()


class ConfigurationError(ValueError):
    """Raised when environment configuration cannot be parsed."""


@dataclass(frozen=True)
class OpenAlexSettings:
    api_key: str | None
    mailto: str | None
    per_route_limit: int
    max_candidates: int
    max_workers: int
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True)
class RankingSettings:
    embedding_model: str
    embedding_batch_size: int
    lexical_weight: float
    semantic_weight: float
    semantic_fallback: str


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    extraction_model: str
    openalex: OpenAlexSettings
    ranking: RankingSettings
    evidence_limit: int

    @classmethod
    def from_env(cls) -> "Settings":
        def integer(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be an integer") from exc

        def number(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be a number") from exc

        lexical = number("RESEARCH_GAP_LEXICAL_WEIGHT", 0.4)
        semantic = number("RESEARCH_GAP_SEMANTIC_WEIGHT", 0.6)
        if lexical == 0 and semantic == 0:
            raise ConfigurationError("weights cannot both be zero")
        return cls(
            openai_api_key=openai_api_key(), openai_model=openai_model(),
            extraction_model=openai_extraction_model(),
            openalex=OpenAlexSettings(
                api_key=os.getenv("OPENALEX_API_KEY") or None,
                mailto=os.getenv("OPENALEX_MAILTO") or None,
                per_route_limit=integer("OPENALEX_CANDIDATE_LIMIT", 20),
                max_candidates=integer("RESEARCH_GAP_MAX_CANDIDATES", 100),
                max_workers=integer("RESEARCH_GAP_RETRIEVAL_WORKERS", 4),
                timeout_seconds=number("OPENALEX_TIMEOUT_SECONDS", 20.0),
                max_retries=integer("OPENALEX_MAX_RETRIES", 2),
            ),
            ranking=RankingSettings(
                embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                embedding_batch_size=integer("RESEARCH_GAP_EMBEDDING_BATCH_SIZE", 100),
                lexical_weight=lexical, semantic_weight=semantic,
                semantic_fallback=os.getenv("RESEARCH_GAP_SEMANTIC_FALLBACK", "lexical"),
            ),
            evidence_limit=integer("RESEARCH_GAP_EVIDENCE_LIMIT", 10),
        )


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------

PER_ROUTE_LIMIT = 20
MAX_CANDIDATES = 100
RETRIEVAL_WORKERS = 4

OPENALEX_TIMEOUT_SECONDS = 20.0
OPENALEX_MAX_RETRIES = 2


# ------------------------------------------------------------------
# Ranking
# ------------------------------------------------------------------

LEXICAL_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
EMBEDDING_BATCH_SIZE = 100

SEMANTIC_FALLBACK = "lexical"


# ------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
