"""Validated runtime configuration for the Research GAP pipeline."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def openai_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY", "").strip() or None


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL


def openai_extraction_model() -> str:
    return os.getenv("OPENAI_EXTRACTION_MODEL", "").strip() or openai_model()


def openai_embedding_model() -> str:
    return os.getenv("OPENAI_EMBEDDING_MODEL", "").strip() or DEFAULT_EMBEDDING_MODEL


def cache_dir() -> Path:
    """Resolve the cache when a component is built, not when its module is imported."""
    value = os.getenv("RESEARCH_GAP_CACHE_DIR", "").strip()
    return Path(value).expanduser() if value else PROJECT_ROOT / "data" / "cache"


def database_path() -> Path:
    """Return the durable analysis-history database path."""
    value = os.getenv("RESEARCH_GAP_DATABASE_PATH", "").strip()
    return Path(value).expanduser() if value else PROJECT_ROOT / "data" / "research_gap.sqlite3"


class ConfigurationError(ValueError):
    """Raised when environment configuration is malformed or out of bounds."""


@dataclass(frozen=True)
class OpenAlexSettings:
    api_key: str | None
    mailto: str | None
    per_route_limit: int
    max_candidates: int
    max_workers: int
    timeout_seconds: float
    max_retries: int
    retrieval_cache_ttl_seconds: float


@dataclass(frozen=True)
class RankingSettings:
    embedding_model: str
    embedding_batch_size: int
    lexical_weight: float
    semantic_weight: float
    constraint_weight: float
    semantic_fallback: Literal["lexical", "error"]


@dataclass(frozen=True)
class FullTextSettings:
    timeout_seconds: float
    max_bytes: int
    max_document_chars: int
    max_section_chars: int
    max_chunk_chars: int
    max_context_chars: int
    max_redirects: int
    negative_cache_ttl_seconds: float


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    extraction_model: str
    openalex: OpenAlexSettings
    ranking: RankingSettings
    full_text: FullTextSettings
    evidence_limit: int
    extraction_workers: int
    extraction_batch_size: int
    cache_directory: Path = field(default_factory=cache_dir)
    analysis_database_path: Path = field(default_factory=database_path)
    max_analysis_workers: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        def integer(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
            try:
                value = int(os.getenv(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be an integer") from exc
            if value < minimum:
                requirement = "positive" if minimum == 1 else f"at least {minimum}"
                raise ConfigurationError(f"{name} must be {requirement}")
            if maximum is not None and value > maximum:
                raise ConfigurationError(f"{name} must not exceed {maximum}")
            return value

        def number(
            name: str,
            default: float,
            *,
            positive: bool = False,
            maximum: float | None = None,
        ) -> float:
            try:
                value = float(os.getenv(name, str(default)))
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be a number") from exc
            if not math.isfinite(value):
                raise ConfigurationError(f"{name} must be finite")
            if value < 0 or (positive and value == 0):
                requirement = "positive" if positive else "non-negative"
                raise ConfigurationError(f"{name} must be {requirement}")
            if maximum is not None and value > maximum:
                raise ConfigurationError(f"{name} must not exceed {maximum:g}")
            return value

        lexical = number("RESEARCH_GAP_LEXICAL_WEIGHT", 0.4)
        semantic = number("RESEARCH_GAP_SEMANTIC_WEIGHT", 0.6)
        if lexical == 0 and semantic == 0:
            raise ConfigurationError("weights cannot both be zero")
        if not math.isfinite(lexical + semantic):
            raise ConfigurationError("combined ranking weights must be finite")
        constraint_weight = number("RESEARCH_GAP_CONSTRAINT_WEIGHT", 0.15)
        if constraint_weight > 1:
            raise ConfigurationError("RESEARCH_GAP_CONSTRAINT_WEIGHT must be between 0 and 1")
        fallback = os.getenv("RESEARCH_GAP_SEMANTIC_FALLBACK", "lexical").strip()
        if fallback not in {"lexical", "error"}:
            raise ConfigurationError("RESEARCH_GAP_SEMANTIC_FALLBACK must be 'lexical' or 'error'")

        return cls(
            openai_api_key=openai_api_key(), openai_model=openai_model(),
            extraction_model=openai_extraction_model(),
            openalex=OpenAlexSettings(
                api_key=os.getenv("OPENALEX_API_KEY") or None,
                mailto=os.getenv("OPENALEX_MAILTO") or None,
                per_route_limit=integer("OPENALEX_CANDIDATE_LIMIT", 20, maximum=100),
                max_candidates=integer("RESEARCH_GAP_MAX_CANDIDATES", 100, maximum=500),
                max_workers=integer("RESEARCH_GAP_RETRIEVAL_WORKERS", 4, maximum=16),
                timeout_seconds=number("OPENALEX_TIMEOUT_SECONDS", 20.0, positive=True),
                max_retries=integer("OPENALEX_MAX_RETRIES", 2, minimum=0),
                retrieval_cache_ttl_seconds=number(
                    "RESEARCH_GAP_RETRIEVAL_CACHE_TTL_SECONDS", 21600, positive=True,
                ),
            ),
            ranking=RankingSettings(
                embedding_model=openai_embedding_model(),
                embedding_batch_size=integer("RESEARCH_GAP_EMBEDDING_BATCH_SIZE", 100),
                lexical_weight=lexical, semantic_weight=semantic,
                constraint_weight=constraint_weight, semantic_fallback=fallback,
            ),
            full_text=FullTextSettings(
                timeout_seconds=number(
                    "RESEARCH_GAP_FULL_TEXT_TIMEOUT_SECONDS", 12.0,
                    positive=True, maximum=120,
                ),
                max_bytes=integer("RESEARCH_GAP_FULL_TEXT_MAX_BYTES", 8_000_000, maximum=25_000_000),
                max_document_chars=integer("RESEARCH_GAP_FULL_TEXT_MAX_DOCUMENT_CHARS", 120_000, maximum=500_000),
                max_section_chars=integer("RESEARCH_GAP_FULL_TEXT_MAX_SECTION_CHARS", 20_000, maximum=100_000),
                max_chunk_chars=integer("RESEARCH_GAP_FULL_TEXT_MAX_CHUNK_CHARS", 8_000, maximum=30_000),
                max_context_chars=integer("RESEARCH_GAP_FULL_TEXT_MAX_CONTEXT_CHARS", 30_000, maximum=100_000),
                max_redirects=integer("RESEARCH_GAP_FULL_TEXT_MAX_REDIRECTS", 3, minimum=0, maximum=10),
                negative_cache_ttl_seconds=number(
                    "RESEARCH_GAP_FULL_TEXT_NEGATIVE_CACHE_TTL_SECONDS", 3600,
                    positive=True, maximum=604800,
                ),
            ),
            evidence_limit=integer("RESEARCH_GAP_EVIDENCE_LIMIT", 10, minimum=0),
            extraction_workers=integer("RESEARCH_GAP_EXTRACTION_WORKERS", 4),
            extraction_batch_size=integer("RESEARCH_GAP_EXTRACTION_BATCH_SIZE", 3),
            cache_directory=cache_dir(),
            analysis_database_path=database_path(),
            max_analysis_workers=integer("RESEARCH_GAP_MAX_ANALYSIS_WORKERS", 2, maximum=8),
        )
