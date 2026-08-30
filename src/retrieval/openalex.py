"""OpenAlex Works API retriever."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import date, datetime, timezone
from typing import Any, BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.models.paper import Paper, RetrievalProvenance
from src.models.query import RetrievalMode
from src.retrieval.base import (
    RetrievalConfigurationError,
    RetrievalError,
    RetrievalRequest,
)


LOGGER = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# OpenAlex semantic search currently returns at most 50 results.
OPENALEX_SEMANTIC_MAX_RESULTS = 50

SELECT_FIELDS = (
    "id,display_name,title,abstract_inverted_index,authorships,"
    "publication_year,publication_date,doi,primary_location,"
    "cited_by_count,relevance_score"
)


class OpenAlexError(RetrievalError):
    """Raised when OpenAlex cannot return a usable response."""


class OpenAlexRetriever:
    """Retrieve and normalize papers from the OpenAlex Works API."""

    provider_name = "openalex"

    def __init__(
        self,
        timeout: float = 20.0,
        mailto: str | None = None,
        api_key: str | None = None,
        *,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        opener: Callable[..., AbstractContextManager[BinaryIO]] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] | None = None,
        enable_memory_cache: bool = True,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")

        self.timeout = timeout
        self.mailto = mailto
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

        self._opener = opener or urlopen
        self._sleeper = sleeper
        self._clock = clock or (lambda: datetime.now(timezone.utc))

        self._enable_memory_cache = enable_memory_cache
        self._cache: dict[
            tuple[tuple[str, str], ...],
            Mapping[str, Any],
        ] = {}

    def search(self, request: RetrievalRequest) -> list[Paper]:
        """Retrieve papers for one OpenAlex retrieval request."""

        if (
            request.mode is RetrievalMode.SEMANTIC
            and not self.api_key
        ):
            raise RetrievalConfigurationError(
                "OpenAlex semantic search requires OPENALEX_API_KEY"
            )

        started = time.monotonic()
        retrieved_at = self._clock().astimezone(timezone.utc)

        limit = (
            min(
                request.limit,
                OPENALEX_SEMANTIC_MAX_RESULTS,
            )
            if request.mode is RetrievalMode.SEMANTIC
            else request.limit
        )

        payload = self._request(self._build_params(request, per_page=limit))

        raw_results = payload.get("results")

        if not isinstance(raw_results, list):
            raise OpenAlexError(
                "OpenAlex response did not contain a results list"
            )

        papers: list[Paper] = []

        for provider_rank, raw_work in enumerate(
            raw_results,
            start=1,
        ):
            if not isinstance(raw_work, Mapping):
                LOGGER.warning(
                    "openalex malformed work skipped "
                    "mode=%s query=%r",
                    request.mode.value,
                    request.query.text,
                )
                continue

            paper = _parse_work(raw_work)

            paper.provenance = [
                RetrievalProvenance(
                    query=request.query,
                    provider=self.provider_name,
                    mode=request.mode,
                    retrieved_at=retrieved_at,
                    provider_rank=provider_rank,
                    provider_score=_optional_float(
                        raw_work.get("relevance_score")
                    ),
                )
            ]

            papers.append(paper)

        LOGGER.info(
            "retrieval provider=openalex mode=%s query=%r "
            "result_count=%d latency_ms=%.1f",
            request.mode.value,
            request.query.text,
            len(papers),
            (time.monotonic() - started) * 1000,
        )

        return papers[:limit]

    def _build_params(self, request: RetrievalRequest, *, per_page: int) -> dict[str, str | int]:
        """Translate a retrieval request into OpenAlex API parameters."""

        params: dict[str, str | int] = {
            "page": 1,
            "per_page": per_page,
            "select": SELECT_FIELDS,
        }

        if request.mode is RetrievalMode.BROAD_LEXICAL:
            params["search"] = request.query.text

        elif request.mode is RetrievalMode.TITLE_ABSTRACT:
            scoped_query = " ".join(
                request.query.text
                .replace(",", " ")
                .split()
            )

            params["filter"] = (
                f"title_and_abstract.search:{scoped_query}"
            )

        elif request.mode is RetrievalMode.SEMANTIC:
            params["search.semantic"] = request.query.text

        else:  # pragma: no cover
            raise ValueError(
                f"unsupported retrieval mode: {request.mode}"
            )

        if self.api_key:
            params["api_key"] = self.api_key

        if self.mailto:
            params["mailto"] = self.mailto

        return params

    def _request(self, params: Mapping[str, str | int]) -> Mapping[str, Any]:
        """Execute one OpenAlex request with retries and memory caching."""

        cache_key = tuple(sorted((key, str(value)) for key, value in params.items()))

        if (self._enable_memory_cache and cache_key in self._cache):
            return self._cache[cache_key]

        url = (
            f"{OPENALEX_WORKS_URL}?"
            f"{urlencode(params)}"
        )

        request = Request(url, headers={"User-Agent": "research-gap/0.3"})

        for attempt in range(
            self.max_retries + 1
        ):
            try:
                with self._opener(
                    request,
                    timeout=self.timeout,
                ) as response:
                    payload = json.load(response)

                if not isinstance(payload, Mapping):
                    raise OpenAlexError(
                        "OpenAlex returned a non-object JSON payload"
                    )

                if self._enable_memory_cache:
                    self._cache[cache_key] = payload

                return payload

            except HTTPError as exc:
                retryable = (exc.code == 429 or 500 <= exc.code < 600)

                if (retryable and attempt < self.max_retries):
                    delay = _retry_delay(exc, self.backoff_seconds * (2**attempt))

                    LOGGER.warning(
                        "openalex retry status=%d "
                        "attempt=%d delay=%.2f",
                        exc.code,
                        attempt + 1,
                        delay,
                    )

                    self._sleeper(delay)
                    continue

                raise OpenAlexError(f"HTTP {exc.code}: {exc.reason}") from exc

            except (URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    delay = (self.backoff_seconds * (2**attempt))

                    LOGGER.warning(
                        "openalex network retry "
                        "attempt=%d delay=%.2f",
                        attempt + 1,
                        delay,
                    )

                    self._sleeper(delay)
                    continue

                reason = getattr(exc, "reason", exc)

                raise OpenAlexError(f"network error: {reason}") from exc

            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise OpenAlexError("OpenAlex returned invalid JSON") from exc

        raise AssertionError("retry loop exhausted")


def reconstruct_abstract(
    inverted_index: Mapping[str, Sequence[int]] | None,
) -> str | None:
    """Convert OpenAlex's inverted abstract index into plain text."""

    if not inverted_index:
        return None

    positioned_words: list[tuple[int, str]] = []

    for word, positions in inverted_index.items():
        if not isinstance(word, str):
            continue

        if not isinstance(positions, Sequence):
            continue

        for position in positions:
            if (isinstance(position, int)and not isinstance(position, bool)):
                positioned_words.append((position, word))

    positioned_words.sort(key=lambda item: (item[0],item[1]))
    return " ".join(word for _, word in positioned_words) or None


def _parse_work(
    work: Mapping[str, Any],
) -> Paper:
    """Normalize one raw OpenAlex work into a Paper."""

    openalex_id = _optional_string(work.get("id"))
    doi = _optional_string(work.get("doi"))
    year = _optional_year(work.get("publication_year"))
    primary_location = work.get("primary_location")
    source_value = location.get("source")
    abstract_value = work.get("abstract_inverted_index")

    title = (_optional_string(work.get("display_name")) or _optional_string(work.get("title")) or "Untitled")
    location = (primary_location if isinstance(primary_location, Mapping) else {})
    source = (source_value if isinstance(source_value, Mapping) else {})
    abstract_index = (abstract_value if isinstance(abstract_value, Mapping) else None)

    return Paper(
        id=_paper_id(openalex_id, doi, title, year,),
        title=title,
        abstract=reconstruct_abstract(abstract_index),
        authors=_authors(work),
        publication_year=year,
        publication_date=_optional_date(work.get("publication_date")),
        doi=doi,
        openalex_id=openalex_id,
        citation_count=max(_optional_int(work.get("cited_by_count")) or 0, 0),
        source=_optional_string(source.get("display_name")),
        url=(_optional_string(location.get("landing_page_url")) or openalex_id),
    )


def _authors(work: Mapping[str, Any]) -> list[str]:
    """Extract author display names from an OpenAlex work."""

    authorships = work.get("authorships")

    if not isinstance(authorships, list):
        return []

    names: list[str] = []

    for authorship in authorships:
        if not isinstance(authorship, Mapping) :
            continue

        author_value = authorship.get("author")

        author = (author_value if isinstance(author_value, Mapping) else {})

        name = _optional_string(author.get("display_name"))

        if name:
            names.append(name)

    return names


def _paper_id(
    openalex_id: str | None,
    doi: str | None,
    title: str,
    year: int | None,
) -> str:
    """Build a stable identifier for a paper."""

    if openalex_id:
        return openalex_id

    if doi:
        return doi

    digest = hashlib.sha256(f"{title.casefold()}|{year or ''}".encode("utf-8")).hexdigest()[:20]

    return (f"openalex:synthetic:{digest}")


def _optional_string(
    value: object,
) -> str | None:
    if not isinstance(value, str) :
        return None

    normalized = " ".join(value.split())
    return normalized or None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) :
        return None

    if isinstance(value, int) :
        return value

    return None


def _optional_year(value: object) -> int | None:
    year = _optional_int(value)

    if (year is not None and 1000 <= year <= 3000) :
        return year

    return None


def _optional_float(value: object) -> float | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))) :
        return None

    return float(value)


def _optional_date(value: object) -> date | None:
    if not isinstance(value, str) :
        return None

    try:
        return date.fromisoformat(value)

    except ValueError:
        return None
    

def _retry_delay(
    exc: HTTPError,
    fallback: float,
) -> float:
    """Use Retry-After when available, otherwise use exponential backoff."""

    retry_after = (
        exc.headers.get("Retry-After")
        if exc.headers
        else None
    )

    return max(0.0, float(retry_after)) if retry_after else fallback
