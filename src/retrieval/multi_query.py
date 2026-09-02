"""Hybrid multi-query retrieval and candidate merging."""

from __future__ import annotations

import logging
import re
import hashlib
import json
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
import time

from src.models.paper import Paper
from src.models.query import RetrievalMode, SearchQuery
from src.retrieval.base import (
    PaperRetriever,
    RetrievalError,
    RetrievalRequest,
)
from src.retrieval.deduplication import deduplicate_paper_models
from src.retrieval.store import RetrievalStore


LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_CANDIDATES = 100
DEFAULT_PER_ROUTE_LIMIT = 20
DEFAULT_RETRIEVAL_CACHE_TTL_SECONDS = 6 * 60 * 60


@dataclass(slots=True)
class RetrievalFailure:
    """One retrieval route that failed."""

    query: str
    provider: str
    mode: str
    query_strategy: str
    query_source: str
    error: str

    def to_dict(self) -> dict[str, str]:
        return {
            "query": self.query,
            "provider": self.provider,
            "mode": self.mode,
            "query_strategy": self.query_strategy,
            "query_source": self.query_source,
            "error": self.error,
        }


@dataclass(slots=True)
class MultiQueryResult:
    """Result of executing all planned retrieval routes."""

    papers: list[Paper] = field(default_factory=list)
    failures: list[RetrievalFailure] = field(default_factory=list)
    requested_routes: int = 0

    @property
    def partial(self) -> bool:
        return bool(self.failures and self.papers)


class MultiQueryRetriever:
    """Retrieve candidates from several search routes and merge them."""

    def __init__(
        self,
        client: PaperRetriever,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        per_route_limit: int = DEFAULT_PER_ROUTE_LIMIT,
        max_workers: int = 4,
        cache_path: str | Path | None = None,
        retrieval_cache_ttl_seconds: float = DEFAULT_RETRIEVAL_CACHE_TTL_SECONDS,
        clock=None,
    ) -> None:
        if not 1 <= max_candidates <= 500:
            raise ValueError(
                "max_candidates must be between 1 and 500"
            )

        if not 1 <= per_route_limit <= 100:
            raise ValueError(
                "per_route_limit must be between 1 and 100"
            )

        if not 1 <= max_workers <= 16:
            raise ValueError(
                "max_workers must be between 1 and 16"
            )

        self.client = client
        self.max_candidates = max_candidates
        self.per_route_limit = per_route_limit
        self.max_workers = max_workers
        self.retrieval_store = RetrievalStore(
            cache_path,
            ttl_seconds=retrieval_cache_ttl_seconds,
            clock=clock,
        )
        self._cache_lock = RLock()
        self._inflight: dict[str, Future[list[Paper]]] = {}
        self._permanent_failures: dict[str, RetrievalError] = {}
        self._metrics: dict[str, int] = {
            "retrieval_cache_hits": 0,
            "retrieval_cache_misses": 0,
            "retrieval_provider_requests": 0,
            "verification_retrieval_cache_hits": 0,
            "retrieval_failure_cache_hits": 0,
        }

    def metrics_snapshot(self) -> dict[str, int]:
        with self._cache_lock:
            return dict(self._metrics)

    def retrieve_hybrid(
        self,
        queries: Sequence[SearchQuery],
        *,
        limit: int | None = None,
    ) -> MultiQueryResult:
        """Retrieve, deduplicate, and bound hybrid candidates."""

        if not queries:
            raise ValueError(
                "at least one query is required"
            )
        
        if limit is not None and not 1 <= limit <= self.max_candidates:
            raise ValueError(
                f"limit must be between 1 "
                f"and {self.max_candidates}"
            )

        ceiling = limit if limit is not None else self.max_candidates
        requests = self._build_requests(queries)

        papers, failures = self._execute(requests)

        unique = deduplicate_paper_models(
            papers
        )

        LOGGER.info(
            "retrieval merge raw_count=%d "
            "deduplicated_count=%d "
            "ceiling=%d requested_routes=%d",
            len(papers),
            len(unique),
            ceiling,
            len(requests),
        )

        return MultiQueryResult(
            papers=unique[:ceiling],
            failures=failures,
            requested_routes=len(requests),
        )

    def retrieve_verification(
        self,
        queries: Sequence[SearchQuery],
        *,
        per_query_limit: int = 10,
        limit: int | None = None,
        adaptive: bool = True,
    ) -> MultiQueryResult:
        """Run a bounded lexical search plan for one candidate hypothesis.

        Verification intentionally uses the existing provider boundary and
        normalization path, but does not add semantic routes or rank against
        the original research idea.  The caller may apply candidate-specific
        ranking after this targeted retrieval.
        """

        if not queries:
            raise ValueError("at least one verification query is required")
        if not 1 <= per_query_limit <= 100:
            raise ValueError("per_query_limit must be between 1 and 100")
        if limit is not None and not 1 <= limit <= self.max_candidates:
            raise ValueError(f"limit must be between 1 and {self.max_candidates}")

        ceiling = limit if limit is not None else self.max_candidates
        requests = [
            RetrievalRequest(
                query=query,
                mode=RetrievalMode.BROAD_LEXICAL,
                limit=min(per_query_limit, 100),
            )
            for query in queries
        ]
        if adaptive:
            papers, failures, requested_routes = self._execute_adaptive_verification(
                requests,
            )
        else:
            papers, failures = self._execute(requests, verification=True)
            requested_routes = len(requests)
        unique = deduplicate_paper_models(papers)
        return MultiQueryResult(
            papers=unique[:ceiling],
            failures=failures,
            requested_routes=requested_routes,
        )

    def _execute_adaptive_verification(
        self,
        requests: Sequence[RetrievalRequest],
    ) -> tuple[list[Paper], list[RetrievalFailure], int]:
        """Stop only after conservative marginal-coverage exhaustion."""

        raw_papers: list[Paper] = []
        failures: list[RetrievalFailure] = []
        previous_queries: list[str] = []
        no_new_streak = 0
        requested_routes = 0
        seen_provider_ids: set[str] = set()

        for request in requests:
            current, current_failures = self._execute([request], verification=True)
            requested_routes += 1
            raw_papers.extend(current)
            failures.extend(current_failures)

            # A provider's paper ID is sufficient for marginal-coverage
            # accounting even when a synthetic/test record lacks the
            # metadata needed by scholarly title/year fallback deduplication.
            # Final merging still uses the repository's conservative
            # scholarly identity rules below.
            current_ids = {
                paper.id.casefold()
                for paper in current
            }
            new_count = len(current_ids - seen_provider_ids)
            seen_provider_ids.update(current_ids)

            if new_count == 0 and not current_failures:
                no_new_streak += 1
            elif current_failures:
                no_new_streak = 0
            else:
                no_new_streak = 0

            query_text = request.query.text
            previous_queries.append(query_text)

            # Keep at least two attempts. A single empty/failed request is not
            # enough to conclude that marginal coverage is exhausted.
            if requested_routes >= 2 and requested_routes < len(requests):
                remaining = requests[requested_routes:]
                remaining_redundant = all(
                    any(
                        _query_overlap(
                            future.query.text,
                            previous,
                        ) >= 0.8
                        for previous in previous_queries
                    )
                    for future in remaining
                )
                if (
                    no_new_streak >= 2
                    and remaining_redundant
                    and not current_failures
                ):
                    break

        return raw_papers, failures, requested_routes

    def _build_requests(
        self,
        queries: Sequence[SearchQuery],
    ) -> list[RetrievalRequest]:
        """Build the hybrid retrieval plan.

        The original idea receives lexical, title/abstract, and semantic
        retrieval. Generated queries expand lexical recall.
        """

        original = queries[0]

        requests = [
            RetrievalRequest(
                query=original,
                mode=RetrievalMode.BROAD_LEXICAL,
                limit=self.per_route_limit,
            ),
            RetrievalRequest(
                query=original,
                mode=RetrievalMode.TITLE_ABSTRACT,
                limit=self.per_route_limit,
            ),
            RetrievalRequest(
                query=original,
                mode=RetrievalMode.SEMANTIC,
                limit=self.per_route_limit,
            ),
        ]

        requests.extend(
            RetrievalRequest(
                query=query,
                mode=RetrievalMode.BROAD_LEXICAL,
                limit=self.per_route_limit,
            )
            for query in queries[1:]
        )

        return requests

    def _execute(
        self,
        requests: Sequence[RetrievalRequest],
        *,
        verification: bool = False,
    ) -> tuple[list[Paper], list[RetrievalFailure]]:
        results: dict[int, list[Paper]] = {}
        failures: dict[int, RetrievalFailure] = {}

        workers = min(
            self.max_workers,
            len(requests),
        )

        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:
            futures = {
                executor.submit(
                    self._search_cached,
                    request,
                    verification,
                ): index
                for index, request in enumerate(requests)
            }

            for future in as_completed(futures):
                index = futures[future]
                request = requests[index]

                try:
                    papers = future.result()

                    if not all(
                        isinstance(paper, Paper)
                        for paper in papers
                    ):
                        raise TypeError(
                            "PaperRetriever.search must "
                            "return Paper models"
                        )

                    results[index] = list(papers)

                except RetrievalError as exc:
                    failure = RetrievalFailure(
                        query=request.query.text,
                        provider=self.client.provider_name,
                        mode=request.mode.value,
                        query_strategy=request.query.strategy,
                        query_source=request.query.source,
                        error=str(exc),
                    )

                    failures[index] = failure

                    LOGGER.error(
                        "retrieval failed provider=%s "
                        "mode=%s query=%r error=%s",
                        failure.provider,
                        failure.mode,
                        failure.query,
                        failure.error,
                    )

        papers = _interleave_route_results(
            results,
            route_count=len(requests),
        )

        ordered_failures = [
            failures[index]
            for index in sorted(failures)
        ]

        return papers, ordered_failures

    def _search_cached(
        self,
        request: RetrievalRequest,
        verification: bool = False,
    ) -> list[Paper]:
        """Use one shared TTL cache for initial and verification retrieval."""

        cache_key = _retrieval_cache_key(self.client, request)
        store_enabled = self.retrieval_store.path is not None

        with self._cache_lock:
            permanent_failure = self._permanent_failures.get(cache_key)
            if permanent_failure is not None:
                self._metrics["retrieval_failure_cache_hits"] += 1
                raise type(permanent_failure)(str(permanent_failure))

        if store_enabled:
            cached = self.retrieval_store.get(cache_key)
            if cached is not None:
                self._record_cache_hit(verification)
                return cached

        with self._cache_lock:
            pending = self._inflight.get(cache_key)
            if pending is None:
                pending = Future()
                self._inflight[cache_key] = pending
                owner = True
                if store_enabled:
                    self._metrics["retrieval_cache_misses"] += 1
                self._metrics["retrieval_provider_requests"] += 1
            else:
                owner = False

        if not owner:
            self._record_cache_hit(verification)
            return pending.result()

        try:
            papers = self.client.search(request)
            if store_enabled:
                try:
                    self.retrieval_store.put(cache_key, papers)
                except Exception as exc:
                    LOGGER.warning(
                        "retrieval cache write failed provider=%s error=%s",
                        getattr(self.client, "provider_name", type(self.client).__name__),
                        exc,
                    )
            pending.set_result(papers)
            return papers
        except BaseException as exc:
            if isinstance(exc, RetrievalError) and _is_permanent_retrieval_failure(exc):
                with self._cache_lock:
                    self._permanent_failures[cache_key] = exc
            pending.set_exception(exc)
            raise
        finally:
            with self._cache_lock:
                self._inflight.pop(cache_key, None)

    def _record_cache_hit(self, verification: bool) -> None:
        with self._cache_lock:
            self._metrics["retrieval_cache_hits"] += 1
            if verification:
                self._metrics["verification_retrieval_cache_hits"] += 1


def _retrieval_cache_key(client: PaperRetriever, request: RetrievalRequest) -> str:
    """Hash provider-request semantics, excluding downstream pipeline state."""

    provider = getattr(client, "provider_name", type(client).__name__)
    payload = {
        "provider": str(provider).casefold(),
        "query": " ".join(request.query.text.split()).casefold(),
        "mode": request.mode.value,
        "limit": request.limit,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_permanent_retrieval_failure(error: RetrievalError) -> bool:
    """Identify same-run provider failures that retrying cannot repair."""

    return bool(re.search(r"\bHTTP\s+4\d\d\b", str(error), re.I))
    

def _interleave_route_results(
    results: dict[int, list[Paper]],
    *,
    route_count: int,
) -> list[Paper]:
    """Interleave routes so no early route monopolizes the candidate pool."""

    routes = [
        results.get(index, [])
        for index in range(route_count)
    ]

    max_depth = max(
        (len(route) for route in routes),
        default=0,
    )

    return [
        route[rank]
        for rank in range(max_depth)
        for route in routes
        if rank < len(route)
    ]


def _query_overlap(left: str, right: str) -> float:
    """Compare query phrase content without interpreting scientific terms."""

    left_terms = set(re.findall(r"[a-z0-9]+", left.casefold()))
    right_terms = set(re.findall(r"[a-z0-9]+", right.casefold()))

    if not left_terms or not right_terms:
        return 0.0

    return len(left_terms & right_terms) / min(len(left_terms), len(right_terms))
