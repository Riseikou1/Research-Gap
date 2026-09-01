"""Hybrid multi-query retrieval and candidate merging."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from src.models.paper import Paper
from src.models.query import RetrievalMode, SearchQuery
from src.retrieval.base import (
    PaperRetriever,
    RetrievalError,
    RetrievalRequest,
)
from src.retrieval.deduplication import deduplicate_paper_models


LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_CANDIDATES = 100
DEFAULT_PER_ROUTE_LIMIT = 20


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
        papers, failures = self._execute(requests)
        unique = deduplicate_paper_models(papers)
        return MultiQueryResult(
            papers=unique[:ceiling],
            failures=failures,
            requested_routes=len(requests),
        )

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
                    self.client.search,
                    request,
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
