"""Deterministic paper identity resolution and provenance merging."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from src.models.paper import Paper, RetrievalProvenance


def deduplicate_paper_models(papers: Iterable[Paper]) -> list[Paper]:
    """Merge duplicate papers while preserving metadata and provenance.

    Identity priority:
    1. OpenAlex ID
    2. DOI
    3. Normalized title + publication year, when strong IDs do not conflict
    """

    items = [paper.model_copy(deep=True) for paper in papers]

    if not items:
        return []

    parents = list(range(len(items)))

    # Strong identifiers known for each union-find cluster.
    cluster_openalex = [
        {key} if (key := normalize_openalex_id(paper.openalex_id)) else set()
        for paper in items
    ]
    cluster_dois = [
        {key} if (key := normalize_doi(paper.doi)) else set()
        for paper in items
    ]

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> int:
        left_root = find(left)
        right_root = find(right)

        if left_root == right_root:
            return left_root

        # Preserve first-seen ordering.
        root, child = sorted((left_root, right_root))
        parents[child] = root

        cluster_openalex[root].update(cluster_openalex[child])
        cluster_dois[root].update(cluster_dois[child])

        return root

    def compatible(left: int, right: int) -> bool:
        """Can these clusters safely merge using weak title/year evidence?"""

        left = find(left)
        right = find(right)

        left_openalex = cluster_openalex[left]
        right_openalex = cluster_openalex[right]

        if (
            left_openalex
            and right_openalex
            and left_openalex.isdisjoint(right_openalex)
        ):
            return False

        left_dois = cluster_dois[left]
        right_dois = cluster_dois[right]

        if left_dois and right_dois and left_dois.isdisjoint(right_dois):
            return False

        return True

    # ------------------------------------------------------------------
    # 1. Merge using strong identifiers.
    # ------------------------------------------------------------------

    openalex_seen: dict[str, int] = {}
    doi_seen: dict[str, int] = {}

    for index, paper in enumerate(items):
        openalex_key = normalize_openalex_id(paper.openalex_id)
        doi_key = normalize_doi(paper.doi)

        if openalex_key:
            if openalex_key in openalex_seen:
                union(index, openalex_seen[openalex_key])
            else:
                openalex_seen[openalex_key] = index

        if doi_key:
            if doi_key in doi_seen:
                union(index, doi_seen[doi_key])
            else:
                doi_seen[doi_key] = index

    # ------------------------------------------------------------------
    # 2. Fallback to exact normalized title + year.
    # ------------------------------------------------------------------

    title_year_seen: dict[
        tuple[str, int | None],
        list[int],
    ] = {}

    for index, paper in enumerate(items):
        identity = title_year_identity(paper)

        if identity is None:
            continue

        root = find(index)
        candidates = title_year_seen.setdefault(identity, [])

        for previous in candidates:
            previous_root = find(previous)

            if root == previous_root:
                break

            if compatible(root, previous_root):
                union(root, previous_root)
                break
        else:
            candidates.append(root)

    # ------------------------------------------------------------------
    # 3. Merge records inside each final cluster.
    # ------------------------------------------------------------------

    groups: dict[int, list[int]] = {}

    for index in range(len(items)):
        groups.setdefault(find(index), []).append(index)

    result: list[Paper] = []

    for indexes in sorted(groups.values(), key=lambda group: group[0]):
        paper = items[indexes[0]]

        for index in indexes[1:]:
            _merge_paper(paper, items[index])

        _refresh_internal_id(paper)
        result.append(paper)

    return result


def normalize_openalex_id(value: str | None) -> str:
    normalized = (value or "").strip().rstrip("/").casefold()

    return re.sub(
        r"^(?:https?://openalex\.org/|openalex:\s*)",
        "",
        normalized,
    )


def normalize_doi(value: str | None) -> str:
    normalized = (value or "").strip().casefold().rstrip("/")

    return re.sub(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
        "",
        normalized,
    )


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(
        r"[^\w]+",
        " ",
        normalized,
        flags=re.UNICODE,
    )
    return " ".join(normalized.split())


def title_year_identity(
    paper: Paper,
) -> tuple[str, int] | None:
    title = normalize_title(paper.title)

    if (
        not title
        or title == "untitled"
        or paper.publication_year is None
    ):
        return None

    return title, paper.publication_year


def _merge_paper(target: Paper, incoming: Paper) -> None:
    """Merge useful metadata from two records of the same paper."""

    target.provenance = _merge_provenance(
        target.provenance,
        incoming.provenance,
    )

    target.citation_count = max(
        target.citation_count,
        incoming.citation_count,
    )

    # Prefer the more complete abstract.
    if not target.abstract or (
        incoming.abstract
        and len(incoming.abstract) > len(target.abstract)
    ):
        target.abstract = incoming.abstract

    # Merge authors while preserving order.
    authors = list(target.authors)
    seen_authors = {author.casefold() for author in authors}

    for author in incoming.authors:
        key = author.casefold()

        if key not in seen_authors:
            authors.append(author)
            seen_authors.add(key)

    target.authors = authors

    # Fill metadata missing from the first-seen record.
    for field_name in (
        "openalex_id",
        "doi",
        "publication_year",
        "publication_date",
        "source",
        "url",
    ):
        if (
            getattr(target, field_name) is None
            and getattr(incoming, field_name) is not None
        ):
            setattr(
                target,
                field_name,
                getattr(incoming, field_name),
            )

    if target.title.casefold() == "untitled":
        target.title = incoming.title


def _merge_provenance(
    current: list[RetrievalProvenance],
    incoming: list[RetrievalProvenance],
) -> list[RetrievalProvenance]:
    """Merge retrieval routes without duplicating identical provenance."""

    result = [item.model_copy(deep=True) for item in current]

    seen = {
        (
            item.query.comparison_key,
            item.provider.casefold(),
            item.mode.value,
            item.provider_rank,
        )
        for item in result
    }

    for item in incoming:
        key = (
            item.query.comparison_key,
            item.provider.casefold(),
            item.mode.value,
            item.provider_rank,
        )

        if key not in seen:
            result.append(item.model_copy(deep=True))
            seen.add(key)

    return result


def _refresh_internal_id(paper: Paper) -> None:
    if paper.openalex_id:
        paper.id = paper.openalex_id
    elif paper.doi:
        paper.id = paper.doi
