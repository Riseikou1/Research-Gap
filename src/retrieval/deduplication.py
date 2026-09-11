"""Deterministic paper identity resolution and provenance merging."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher

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
        """Reject title-based merges only when authoritative DOIs conflict."""

        left = find(left)
        right = find(right)

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
    # 3. Conservative near-title fallback. Exact title normalization above
    # handles punctuation/hyphen variants. Anything less exact additionally
    # requires author overlap and very high title similarity.
    # ------------------------------------------------------------------

    for index, paper in enumerate(items):
        for previous in range(index):
            root = find(index)
            previous_root = find(previous)
            if root == previous_root or not compatible(root, previous_root):
                continue
            if _guarded_near_title_match(paper, items[previous]):
                union(root, previous_root)
                break

    # ------------------------------------------------------------------
    # 4. Merge records inside each final cluster.
    # ------------------------------------------------------------------

    groups: dict[int, list[int]] = {}

    for index in range(len(items)):
        groups.setdefault(find(index), []).append(index)

    result: list[Paper] = []

    for indexes in sorted(groups.values(), key=lambda group: group[0]):
        richest = max(indexes, key=lambda index: _paper_richness(items[index]))
        paper = items[richest]
        ordered_provenance: list[RetrievalProvenance] = []
        for index in indexes:
            ordered_provenance = _merge_provenance(
                ordered_provenance,
                items[index].provenance,
            )

        for index in indexes:
            if index != richest:
                _merge_paper(paper, items[index])

        paper.provenance = ordered_provenance
        _merge_openalex_aliases(paper)
        _refresh_internal_id(paper)
        result.append(paper)

    return result


def canonicalize_against(
    papers: Iterable[Paper],
    canonical_references: Iterable[Paper],
) -> list[Paper]:
    """Map newly retrieved aliases onto an existing canonical paper roster."""

    references = deduplicate_paper_models(canonical_references)
    mapped: list[Paper] = []
    for incoming in papers:
        reference = next(
            (item for item in references if papers_represent_same_work(item, incoming)),
            None,
        )
        if reference is None:
            mapped.append(incoming.model_copy(deep=True))
            continue
        merged = deduplicate_paper_models([reference, incoming])[0]
        # The initial selected roster owns the cross-stage canonical ID.
        merged.id = reference.id
        if reference.openalex_id:
            merged.openalex_id = reference.openalex_id
        _merge_openalex_aliases(merged)
        mapped.append(merged)
    return deduplicate_paper_models(mapped)


def papers_represent_same_work(left: Paper, right: Paper) -> bool:
    """Conservative public identity predicate shared across pipeline stages."""

    left_openalex = {
        normalize_openalex_id(value)
        for value in [left.openalex_id, *left.openalex_aliases]
        if normalize_openalex_id(value)
    }
    right_openalex = {
        normalize_openalex_id(value)
        for value in [right.openalex_id, *right.openalex_aliases]
        if normalize_openalex_id(value)
    }
    if left_openalex & right_openalex:
        return True
    left_doi = normalize_doi(left.doi)
    right_doi = normalize_doi(right.doi)
    if left_doi and right_doi:
        return left_doi == right_doi
    left_title_year = title_year_identity(left)
    if left_title_year and left_title_year == title_year_identity(right):
        return True
    return _guarded_near_title_match(left, right)


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


def _guarded_near_title_match(left: Paper, right: Paper) -> bool:
    if (
        left.publication_year is None
        or left.publication_year != right.publication_year
        or not _authors_overlap(left.authors, right.authors)
    ):
        return False
    left_title = normalize_title(left.title)
    right_title = normalize_title(right.title)
    if not left_title or not right_title or left_title == right_title:
        return False
    left_tokens = set(left_title.split())
    right_tokens = set(right_title.split())
    if min(len(left_tokens), len(right_tokens)) < 5:
        return False
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    similarity = SequenceMatcher(None, left_title, right_title).ratio()
    return overlap >= 0.9 and similarity >= 0.96


def _authors_overlap(left: list[str], right: list[str]) -> bool:
    if not left or not right:
        return False
    left_keys = {normalize_title(author) for author in left}
    right_keys = {normalize_title(author) for author in right}
    return bool((left_keys - {""}) & (right_keys - {""}))


def _paper_richness(paper: Paper) -> tuple[int, ...]:
    """Prefer the record with the most useful scientific/source metadata."""

    return (
        int(bool(paper.abstract)),
        len(paper.abstract or ""),
        int(bool(paper.doi)),
        len(paper.full_text_locations),
        len(paper.authors),
        int(bool(paper.publication_date)),
        int(bool(paper.source)),
        int(bool(paper.url)),
        len(paper.title),
    )


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
    target.openalex_aliases = list(target.openalex_aliases)
    for alias in [incoming.openalex_id, *incoming.openalex_aliases]:
        if alias and alias.casefold().rstrip("/") not in {
            value.casefold().rstrip("/") for value in target.openalex_aliases
        }:
            target.openalex_aliases.append(alias)

    location_urls = {
        item.url.casefold().rstrip("/") for item in target.full_text_locations
    }
    for location in incoming.full_text_locations:
        key = location.url.casefold().rstrip("/")
        if key not in location_urls:
            target.full_text_locations.append(location.model_copy(deep=True))
            location_urls.add(key)

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

    if target.title.casefold() == "untitled" or len(incoming.title) > len(target.title):
        target.title = incoming.title


def _merge_openalex_aliases(paper: Paper) -> None:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in [paper.openalex_id, *paper.openalex_aliases]:
        key = normalize_openalex_id(value)
        if key and key not in seen:
            seen.add(key)
            aliases.append(f"https://openalex.org/{key.upper()}")
    paper.openalex_aliases = aliases


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
