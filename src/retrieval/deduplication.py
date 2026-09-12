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
        {
            key
            for value in [paper.openalex_id, *paper.openalex_aliases]
            if (key := normalize_openalex_id(value))
        }
        for paper in items
    ]
    cluster_dois = [
        {
            key
            for value in [paper.doi, *paper.doi_aliases]
            if (key := normalize_doi(value))
        }
        for paper in items
    ]
    cluster_members = [{index} for index in range(len(items))]

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
        cluster_members[root].update(cluster_members[child])

        return root

    def compatible(left: int, right: int) -> bool:
        """Reject title-based merges only when authoritative DOIs conflict."""

        left = find(left)
        right = find(right)

        left_dois = cluster_dois[left]
        right_dois = cluster_dois[right]

        if left_dois and right_dois and left_dois.isdisjoint(right_dois):
            # Distinct DOI strings are not always distinct scholarly works:
            # preprints, conference presentations, and versions of record can
            # each receive their own DOI. Only let exact/near bibliographic
            # evidence override that conflict when the authorship is strong.
            return any(
                _strong_authorship_match(items[left_index], items[right_index])
                for left_index in cluster_members[left]
                for right_index in cluster_members[right]
            )

        return True

    # ------------------------------------------------------------------
    # 1. Merge using strong identifiers.
    # ------------------------------------------------------------------

    openalex_seen: dict[str, int] = {}
    doi_seen: dict[str, int] = {}

    for index, paper in enumerate(items):
        openalex_keys = {
            key
            for value in [paper.openalex_id, *paper.openalex_aliases]
            if (key := normalize_openalex_id(value))
        }
        doi_keys = {
            key
            for value in [paper.doi, *paper.doi_aliases]
            if (key := normalize_doi(value))
        }

        for openalex_key in openalex_keys:
            if openalex_key in openalex_seen:
                union(index, openalex_seen[openalex_key])
            else:
                openalex_seen[openalex_key] = index

        for doi_key in doi_keys:
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
        _merge_doi_aliases(paper)
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
        _merge_doi_aliases(merged)
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
    left_dois = {
        normalize_doi(value)
        for value in [left.doi, *left.doi_aliases]
        if normalize_doi(value)
    }
    right_dois = {
        normalize_doi(value)
        for value in [right.doi, *right.doi_aliases]
        if normalize_doi(value)
    }
    if left_dois & right_dois:
        return True
    left_title_year = title_year_identity(left)
    if left_title_year and left_title_year == title_year_identity(right):
        return not (left_dois and right_dois) or _strong_authorship_match(left, right)
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
        or not _strong_authorship_match(left, right)
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


def _strong_authorship_match(left: Paper, right: Paper) -> bool:
    """Match versioned records despite accents, order, and abbreviated names.

    A single author is sufficient only for two single-author records. For
    multi-author works, two independently matching personal names are needed
    before different DOI values can be treated as manifestation identifiers.
    """

    left_authors = [_author_tokens(value) for value in left.authors]
    right_authors = [_author_tokens(value) for value in right.authors]
    left_authors = [value for value in left_authors if value]
    right_authors = [value for value in right_authors if value]
    if not left_authors or not right_authors:
        return False

    matched_right: set[int] = set()
    matches = 0
    for left_tokens in left_authors:
        for index, right_tokens in enumerate(right_authors):
            if index in matched_right or not _author_names_match(left_tokens, right_tokens):
                continue
            matched_right.add(index)
            matches += 1
            break
    return matches >= 2 or (
        matches == 1 and len(left_authors) == len(right_authors) == 1
    )


_NON_PERSON_AUTHOR_TERMS = frozenset({
    "association", "collaboration", "committee", "conference", "consortium",
    "group", "institute", "society", "team", "university",
})


def _author_tokens(value: str) -> frozenset[str]:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_letters = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    tokens = re.findall(r"[^\W\d_]+", ascii_letters.casefold(), flags=re.UNICODE)
    if (
        len(tokens) < 2
        or any(term in tokens for term in _NON_PERSON_AUTHOR_TERMS)
        or any(character.isdigit() for character in value)
    ):
        return frozenset()
    return frozenset(tokens)


def _author_names_match(left: frozenset[str], right: frozenset[str]) -> bool:
    common = left & right
    if len(common) >= 2:
        return len(common) / max(len(left), len(right)) >= 2 / 3
    # Initials can safely supplement a shared family/given token, but never
    # establish identity by themselves.
    if len(common) != 1 or len(left) > 3 or len(right) > 3:
        return False
    left_initials = {token[0] for token in left - common}
    right_initials = {token[0] for token in right - common}
    return bool(left_initials & right_initials)


def _paper_richness(paper: Paper) -> tuple[int, ...]:
    """Prefer the record with the most useful scientific/source metadata."""

    return (
        _work_type_priority(paper.work_type),
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


def _work_type_priority(value: str | None) -> int:
    normalized = normalize_title(value or "")
    if normalized == "preprint":
        return 0
    if normalized in {"", "other"}:
        return 1
    return 2


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

    for author in incoming.authors:
        author_tokens = _author_tokens(author)
        matching_index = next(
            (
                index
                for index, existing in enumerate(authors)
                if existing.casefold() == author.casefold()
                or (
                    author_tokens
                    and _author_names_match(_author_tokens(existing), author_tokens)
                )
            ),
            None,
        )
        if matching_index is None:
            authors.append(author)
        elif _author_display_richness(author) > _author_display_richness(
            authors[matching_index]
        ):
            authors[matching_index] = author

    target.authors = authors
    target.doi_aliases = list(target.doi_aliases)
    for alias in [incoming.doi, *incoming.doi_aliases]:
        if alias and normalize_doi(alias) not in {
            normalize_doi(value) for value in target.doi_aliases
        }:
            target.doi_aliases.append(alias)
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
        "work_type",
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


def _merge_doi_aliases(paper: Paper) -> None:
    aliases: list[str] = []
    seen: set[str] = set()
    for value in [paper.doi, *paper.doi_aliases]:
        key = normalize_doi(value)
        if key and key not in seen:
            seen.add(key)
            aliases.append(f"https://doi.org/{key}")
    paper.doi_aliases = aliases


def _author_display_richness(value: str) -> tuple[int, int, int]:
    return (
        len(_author_tokens(value)),
        sum(ord(character) > 127 for character in value),
        len(value),
    )


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
