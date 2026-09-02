"""Transparent, dependency-free query decomposition baseline."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from src.models.idea import ResearchIdea


# ---------------------------------------------------------------------------
# Connector parsing
# ---------------------------------------------------------------------------

_CONNECTOR_TO_FIELD: dict[str, str] = {
    "using": "intervention_or_method",
    "with": "data_or_modality",
    "from": "data_or_modality",
    "based on": "data_or_modality",
    "among": "population",
    "compared with": "comparison",
    "compared to": "comparison",
    "versus": "comparison",
    "vs": "comparison",
    "without": "constraints",
    "under": "constraints",
    "while": "comparison",
    "in": "domain",
}

_CONNECTOR_RE = re.compile(
    r"\b("
    r"compared\s+(?:with|to)"
    r"|using"
    r"|with"
    r"|from"
    r"|based\s+on"
    r"|among"
    r"|without"
    r"|while"
    r"|versus"
    r"|under"
    r"|vs\.?"
    r"|in"
    r"|for"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Generic grammatical cues
#
# These describe grammatical roles rather than scientific concepts.
# The deterministic decomposer intentionally avoids domain vocabularies,
# population dictionaries, method ontologies, or scientific synonym tables.
# ---------------------------------------------------------------------------

_OUTCOME_VERBS = {
    "improve",
    "improving",
    "increase",
    "increasing",
    "reduce",
    "reducing",
    "decrease",
    "decreasing",
    "minimize",
    "minimizing",
    "maximize",
    "maximizing",
    "enhance",
    "enhancing",
}

_PROBLEM_VERBS = {
    "classify",
    "classifying",
    "detect",
    "detecting",
    "diagnose",
    "diagnosing",
    "estimate",
    "estimating",
    "generate",
    "generating",
    "identify",
    "identifying",
    "predict",
    "predicting",
    "retrieve",
    "retrieving",
    "summarize",
    "summarizing",
}

_CONSTRAINT_PREFIXES = {
    "few",
    "limited",
    "scarce",
    "restricted",
    "insufficient",
    "small",
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
    "without",
    "while",
    "among",
    "compared",
    "based",
    "build",
    "building",
    "develop",
    "developing",
    "study",
    "studying",
}


# ---------------------------------------------------------------------------
# Generic data/modality wording
#
# These terms describe structural input roles rather than application domains.
# ---------------------------------------------------------------------------

_DATA_PHRASE_HINTS = {
    "data",
    "dataset",
    "datasets",
    "signal",
    "signals",
    "sensor",
    "sensors",
    "image",
    "images",
    "video",
    "videos",
    "text",
    "audio",
    "speech",
    "measurement",
    "measurements",
    "record",
    "records",
    "sample",
    "samples",
    "input",
    "inputs",
    "feature",
    "features",
    "embedding",
    "embeddings",
    "representation",
    "representations",
    "modality",
}

_DATA_PHRASE_ENDINGS = {
    "data",
    "dataset",
    "datasets",
    "signal",
    "signals",
    "image",
    "images",
    "video",
    "videos",
    "text",
    "audio",
    "speech",
    "measurement",
    "measurements",
    "record",
    "records",
    "sample",
    "samples",
    "input",
    "inputs",
    "feature",
    "features",
    "embedding",
    "embeddings",
    "representation",
    "representations",
    "modality",
    "form",
}


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def clean_idea_text(idea: str) -> str:
    """Normalize a research idea while preserving useful technical symbols."""

    if not isinstance(idea, str) or not idea.strip():
        raise ValueError("idea must not be empty")

    text = unicodedata.normalize("NFKC", idea)
    text = re.sub(r"[\u2010-\u2015]", "-", text)

    # Preserve useful technical characters such as:
    # C++, C#, R&D, input/output, GPT-5, don't, decimal.dot.
    text = re.sub(r"[^\w\s+/#&.'()%-]", " ", text, flags=re.UNICODE)
    text = " ".join(text.split())

    return text.strip(" .,-")


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

class DeterministicDecomposer:
    """Conservative rule-based research-idea decomposer.

    This baseline uses explicit grammar and generic structural cues only.
    It intentionally prefers missing information over unsupported semantic
    classification.
    """

    def decompose(self, idea: str) -> ResearchIdea:
        cleaned = clean_idea_text(idea)
        facets = _empty_facets()
        matches = list(_CONNECTOR_RE.finditer(cleaned))

        if not matches:
            _append_unique(facets["problem"], cleaned)
        else:
            self._parse_connector_clauses(cleaned, matches, facets)

        self._split_using_goal(facets)
        self._split_method_data_facets(facets)
        self._refine_goal_facets(facets)

        synonyms = _extract_explicit_synonyms(cleaned)
        keywords = _keywords(cleaned, facets)

        return ResearchIdea(
            original_text=cleaned,
            keywords=keywords,
            synonyms=synonyms,
            canonical_facets=_identity_canonical_facets(facets),
            **facets,
        )

    def _parse_connector_clauses(
        self,
        text: str,
        matches: list[re.Match[str]],
        facets: dict[str, list[str]],
    ) -> None:
        """Split text around explicit connector clauses."""
        prefix = text[:matches[0].start()].strip(" ,.-")

        if prefix:
            first_connector = _normalize_connector(matches[0].group())
            first_end = matches[1].start() if len(matches) > 1 else len(text)
            first_phrase = text[matches[0].end():first_end].strip(" ,.-")

            if (
                first_connector == "for"
                and (
                    _starts_with_any(first_phrase, _PROBLEM_VERBS)
                    or _starts_with_any(first_phrase, _OUTCOME_VERBS)
                )
            ):
                _append_unique(facets["intervention_or_method"], prefix)
            else:
                _append_unique(facets["problem"], prefix)

        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            phrase = text[start:end].strip(" ,.-")

            if not phrase:
                continue

            connector = _normalize_connector(match.group())
            field = self._resolve_connector_field(connector, phrase)
            _append_unique(facets[field], phrase)

    def _resolve_connector_field(self, connector: str, phrase: str) -> str:
        """Resolve ambiguous connectors conservatively.

        Explicit connectors retain fixed structural roles. Ambiguous
        constructions are resolved only when their grammar gives enough
        information; otherwise the baseline falls back to ``problem``.
        """

        if connector == "for":
            return _classify_for_phrase(phrase)

        if connector == "with" and _looks_like_constraint_phrase(phrase):
            return "constraints"

        if connector in {"with", "from", "based on"} and not _looks_like_data_phrase(phrase):
            return "intervention_or_method"

        return _CONNECTOR_TO_FIELD[connector]

    def _split_using_goal(self, facets: dict[str, list[str]]) -> None:
        """Split ``using METHOD to GOAL`` into method plus goal."""

        methods = facets["intervention_or_method"]

        if not methods:
            return

        parts = re.split(r"\s+to\s+", methods[0], maxsplit=1, flags=re.IGNORECASE)

        if len(parts) != 2:
            return

        method, goal = (part.strip() for part in parts)

        if not method or not goal:
            return

        methods[0] = method

        if _starts_with_any(goal, _OUTCOME_VERBS):
            _append_unique(facets["outcomes"], goal)
        else:
            _append_unique(facets["problem"], goal)

    def _refine_goal_facets(self, facets: dict[str, list[str]]) -> None:
        """Move explicit improvement-oriented goals into outcomes."""

        remaining: list[str] = []

        for problem in facets["problem"]:
            if _starts_with_any(problem, _OUTCOME_VERBS):
                _append_unique(facets["outcomes"], problem)
            else:
                remaining.append(problem)

        facets["problem"] = remaining

    def _split_method_data_facets(self, facets: dict[str, list[str]]) -> None:
        """Separate explicit data/modality clauses from method phrases."""

        remaining: list[str] = []

        for phrase in facets["intervention_or_method"]:
            parts = re.split(
                r"\s+(?:(?:with|from|based\s+on))\s+",
                phrase,
                maxsplit=1,
                flags=re.IGNORECASE,
            )

            if len(parts) == 2:
                method, trailing = (part.strip(" ,.-") for part in parts)

                if _looks_like_constraint_phrase(trailing):
                    if method:
                        remaining.append(method)
                    if trailing:
                        _append_unique(facets["constraints"], trailing)
                    continue

                if _looks_like_data_phrase(trailing):
                    if method:
                        remaining.append(method)
                    if trailing:
                        _append_unique(facets["data_or_modality"], trailing)
                    continue

            if _looks_like_constraint_phrase(phrase):
                _append_unique(facets["constraints"], phrase)
            elif _looks_like_data_phrase(phrase):
                _append_unique(facets["data_or_modality"], phrase)
            else:
                remaining.append(phrase)

        facets["intervention_or_method"] = remaining


# ---------------------------------------------------------------------------
# Ambiguous connector resolution
# ---------------------------------------------------------------------------

def _classify_for_phrase(phrase: str) -> str:
    """Classify ``for ...`` using generic structural cues only.

    Explicit task/outcome grammar takes priority. Short plural noun phrases
    are conservatively treated as populations/settings without maintaining a
    vocabulary of scientific or demographic entities.
    """

    tokens = _normalized_tokens(phrase)

    if not tokens:
        return "problem"

    if _starts_with_any(phrase, _OUTCOME_VERBS):
        return "outcomes"

    if _starts_with_any(phrase, _PROBLEM_VERBS):
        return "problem"

    if _looks_like_data_phrase(phrase):
        return "data_or_modality"

    if _looks_like_population_phrase(phrase):
        return "population"

    return "problem"
# ---------------------------------------------------------------------------
# Explicit synonym/acronym extraction
# ---------------------------------------------------------------------------

_EXPLICIT_ACRONYM_RE = re.compile(
    r"\b(?P<long>[A-Za-z][A-Za-z0-9 -]{4,}?)\s*"
    r"\((?P<short>[A-Z][A-Z0-9+-]{1,9})\)"
)


def _extract_explicit_synonyms(text: str) -> dict[str, list[str]]:
    """Extract only synonyms explicitly written by the user."""

    synonyms: dict[str, list[str]] = {}

    for match in _EXPLICIT_ACRONYM_RE.finditer(text):
        long_form = " ".join(match.group("long").split()).strip()
        short_form = match.group("short").strip()
        long_form = _trim_acronym_long_form(long_form, short_form)

        if not long_form:
            continue

        values = synonyms.setdefault(long_form, [])

        if short_form.casefold() != long_form.casefold():
            _append_unique(values, short_form)

    return synonyms


def _trim_acronym_long_form(long_form: str, acronym: str) -> str:
    """Keep only a plausible acronym-length suffix."""

    words = long_form.split()
    max_words = max(len(acronym) + 2, 3)

    if len(words) > max_words:
        words = words[-max_words:]

    return " ".join(words).strip()


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

def _keywords(
    text: str,
    facets: dict[str, list[str]],
    *,
    limit: int = 12,
) -> list[str]:
    """Build bounded deterministic retrieval keywords."""

    candidates: list[str] = []

    for phrases in facets.values():
        candidates.extend(phrases)

    tokens = re.findall(r"[^\W_][\w+#.-]*", text, flags=re.UNICODE)
    candidates.extend(token for token in tokens if _is_useful_token(token))

    return _deduplicate(candidates, limit=limit)


def _is_useful_token(token: str) -> bool:
    normalized = token.casefold().strip(".-")

    if not normalized or normalized in _STOPWORDS:
        return False

    if token.isupper() and 2 <= len(token) <= 10:
        return True

    return len(normalized) >= 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_facets() -> dict[str, list[str]]:
    return {
        "problem": [],
        "population": [],
        "intervention_or_method": [],
        "data_or_modality": [],
        "comparison": [],
        "outcomes": [],
        "domain": [],
        "constraints": [],
    }


def _identity_canonical_facets(
    facets: dict[str, list[str]],
) -> dict[str, dict[str, str]]:
    """Expose deterministic facets through the canonical identity contract."""

    return {
        facet: {value: value for value in values}
        for facet, values in facets.items()
        if values
    }


def _normalize_connector(connector: str) -> str:
    connector = connector.casefold().rstrip(".")
    return re.sub(r"\s+", " ", connector)


def _normalized_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"\b[\w-]+\b", text, flags=re.UNICODE)
    ]


def _starts_with_any(text: str, words: Iterable[str]) -> bool:
    first = next(iter(_normalized_tokens(text)), "")
    return first in words


def _looks_like_constraint_phrase(phrase: str) -> bool:
    """Recognize generic limiting grammar without scientific vocabulary."""

    tokens = _normalized_tokens(phrase)

    if not tokens:
        return False

    return tokens[0] in _CONSTRAINT_PREFIXES

def _looks_like_population_phrase(phrase: str) -> bool:
    """Recognize a conservative population/entity-group phrase.

    This uses grammatical shape only. It deliberately avoids hand-written
    lists such as patients, students, crops, bridges, developers, etc.
    """

    tokens = _normalized_tokens(phrase)

    if not 1 <= len(tokens) <= 4:
        return False

    head = tokens[-1]

    # A short phrase ending in an ordinary plural noun is a reasonable
    # deterministic population/entity-group signal:
    #
    #   adolescents
    #   language learners
    #   software developers
    #   bridge structures
    #
    # Explicit problem/outcome/data grammar is checked before this helper.
    if len(head) > 3 and head.endswith("s") and not head.endswith("ss"):
        return True

    return False


def _looks_like_data_phrase(phrase: str) -> bool:
    """Recognize generic input/data wording without domain classification."""

    tokens = _normalized_tokens(phrase)

    if not tokens:
        return False

    if not set(tokens) & _DATA_PHRASE_HINTS:
        return False

    return tokens[-1] in _DATA_PHRASE_ENDINGS


def _append_unique(values: list[str], value: str) -> None:
    value = " ".join(value.split()).strip()

    if not value:
        return

    normalized = value.casefold()

    if all(existing.casefold() != normalized for existing in values):
        values.append(value)


def _deduplicate(values: Iterable[str], *, limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        value = " ".join(value.split()).strip(" ,.-")

        if not value:
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

        if len(result) >= limit:
            break

    return result