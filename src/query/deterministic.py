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
# Conservative lexical hints
#
# These do NOT try to understand arbitrary language. They only help resolve
# ambiguous connectors such as "for".
# ---------------------------------------------------------------------------

_POPULATION_HINTS = {
    "adult",
    "adults",
    "adolescent",
    "adolescents",
    "child",
    "children",
    "clinician",
    "clinicians",
    "developer",
    "developers",
    "doctor",
    "doctors",
    "elderly",
    "learner",
    "learners",
    "men",
    "patient",
    "patients",
    "participant",
    "participants",
    "physician",
    "physicians",
    "researcher",
    "researchers",
    "respondent",
    "respondents",
    "student",
    "students",
    "subject",
    "subjects",
    "survivor",
    "survivors",
    "teacher",
    "teachers",
    "user",
    "users",
    "volunteer",
    "volunteers",
    "women",
    "worker",
    "workers",
}

_DOMAIN_HINTS = {
    "agriculture",
    "biology",
    "education",
    "finance",
    "healthcare",
    "medicine",
    "medical",
    "robotics",
    "security",
    "cybersecurity",
    "manufacturing",
    "transportation",
}

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

_PROBLEM_NOUN_HINTS = {
    "answering",
    "classification",
    "detection",
    "diagnosis",
    "estimation",
    "generation",
    "prediction",
    "question",
    "retrieval",
    "summarization",
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
# Text normalization
# ---------------------------------------------------------------------------


def clean_idea_text(idea: str) -> str:
    """Normalize a research idea while preserving useful technical symbols."""
    if not isinstance(idea, str) or not idea.strip():
        raise ValueError("idea must not be empty")

    text = unicodedata.normalize("NFKC", idea)

    # Normalize Unicode dashes.
    text = re.sub(r"[\u2010-\u2015]", "-", text)

    # Remove punctuation unlikely to carry retrieval meaning.
    #
    # Preserve:
    #   C++, C#, R&D, input/output, GPT-5, don't, decimal.dot
    text = re.sub(r"[^\w\s+/#&.'()%-]", " ", text, flags=re.UNICODE)

    # Collapse all whitespace.
    text = " ".join(text.split())

    return text.strip(" .,-")


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------


class DeterministicDecomposer:
    """
    Conservative rule-based query decomposer.

    This implementation intentionally prefers missing information over
    unsupported semantic guesses. Its purpose is to provide a transparent,
    reproducible baseline against which semantic decomposers can be measured.
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
        self._refine_goal_facets(facets)

        synonyms = _extract_explicit_synonyms(cleaned)
        keywords = _keywords(cleaned, facets)

        return ResearchIdea(
            original_text=cleaned,
            keywords=keywords,
            synonyms=synonyms,
            **facets,
        )

    def _parse_connector_clauses(
        self,
        text: str,
        matches: list[re.Match[str]],
        facets: dict[str, list[str]],
    ) -> None:
        """Split text around explicit connector clauses."""

        prefix = text[: matches[0].start()].strip(" ,.-")

        if prefix:
            _append_unique(facets["problem"], prefix)

        for index, match in enumerate(matches):
            start = match.end()

            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )

            phrase = text[start:end].strip(" ,.-")

            if not phrase:
                continue

            connector = _normalize_connector(match.group())

            field = self._resolve_connector_field(connector, phrase)

            _append_unique(facets[field], phrase)

    def _resolve_connector_field(self, connector: str, phrase: str) -> str:
        """
        Resolve connector meaning conservatively.

        Most connectors are fixed. "for" is ambiguous enough that blindly
        mapping it to population causes obvious errors, so it receives a small
        deterministic classifier.
        """

        if connector != "for":
            return _CONNECTOR_TO_FIELD[connector]

        return _classify_for_phrase(phrase)

    def _split_using_goal(self, facets: dict[str, list[str]]) -> None:
        """
        Split:

            Using METHOD to GOAL

        into method + problem/outcome when the grammar is explicit.
        """

        methods = facets["intervention_or_method"]

        if not methods:
            return

        first = methods[0]

        parts = re.split(
            r"\s+to\s+",
            first,
            maxsplit=1,
            flags=re.IGNORECASE,
        )

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
        """
        Move clearly outcome-oriented problem phrases into outcomes.

        Example:
            "improving retrieval accuracy"
                -> outcomes

        This only fires on explicit outcome verbs.
        """

        remaining_problems: list[str] = []

        for problem in facets["problem"]:
            if _starts_with_any(problem, _OUTCOME_VERBS):
                _append_unique(facets["outcomes"], problem)
            else:
                remaining_problems.append(problem)

        facets["problem"] = remaining_problems


# ---------------------------------------------------------------------------
# Ambiguous connector resolution
# ---------------------------------------------------------------------------


def _classify_for_phrase(phrase: str) -> str:
    """
    Classify the phrase following ``for``.

    Examples:

        for elderly patients
            -> population

        for language learners
            -> population

        for medical question answering
            -> problem

        for predicting mortality
            -> problem

        for improving retrieval accuracy
            -> outcomes

        for healthcare
            -> domain

    Unknown cases default to ``problem`` rather than incorrectly inventing
    a population.
    """

    tokens = _normalized_tokens(phrase)

    if not tokens:
        return "problem"

    token_set = set(tokens)

    if _looks_like_population(phrase):
        return "population"

    if _starts_with_any(phrase, _OUTCOME_VERBS):
        return "outcomes"

    if _starts_with_any(phrase, _PROBLEM_VERBS):
        return "problem"

    if token_set & _PROBLEM_NOUN_HINTS:
        return "problem"

    if len(tokens) <= 3 and token_set & _DOMAIN_HINTS:
        return "domain"

    return "problem"

# ---------------------------------------------------------------------------
# Explicit synonym/acronym extraction
# ---------------------------------------------------------------------------


_EXPLICIT_ACRONYM_RE = re.compile(
    r"\b(?P<long>[A-Za-z][A-Za-z0-9 -]{4,}?)\s*"
    r"\((?P<short>[A-Z][A-Z0-9+-]{1,9})\)"
)


def _extract_explicit_synonyms(text: str) -> dict[str, list[str]]:
    """
    Extract only synonyms explicitly provided by the user.

    Example:

        retrieval augmented generation (RAG)

    becomes:

        {
            "retrieval augmented generation": ["RAG"]
        }

    The deterministic decomposer deliberately does not invent synonyms.
    """

    synonyms: dict[str, list[str]] = {}

    for match in _EXPLICIT_ACRONYM_RE.finditer(text):
        long_form = " ".join(match.group("long").split()).strip()
        short_form = match.group("short").strip()

        # Avoid swallowing a large preceding clause.
        long_form = _trim_acronym_long_form(long_form, short_form)

        if not long_form:
            continue

        values = synonyms.setdefault(long_form, [])

        if short_form.casefold() != long_form.casefold():
            _append_unique(values, short_form)

    return synonyms


def _trim_acronym_long_form(long_form: str, acronym: str) -> str:
    """
    Keep only a plausible acronym-length suffix.

    For RAG, for example, there is no reason to retain fifteen preceding words.
    """

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
    """
    Build bounded, deterministic keywords.

    Priority:
        1. complete extracted facet phrases
        2. technical/acronym tokens
        3. meaningful lexical tokens
    """

    candidates: list[str] = []

    # Full research phrases carry more retrieval meaning than isolated words.
    for phrases in facets.values():
        candidates.extend(phrases)

    tokens = re.findall(
        r"[^\W_][\w+#.-]*",
        text,
        flags=re.UNICODE,
    )

    # Preserve short scientific acronyms such as AI, ML, RL, CV and QA.
    candidates.extend(
        token
        for token in tokens
        if _is_useful_token(token)
    )

    return _deduplicate(candidates, limit=limit)


def _is_useful_token(token: str) -> bool:
    normalized = token.casefold().strip(".-")

    if not normalized:
        return False

    if normalized in _STOPWORDS:
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
        "comparison": [],
        "outcomes": [],
        "domain": [],
        "constraints": [],
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

def _looks_like_population(phrase: str) -> bool:
    """Return True for explicit human/study population phrases."""

    tokens = _normalized_tokens(phrase)

    if not tokens:
        return False

    token_set = set(tokens)

    if token_set & _POPULATION_HINTS:
        return True

    # Conservative plural human-group suffixes.
    #
    # Examples:
    #   language learners
    #   software developers
    #   cancer survivors
    #
    # Keep this deliberately narrow. The deterministic baseline should
    # prefer missing a population over inventing one.
    population_suffixes = (
        "learners",
        "developers",
        "survivors",
        "respondents",
        "subjects",
        "volunteers",
    )

    return tokens[-1] in population_suffixes
