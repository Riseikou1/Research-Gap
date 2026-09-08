"""Normalized full-text documents consumed by evidence extraction."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SectionType = Literal[
    "abstract", "introduction", "related_work", "methods", "materials",
    "dataset", "experimental_setup", "results", "discussion", "limitations",
    "conclusion", "future_work", "appendix", "other",
]
FullTextStatus = Literal[
    "not_attempted", "unavailable", "fetch_failed", "parse_failed", "usable"
]
SourceFormat = Literal["pdf", "xml", "html", "unknown"]


class PaperSection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    id: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    section_types: list[SectionType] = Field(min_length=1)
    text: str = Field(min_length=1)
    truncated: bool = False


class PaperDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    paper_id: str = Field(min_length=1)
    source_url: str | None = None
    source_format: SourceFormat = "unknown"
    sections: list[PaperSection] = Field(default_factory=list)
    status: FullTextStatus = "not_attempted"
    structure_available: bool = False
    truncated: bool = False
    notices: list[str] = Field(default_factory=list)

    def section(self, identifier: str) -> PaperSection | None:
        return next((item for item in self.sections if item.id == identifier), None)


_HEADING_RULES: tuple[tuple[SectionType, re.Pattern[str]], ...] = (
    ("abstract", re.compile(r"\babstract\b")),
    ("introduction", re.compile(r"\b(?:introduction|background)\b")),
    ("related_work", re.compile(r"\b(?:related work|literature review|prior work)\b")),
    ("methods", re.compile(r"\b(?:methods?|methodology|approach)\b")),
    ("materials", re.compile(r"\bmaterials?\b")),
    ("dataset", re.compile(r"\b(?:data(?:set)?s?|corpus|cohort)\b")),
    ("experimental_setup", re.compile(r"\b(?:experiments?|experimental setup|evaluation setup|implementation details)\b")),
    ("results", re.compile(r"\bresults?\b")),
    ("discussion", re.compile(r"\bdiscussion\b")),
    ("limitations", re.compile(r"\b(?:limitations?|weaknesses?)\b")),
    ("conclusion", re.compile(r"\b(?:conclusions?|concluding remarks|summary)\b")),
    ("future_work", re.compile(r"\b(?:future work|future directions?|outlook)\b")),
    ("appendix", re.compile(r"\b(?:appendix|appendices)\b")),
)


def normalize_heading(heading: str) -> list[SectionType]:
    """Map a generic heading to every applicable semantic role."""

    value = re.sub(r"[^\w]+", " ", heading.casefold()).strip()
    roles = [role for role, pattern in _HEADING_RULES if pattern.search(value)]
    return roles or ["other"]


FIELD_SECTION_PREFERENCES: dict[str, tuple[SectionType, ...]] = {
    "research_objective": ("abstract", "introduction"),
    "population_or_setting": ("methods", "materials", "dataset"),
    "data_or_modality": ("methods", "materials", "dataset"),
    "method_or_intervention": ("methods", "experimental_setup"),
    "constraints": ("methods", "experimental_setup"),
    "datasets": ("dataset", "methods", "materials", "experimental_setup"),
    "sample_size": ("dataset", "methods", "materials", "experimental_setup"),
    "comparison_or_baseline": ("experimental_setup", "results"),
    "evaluation_metrics": ("experimental_setup", "results"),
    "main_findings": ("results", "discussion", "abstract"),
    "limitations": ("limitations", "discussion", "conclusion"),
    "future_work": ("future_work", "conclusion", "discussion"),
}


def build_extraction_context(
    document: PaperDocument,
    *,
    max_chars: int,
    max_sections: int = 12,
) -> tuple[str, list[PaperSection], bool]:
    """Choose relevant sections once, preserving deterministic document order."""

    preferred = {
        role for roles in FIELD_SECTION_PREFERENCES.values() for role in roles
    }
    ranked = sorted(
        enumerate(document.sections),
        key=lambda pair: (
            0 if preferred.intersection(pair[1].section_types) else 1,
            pair[0],
        ),
    )[:max_sections]
    selected_indexes = {index for index, _ in ranked}
    selected = [
        section for index, section in enumerate(document.sections)
        if index in selected_indexes
    ]

    blocks: list[str] = []
    used: list[PaperSection] = []
    remaining = max_chars
    truncated = document.truncated
    for section in selected:
        header = (
            f"[Full text section id={section.id}; types="
            f"{','.join(section.section_types)}; heading={section.heading}]\n"
        )
        if remaining <= len(header):
            truncated = True
            break
        body = section.text[: remaining - len(header)]
        if len(body) < len(section.text):
            truncated = True
        blocks.append(header + body)
        used.append(section.model_copy(update={"text": body, "truncated": section.truncated or len(body) < len(section.text)}))
        remaining -= len(blocks[-1]) + 2
        if remaining <= 0:
            break
    return "\n\n".join(blocks), used, truncated


__all__ = [
    "FIELD_SECTION_PREFERENCES", "FullTextStatus", "PaperDocument", "PaperSection",
    "SectionType", "SourceFormat", "build_extraction_context", "normalize_heading",
]
