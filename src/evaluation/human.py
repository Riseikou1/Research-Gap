"""Human annotation export and simple aggregation utilities."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Literal

from pydantic import Field

from .models import StrictModel


class AnnotationRecord(StrictModel):
    case_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    reviewer_id: str = Field(min_length=1)
    criterion: Literal["overall", "gap_usefulness", "gap_correctness", "evidence_sufficiency", "rationale_quality"] = "overall"
    notes: str | None = None


def export_gap_annotations(gaps: Iterable[object], path: str | Path, *, case_id: str, reviewer_id: str = "anonymous") -> None:
    """Export candidate context as unrated JSONL annotation tasks."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for gap in gaps:
            context = {
                "case_id": case_id,
                "candidate_id": getattr(gap, "id", ""),
                "title": getattr(gap, "title", ""),
                "description": getattr(gap, "description", ""),
                "rationale": getattr(gap, "rationale", ""),
                "supporting_evidence": [item.model_dump(mode="json") for item in getattr(gap, "supporting_evidence", [])],
                "rating": None,
                "reviewer_id": reviewer_id,
                "notes": None,
            }
            for criterion in ("gap_usefulness", "gap_correctness", "evidence_sufficiency", "rationale_quality"):
                record = {**context, "criterion": criterion}
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def aggregate_ratings(records: Iterable[AnnotationRecord | Mapping[str, object]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[int]] = {}
    for item in records:
        record = item if isinstance(item, AnnotationRecord) else AnnotationRecord.model_validate(item)
        key = record.candidate_id if record.criterion == "overall" else f"{record.candidate_id}:{record.criterion}"
        grouped.setdefault(key, []).append(record.rating)
    return {
        candidate_id: {
            "mean": mean(values),
            "median": median(values),
            "count": len(values),
            "standard_deviation": pstdev(values) if len(values) > 1 else 0.0,
        }
        for candidate_id, values in sorted(grouped.items())
    }


def write_annotations_csv(records: Iterable[AnnotationRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "candidate_id", "criterion", "rating", "reviewer_id", "notes"))
        writer.writeheader()
        for record in records:
            writer.writerow(record.model_dump())
