"""Strict JSONL loading and prediction persistence for offline evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    version: str
    cases: list[BaseModel]


def load_jsonl(path: str | Path, model_type: type[T], *, dataset_version: str = "unspecified") -> EvaluationDataset:
    """Load and validate JSONL; reject malformed rows and duplicate case IDs."""
    source = Path(path)
    if not dataset_version.strip():
        raise ValueError("dataset_version must not be empty")
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: list[T] = []
    seen: set[str] = set()
    version = dataset_version
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if isinstance(raw, dict) and "_meta" in raw:
                meta = raw["_meta"]
                if not isinstance(meta, dict) or not isinstance(meta.get("dataset_version"), str):
                    raise ValueError(f"{source}:{line_number}: _meta.dataset_version must be a string")
                version = meta["dataset_version"]
                continue
            try:
                item = model_type.model_validate(raw)
            except Exception as exc:
                raise ValueError(f"{source}:{line_number}: invalid evaluation row: {exc}") from exc
            case_id = getattr(item, "id", None)
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{source}:{line_number}: evaluation rows require a non-empty id")
            if case_id in seen:
                raise ValueError(f"{source}:{line_number}: duplicate case ID {case_id!r}")
            seen.add(case_id)
            rows.append(item)
    return EvaluationDataset(version=version, cases=rows)


def load_cases(path: str | Path, model_type: type[T], *, dataset_version: str = "unspecified") -> list[T]:
    return list(load_jsonl(path, model_type, dataset_version=dataset_version).cases)  # type: ignore[return-value]


def write_jsonl(path: str | Path, records: list[dict[str, Any]], *, dataset_version: str | None = None) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        if dataset_version is not None:
            handle.write(json.dumps({"_meta": {"dataset_version": dataset_version}}, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
