#!/usr/bin/env bash
set -Eeuo pipefail

# Milestone 7 offline harness acceptance check.
#
# Usage:
#   ./check_milestone7.sh
#   ./check_milestone7.sh evaluation/dev 20
#
# Argument 1: dataset directory (default: evaluation/dev)
# Argument 2: expected records in each dataset (default: 20)
#
# This creates oracle predictions from the gold rows. It verifies the evaluation
# machinery itself; it does not evaluate the live Milestones 1-6 pipeline.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python}"
DATASET_DIR="${1:-evaluation/dev}"
EXPECTED_CASES="${2:-20}"
RESULT_DIR="evaluation/results/milestone7_smoke"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/research-gap-m7.XXXXXX")"

cleanup() {
    rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT

fail() {
    echo "FAILED: $*" >&2
    exit 1
}

step() {
    echo
    echo "[$1/5] $2"
}

[[ "$EXPECTED_CASES" =~ ^[1-9][0-9]*$ ]] || fail "expected case count must be a positive integer"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python command not found: $PYTHON_BIN"

declare -A DATASETS=(
    [retrieval]="$DATASET_DIR/retrieval.json"
    [deduplication]="$DATASET_DIR/deduplication.json"
    [extraction]="$DATASET_DIR/extraction.json"
    [verification]="$DATASET_DIR/verification.json"
)

step 1 "Checking the project and dataset files"
[[ -f src/evaluation/runner.py ]] || fail "run this script from the research-gap repository root"
for evaluation_type in retrieval deduplication extraction verification; do
    [[ -f "${DATASETS[$evaluation_type]}" ]] || fail "missing dataset: ${DATASETS[$evaluation_type]}"
done
"$PYTHON_BIN" -c "import pydantic; import src.evaluation; import src.evaluation.runner" \
    || fail "evaluation imports failed; activate the project virtual environment first"

step 2 "Running the Milestone 7 unit tests"
"$PYTHON_BIN" -m unittest tests.unit.test_milestone7_evaluation -v

step 3 "Strictly loading datasets and creating oracle predictions"
"$PYTHON_BIN" - "$DATASET_DIR" "$TEMP_DIR" "$EXPECTED_CASES" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.evaluation.dataset import load_jsonl
from src.evaluation.models import (
    DeduplicationEvaluationCase,
    ExtractionEvaluationCase,
    RetrievalEvaluationCase,
    VerificationEvaluationCase,
)

dataset_dir = Path(sys.argv[1])
prediction_dir = Path(sys.argv[2])
expected_cases = int(sys.argv[3])

specifications = {
    "retrieval": (RetrievalEvaluationCase, "retrieval.json"),
    "deduplication": (DeduplicationEvaluationCase, "deduplication.json"),
    "extraction": (ExtractionEvaluationCase, "extraction.json"),
    "verification": (VerificationEvaluationCase, "verification.json"),
}


def prediction_for(evaluation_type: str, case: object) -> dict[str, object]:
    if evaluation_type == "retrieval":
        relevant = sorted(case.relevant_papers, key=lambda item: item.relevance, reverse=True)
        return {"case_id": case.id, "retrieved_ids": [item.paper_id for item in relevant]}
    if evaluation_type == "deduplication":
        return {"case_id": case.id, "duplicate_pairs": case.gold_duplicate_pairs}
    if evaluation_type == "extraction":
        return {"case_id": case.id, "evidence": case.gold}
    return {
        "case_id": case.id,
        "assessment": {
            "label": case.expected_label,
            "counterexample_paper_ids": case.known_counterexample_ids,
            "candidate_labels": case.expected_candidate_labels,
        },
    }


for evaluation_type, (model, filename) in specifications.items():
    dataset = load_jsonl(dataset_dir / filename, model)
    if len(dataset.cases) != expected_cases:
        raise SystemExit(
            f"{filename}: expected {expected_cases} cases, found {len(dataset.cases)}"
        )
    if dataset.version == "unspecified":
        raise SystemExit(f"{filename}: missing first-row _meta.dataset_version")

    output = prediction_dir / f"{evaluation_type}_predictions.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for case in dataset.cases:
            handle.write(json.dumps(prediction_for(evaluation_type, case), sort_keys=True) + "\n")
    print(f"{filename}: {len(dataset.cases)} valid cases, version={dataset.version}")
PY

step 4 "Running every evaluator through the command-line interface"
mkdir -p "$RESULT_DIR"
for evaluation_type in retrieval deduplication extraction verification; do
    report="$RESULT_DIR/${evaluation_type}_report.json"
    "$PYTHON_BIN" -m src.evaluation.runner \
        --dataset "${DATASETS[$evaluation_type]}" \
        --evaluation-type "$evaluation_type" \
        --predictions "$TEMP_DIR/${evaluation_type}_predictions.jsonl" \
        --output "$report"
done

step 5 "Checking report structure and perfect-oracle scores"
"$PYTHON_BIN" - "$RESULT_DIR" "$EXPECTED_CASES" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

result_dir = Path(sys.argv[1])
expected_cases = int(sys.argv[2])


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def perfect(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isclose(float(value), 1.0)


reports = {
    name: json.loads((result_dir / f"{name}_report.json").read_text(encoding="utf-8"))
    for name in ("retrieval", "deduplication", "extraction", "verification")
}

for name, report in reports.items():
    require(report.get("schema_version") == "m7-v1", f"{name}: wrong schema_version")
    require(report.get("dataset_version") not in (None, "", "unspecified"), f"{name}: missing dataset version")
    require(report.get("cases_total") == expected_cases, f"{name}: wrong cases_total")
    require(report.get("cases_completed") == expected_cases, f"{name}: not every case completed")
    require(report.get("cases_failed") == 0, f"{name}: one or more cases failed")
    require(not report.get("failures"), f"{name}: report contains failures")

retrieval = reports["retrieval"]["retrieval"]
require(retrieval["cases"] == expected_cases, "retrieval: not every case was scored")
for metric in ("recall_at_10", "recall_at_50", "mrr", "ndcg_at_10"):
    require(perfect(retrieval[metric]), f"retrieval: oracle {metric} was not 1.0")

deduplication = reports["deduplication"]["deduplication"]
for metric in ("precision", "recall", "f1"):
    require(perfect(deduplication[metric]), f"deduplication: oracle {metric} was not 1.0")

extraction = reports["extraction"]["extraction"]
require(extraction and extraction["micro"], "extraction: missing aggregate metrics")
for metric in ("precision", "recall", "f1"):
    require(perfect(extraction["micro"][metric]), f"extraction: oracle micro {metric} was not 1.0")

verification = reports["verification"]["verification"]
require(verification["cases_scored"] == expected_cases, "verification: not every case was scored")
require(perfect(verification["accuracy"]), "verification: oracle accuracy was not 1.0")
require(verification["false_promising_gap_count"] == 0, "verification: unexpected false gap claim")
if verification["counterexample_discovery_rate"] is not None:
    require(perfect(verification["counterexample_discovery_rate"]), "verification: known counterexamples were missed")

print("All four reports passed structural and metric checks.")
PY

echo
echo "MILESTONE 7 SMOKE TEST PASSED"
echo "Reports: $RESULT_DIR"
echo "Note: oracle predictions validate the harness, not live pipeline quality."
