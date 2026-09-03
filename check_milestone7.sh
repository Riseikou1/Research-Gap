#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python interpreter not found: ${PYTHON}" >&2
    echo "Create the project virtual environment first." >&2
    exit 1
fi

cd "${ROOT}"

echo "[1/4] Running full test suite"
"${PYTHON}" -m unittest discover -s tests -v

echo "[2/4] Compiling project Python files"
"${PYTHON}" -m py_compile main.py $(find src -name '*.py' -type f)

echo "[3/4] Checking patch formatting"
git diff --check

echo "[4/4] Running the real Milestone 7 smoke case"
"${PYTHON}" main.py \
    "molecular property prediction using graph neural networks with limited labeled data" \
    --decomposer openai \
    --query-generator openai \
    --show-gaps \
    --show-landscape

echo
echo "Milestone 7 correctness checks passed."
