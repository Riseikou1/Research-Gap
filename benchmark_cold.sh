#!/usr/bin/env bash

set -uo pipefail

PYTHON="${PYTHON:-python3}"
RESULT_DIR="benchmark_results"
CACHE_DIR="${RESEARCH_GAP_CACHE_DIR:-.cache/research-gap}"

mkdir -p "$RESULT_DIR"

IDEAS=(
    "molecular property prediction using graph neural networks with limited labeled data"
    "causal inference from observational healthcare data under unmeasured confounding"
    "federated reinforcement learning for adaptive traffic signal control across multiple cities under non-IID data and limited communication"
)

clear_project_cache() {
    echo
    echo "Clearing cache: $CACHE_DIR"

    rm -rf "$CACHE_DIR"
    mkdir -p "$CACHE_DIR"
}

run_benchmark() {
    local number="$1"
    local idea="$2"
    local output="$RESULT_DIR/full_cold_${number}.txt"

    clear_project_cache

    echo
    echo "============================================================"
    echo "FULL COLD BENCHMARK $number"
    echo "============================================================"
    echo "Idea: $idea"
    echo

    {
        echo "Benchmark: $number"
        echo "Idea: $idea"
        echo "Started: $(date --iso-8601=seconds)"
        echo

        /usr/bin/time \
            -f $'\n=== PROCESS TIMING ===\nWall time: %e s\nCPU user: %U s\nCPU system: %S s\nMax RSS: %M KB\nExit code: %x' \
            "$PYTHON" main.py \
            "$idea" \
            --decomposer openai \
            --query-generator openai \
            --show-gaps

        status=$?

        echo
        echo "Finished: $(date --iso-8601=seconds)"
        echo "Program exit status: $status"

        exit "$status"
    } 2>&1 | tee "$output"

    local status=${PIPESTATUS[0]}

    if [[ "$status" -ne 0 ]]; then
        echo "Benchmark $number FAILED."
        return "$status"
    fi
}

for i in "${!IDEAS[@]}"; do
    run_benchmark "$((i + 1))" "${IDEAS[$i]}"
done

echo
echo "============================================================"
echo "COMPLETE"
echo "============================================================"

grep -Eini \
    'planning|retrieval|evidence|extract|verification|cache|provider|request|pipeline|duration|timing|total' \
    "$RESULT_DIR"/full_cold_*.txt \
    || true