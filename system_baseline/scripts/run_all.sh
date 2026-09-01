#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

"$PYTHON_BIN" -m system_baseline.scripts.find_project_data
"$PYTHON_BIN" -m system_baseline.scripts.prepare_ed_test \
  --config system_baseline/configs/system_eval.yaml

"$PYTHON_BIN" -m system_baseline.scripts.run_system_generation \
  --config system_baseline/configs/system_eval.yaml \
  --input system_baseline/data/ed_test_canonical.jsonl \
  --output_dir system_baseline/outputs/generations

"$PYTHON_BIN" -m system_baseline.scripts.evaluate_system_metrics \
  --config system_baseline/configs/system_eval.yaml \
  --input system_baseline/data/ed_test_canonical.jsonl \
  --generation_dir system_baseline/outputs/generations \
  --output_dir system_baseline/outputs/metrics

"$PYTHON_BIN" -m system_baseline.scripts.paired_bootstrap \
  --metric all \
  --generation_dir system_baseline/outputs/generations \
  --output system_baseline/outputs/metrics/pairwise_bootstrap.csv

"$PYTHON_BIN" -m system_baseline.scripts.make_latex_tables \
  --metrics system_baseline/outputs/metrics/system_metrics_full.csv \
  --output system_baseline/outputs/metrics/system_metrics_table.tex
