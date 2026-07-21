#!/usr/bin/env bash
set -euo pipefail

# Train-split-only SVCBench A2 recipe. This script is intentionally inert until
# invoked. It consumes only the canonical fold's existing train split, so every
# Support observation is covered by the prepared train-Support cache.

if [[ $# -gt 1 ]]; then
  echo "usage: bash scripts/h200/train_a2_allsvcbench_4epoch.sh [dataset_manifest.json]" >&2
  exit 2
fi

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
DEFAULT_MANIFEST="$PROJECT_ROOT/runs/0719_215434_prepare_svcbench_k8/dataset_manifest.json"
DEFAULT_CACHE_ROOT="$PROJECT_ROOT/.cache/preprocess/260720_ttt8_benchmark"

export TTT_PROJECT_ROOT="$PROJECT_ROOT"
export YAML="${YAML:-$PROJECT_ROOT/configs/h200/a2_qwen3vl8b_trainsplit_costbalanced_4epoch_4gpu.yaml}"
export SVCBENCH_DATASET_MANIFEST="${SVCBENCH_DATASET_MANIFEST:-${1:-$DEFAULT_MANIFEST}}"
export TTT_PREPROCESS_CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"
export VISUAL_COST_INDEX="${VISUAL_COST_INDEX:-$PROJECT_ROOT/artifacts/a2_trainsplit_state16_answer256_ema_cost_index.json}"
export RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)_a2_trainsplit_4epoch_state16_answer256_ema}"
export SESSION="${SESSION:-a2_trainsplit_4epoch_${RUN_ID}}"

if [[ ! -f "$SVCBENCH_DATASET_MANIFEST" ]]; then
  echo "dataset manifest not found: $SVCBENCH_DATASET_MANIFEST" >&2
  exit 1
fi
if [[ ! -d "$TTT_PREPROCESS_CACHE_ROOT" ]]; then
  echo "SVCBench Support cache root not found: $TTT_PREPROCESS_CACHE_ROOT" >&2
  exit 1
fi
if [[ ! -f "$VISUAL_COST_INDEX" ]]; then
  echo "schema-4 State16/Answer256 visual cost index not found: $VISUAL_COST_INDEX" >&2
  echo "run: bash scripts/h200/prepare_a2_semantic_visual_cost.sh" >&2
  exit 1
fi

exec bash "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" a2 "$SVCBENCH_DATASET_MANIFEST"
