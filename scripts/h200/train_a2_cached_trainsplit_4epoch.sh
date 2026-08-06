#!/usr/bin/env bash
set -euo pipefail

# Production A2 recipe for the currently prepared SVCBench train split.
# Cached: Support + State Query. Real-time decode: 256-frame Answer Query.
# Persistent checkpoints: epoch-2-checkpoint and epoch-4-checkpoint only.

if [[ $# -gt 1 ]]; then
  echo "usage: bash scripts/h200/train_a2_cached_trainsplit_4epoch.sh [dataset_manifest.json]" >&2
  exit 2
fi

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
DEFAULT_MANIFEST="$PROJECT_ROOT/runs/0719_215434_prepare_svcbench_k8/dataset_manifest.json"
DEFAULT_CACHE_ROOT="$PROJECT_ROOT/.cache/preprocess/260720_ttt8_benchmark"

export TTT_PROJECT_ROOT="$PROJECT_ROOT"
export YAML="$PROJECT_ROOT/configs/h200/a2_qwen3vl8b_trainsplit_costbalanced_4epoch_4gpu.yaml"
export SVCBENCH_DATASET_MANIFEST="${SVCBENCH_DATASET_MANIFEST:-${1:-$DEFAULT_MANIFEST}}"
export TTT_PREPROCESS_CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$DEFAULT_CACHE_ROOT}"
export VISUAL_COST_INDEX="${VISUAL_COST_INDEX:-$PROJECT_ROOT/artifacts/a2_trainsplit_state16_answer256_ema_cost_index.json}"
export TTT_CHECKPOINT_POLICY="epoch_2_and_epoch_4"
export RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)_a2_cached_trainsplit_4epoch_epoch2_epoch4}"
export SESSION="${TTT_TRAIN_SESSION:-a2_cached_trainsplit_4epoch_${RUN_ID}}"

if [[ ! -f "$SVCBENCH_DATASET_MANIFEST" ]]; then
  echo "dataset manifest not found: $SVCBENCH_DATASET_MANIFEST" >&2
  exit 1
fi
if [[ ! -d "$TTT_PREPROCESS_CACHE_ROOT" ]]; then
  echo "prepared SVCBench train cache root not found: $TTT_PREPROCESS_CACHE_ROOT" >&2
  exit 1
fi
if [[ ! -f "$VISUAL_COST_INDEX" ]]; then
  echo "building the current schema-4 State16/Answer256 visual cost index"
  bash "$PROJECT_ROOT/scripts/h200/prepare_a2_semantic_visual_cost.sh" \
    "$SVCBENCH_DATASET_MANIFEST" \
    "$VISUAL_COST_INDEX"
fi

exec bash "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" a2 "$SVCBENCH_DATASET_MANIFEST"
