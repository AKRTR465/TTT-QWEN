#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]] || [[ "$1" != "a2" && "$1" != "a5" ]]; then
  echo "usage: bash scripts/h200/train_fullprefix256.sh a2 [manifest]" >&2
  echo "   or: bash scripts/h200/train_fullprefix256.sh a5 <a2_checkpoint> [manifest]" >&2
  exit 2
fi

STAGE="$1"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-/mnt/shared-storage-user/mineru2-shared/niujunbo/play/projects/ttt_qwen}"
if [[ "$STAGE" == "a2" ]]; then
  YAML="${YAML:-$PROJECT_ROOT/configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml}"
  TASK_NAME="qwen3vl8b_ttt_a2_fullprefix256_4h200"
  DEFAULT_VISUAL_COST_INDEX="$PROJECT_ROOT/artifacts/a2_fullprefix256_visual_cost_index.json"
  export VISUAL_COST_INDEX="${VISUAL_COST_INDEX:-$DEFAULT_VISUAL_COST_INDEX}"
  if [[ ! -f "$VISUAL_COST_INDEX" && "${TTT_VISUAL_COST_PREFLIGHT:-0}" != "1" ]]; then
    echo "measured schema-3 visual cost index not found: $VISUAL_COST_INDEX" >&2
    exit 1
  fi
else
  YAML="${YAML:-$PROJECT_ROOT/configs/h200/a5_meta_ttt_k8_fullprefix256_4gpu.yaml}"
  TASK_NAME="qwen3vl8b_ttt_a5_k8_fullprefix256_4h200"
fi

export YAML
export TTT_SKIP_ENV_SETUP="${TTT_SKIP_ENV_SETUP:-1}"
export TTT_QUERY_ACTIVATION_OFFLOAD="${TTT_QUERY_ACTIVATION_OFFLOAD:-0}"
export RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_${TASK_NAME}}"
export SESSION="${SESSION:-${TASK_NAME}_${RUN_ID}}"

exec bash "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" "$@"
