#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: bash scripts/h200/train_a2_baselineclips_4epoch.sh" >&2
  exit 2
fi

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
MODEL="${MODEL:-$PLAY_ROOT/model/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-$PLAY_ROOT/datasets/qwensft-data/svcbench-part}"
DATASET_NAME="svcbench_qwen3vl_sft"
WEAK_SIDECAR_PATH="${WEAK_SIDECAR_PATH:-$DATASET_DIR/raw/data__vcbench_data.jsonl}"
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-py312-torch28}"
PYTHON="$VENV/bin/python"
CONFIG="$PROJECT_ROOT/configs/h200/a2_qwen3vl8b_baselineclips_4epoch_4gpu.yaml"
CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260723_a2_baselineclips_statequery_only}"
CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-a2_baselineclips_statequery_v1}"

[[ "$(id -un)" == "niujunbo" ]] || { echo "refusing non-niujunbo shell" >&2; exit 97; }
[[ -x "$PYTHON" ]] || { echo "existing H200 Python not found: $PYTHON" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "Qwen3-VL-8B model not found: $MODEL" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "training config not found: $CONFIG" >&2; exit 2; }
[[ -f "$WEAK_SIDECAR_PATH" ]] || { echo "official-weak sidecar not found" >&2; exit 2; }
[[ -d "$CACHE_ROOT" ]] || {
  echo "baseline-clips State cache is absent; run prewarm_a2_baselineclips_state_cache.sh" >&2
  exit 2
}

expected_sha="aae450f9d82ea067a28c294d2ab8c8dcde99be58c225651546fc62bde5a3d7eb"
actual_sha="$(sha256sum "$DATASET_DIR/svcbench_qwen3vl_sft.json" | awk '{print $1}')"
[[ "$actual_sha" == "$expected_sha" ]] || {
  echo "baseline JSON SHA256 drift: $actual_sha" >&2
  exit 2
}

export PYTHONPATH="$PROJECT_ROOT/src:$PLAY_ROOT/LLaMA-Factory/src${PYTHONPATH:+:$PYTHONPATH}"
export SVCBENCH_VIDEO_ROOT="$DATASET_DIR"
export DATASET_DIR DATASET_NAME WEAK_SIDECAR_PATH
export MODEL
export TTT_PROJECT_ROOT="$PROJECT_ROOT"
export TTT_H200_PLAY_ROOT="$PLAY_ROOT"
export TTT_A2_DATA_MODE="llamafactory_sft_clips"
export TTT_PREPROCESS_CACHE_ROOT="$CACHE_ROOT"
export TTT_PREPROCESS_CACHE_NAMESPACE="$CACHE_NAMESPACE"
export TTT_CHECKPOINT_POLICY="${TTT_CHECKPOINT_POLICY:-epoch_2_and_epoch_4}"
export TTT_SMOKE_SHORTEST_FIRST="0"
export YAML="$CONFIG"
export RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)_a2_baselineclips_4epoch_epoch2_epoch4}"

"$PYTHON" "$PROJECT_ROOT/scripts/preprocess_cache.py" verify-inputs \
  --root "$CACHE_ROOT" \
  --max-gb 200 \
  --namespace "$CACHE_NAMESPACE" \
  --dataset-dir "$DATASET_DIR" \
  --dataset "$DATASET_NAME" \
  --weak-sidecar "$WEAK_SIDECAR_PATH" \
  --project-config "$PROJECT_ROOT/configs/model_state_ttt_8b.yaml" \
  --training-config "$CONFIG" \
  --video-root "$DATASET_DIR" \
  --stage a2 \
  --minimum-pixels 256 \
  --maximum-pixels 131072 \
  --split train \
  --roles state_query

echo "dataset=$DATASET_DIR/svcbench_qwen3vl_sft.json rows=4576 sha256=$actual_sha"
echo "video_root=$SVCBENCH_VIDEO_ROOT"
echo "cache_root=$TTT_PREPROCESS_CACHE_ROOT namespace=$CACHE_NAMESPACE"
echo "git_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
git -C "$PROJECT_ROOT" status --short

exec bash "$PROJECT_ROOT/scripts/h200/launch_4gpu.sh" a2
