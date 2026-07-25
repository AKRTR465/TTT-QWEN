#!/usr/bin/env bash
set -euo pipefail

# Independently evaluate one A2 checkpoint with static W0. No OnlineTTTUpdater,
# transient fast matrices, or Inner SGD are constructed.

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/h200/eval_svcbench_train3706_a2_static.sh <a2_checkpoint> [dataset_manifest.json]" >&2
  exit 2
fi

test "$(id -un)" = niujunbo || {
  echo "refusing to evaluate as $(id -un); expected niujunbo" >&2
  exit 97
}

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
LF_ROOT="${LLAMAFACTORY_ROOT:-$PLAY_ROOT/LLaMA-Factory}"
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-py312-torch28}"
PYTHON="$VENV/bin/python"
MODEL="${MODEL:-$PLAY_ROOT/model/Qwen3-VL-8B-Instruct}"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-$PLAY_ROOT/datasets/qwensft-data/svcbench-part}"
SOURCE_SFT="${SOURCE_SFT:-$SOURCE_DATASET_DIR/svcbench_qwen3vl_sft.json}"
RAW_ANNOTATIONS="${RAW_ANNOTATIONS:-$SOURCE_DATASET_DIR/raw/data__vcbench_data.jsonl}"
SCORE_ANNOTATIONS="${SCORE_ANNOTATIONS:-$PLAY_ROOT/datasets/SVCBench/vcbench_eval.jsonl}"
SOURCE_VIDEO_ROOT="${SVCBENCH_VIDEO_ROOT:-$PLAY_ROOT/datasets/SVCBench/videos}"
SCORER="${SVCBENCH_SCORER:-$PLAY_ROOT/projects/qwen3vl_dist_train/scripts/evaluate_svcbench_predictions.py}"
DEFAULT_MANIFEST="$PROJECT_ROOT/runs/0719_215434_prepare_svcbench_k8/dataset_manifest.json"
MANIFEST="${2:-${SVCBENCH_DATASET_MANIFEST:-$DEFAULT_MANIFEST}}"
A2_CHECKPOINT="$1"
EVAL_YAML="${A2_EVAL_YAML:-$PROJECT_ROOT/configs/h200/a2_qwen3vl8b_trainsplit_costbalanced_4epoch_4gpu.yaml}"
CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260723_a2_original_trainsplit_support_statequery}"
CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-a2_original_trainsplit_support_statequery_v1}"
VISUAL_COST_INDEX="${VISUAL_COST_INDEX:-$PROJECT_ROOT/artifacts/a2_trainsplit_state16_answer256_ema_cost_index.json}"
EXPECTED_SFT_SHA256="${EXPECTED_SFT_SHA256:-aae450f9d82ea067a28c294d2ab8c8dcde99be58c225651546fc62bde5a3d7eb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)_ttt_a2_static_train3706_eval}"
SESSION="${SESSION:-ttt_a2_static_train3706_eval_${RUN_ID}}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/$RUN_ID}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/experiment.log}"
DATASET_OUT="$RUN_ROOT/dataset"
PREDICT_OUT="$RUN_ROOT/predict"
SCORE_OUT="$RUN_ROOT/score"
SOURCE_DATASET_NAME="svcbench_qwen3vl_sft"
EVAL_DATASET_NAME="svcbench_train3706"

if [[ "${RUN_IN_TMUX:-0}" != "1" ]]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 1; }
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 1
  fi
  mkdir -p "$RUN_ROOT" "$LOG_DIR"
  inner_env=(
    env
    "RUN_IN_TMUX=1"
    "TTT_PROJECT_ROOT=$PROJECT_ROOT"
    "LLAMAFACTORY_ROOT=$LF_ROOT"
    "TTT_H200_VENV=$VENV"
    "MODEL=$MODEL"
    "SOURCE_DATASET_DIR=$SOURCE_DATASET_DIR"
    "SOURCE_SFT=$SOURCE_SFT"
    "RAW_ANNOTATIONS=$RAW_ANNOTATIONS"
    "SCORE_ANNOTATIONS=$SCORE_ANNOTATIONS"
    "SVCBENCH_VIDEO_ROOT=$SOURCE_VIDEO_ROOT"
    "SVCBENCH_SCORER=$SCORER"
    "A2_EVAL_YAML=$EVAL_YAML"
    "TTT_PREPROCESS_CACHE_ROOT=$CACHE_ROOT"
    "TTT_PREPROCESS_CACHE_NAMESPACE=$CACHE_NAMESPACE"
    "VISUAL_COST_INDEX=$VISUAL_COST_INDEX"
    "EXPECTED_SFT_SHA256=$EXPECTED_SFT_SHA256"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "RUN_ID=$RUN_ID"
    "SESSION=$SESSION"
    "RUN_ROOT=$RUN_ROOT"
    "LOG_DIR=$LOG_DIR"
    "LOG_FILE=$LOG_FILE"
  )
  for forwarded_name in OMP_NUM_THREADS NCCL_IB_DISABLE NCCL_DEBUG MASTER_PORT; do
    if [[ -n "${!forwarded_name:-}" ]]; then
      inner_env+=("$forwarded_name=${!forwarded_name}")
    fi
  done
  printf -v inner_command '%q ' "${inner_env[@]}" bash \
    "$PROJECT_ROOT/scripts/h200/eval_svcbench_train3706_a2_static.sh" \
    "$A2_CHECKPOINT" "$MANIFEST"
  printf -v root_q '%q' "$PROJECT_ROOT"
  printf -v log_q '%q' "$LOG_FILE"
  command_text="set -o pipefail; cd $root_q && $inner_command 2>&1 | tee -a $log_q"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "$command_text"
    exit 0
  fi
  tmux new-session -d -s "$SESSION" "$command_text"
  echo "session=$SESSION"
  echo "run_root=$RUN_ROOT"
  echo "log=$LOG_FILE"
  exit 0
fi

for path in "$PYTHON" "$MODEL/config.json" "$SOURCE_SFT" "$RAW_ANNOTATIONS" "$SCORE_ANNOTATIONS" "$SCORER" "$MANIFEST" "$EVAL_YAML" "$CACHE_ROOT"; do
  [[ -e "$path" ]] || { echo "required path missing: $path" >&2; exit 1; }
done
if [[ ! -f "$A2_CHECKPOINT/model.safetensors" && ! -f "$A2_CHECKPOINT/model.safetensors.index.json" ]]; then
  echo "A2 checkpoint has no safetensors model: $A2_CHECKPOINT" >&2
  exit 1
fi
[[ "$(git -C "$LF_ROOT" rev-parse --short HEAD)" == "523f801" ]] || {
  echo "LLaMA-Factory commit must be 523f801" >&2
  exit 1
}
GPU_COUNT="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
[[ "$GPU_COUNT" -eq 4 ]] || { echo "exactly four visible GPUs are required" >&2; exit 1; }
[[ "$(nvidia-smi -L | wc -l)" -ge 4 ]] || { echo "four H200 GPUs are not visible" >&2; exit 1; }

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/src:$LF_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES
export MODEL
export DATASET_DIR="$SOURCE_DATASET_DIR"
export DATASET_NAME="$SOURCE_DATASET_NAME"
export OUTPUT_DIR="$RUN_ROOT/_llamafactory_unused"
export RUN_ROOT
export SVCBENCH_DATASET_MANIFEST="$MANIFEST"
export SVCBENCH_VIDEO_ROOT="$SOURCE_VIDEO_ROOT"
export TTT_PREPROCESS_CACHE_ROOT="$CACHE_ROOT"
export TTT_PREPROCESS_CACHE_NAMESPACE="$CACHE_NAMESPACE"
export VISUAL_COST_INDEX
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="/tmp/triton_niujunbo_${RUN_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$PREDICT_OUT" "$SCORE_OUT"

"$PYTHON" - "$RUN_ROOT/run_config.json" <<PY
import json
from pathlib import Path
payload = {
    "run_id": "$RUN_ID",
    "method": "ttt_a2_static",
    "evaluation_scope": "train_set",
    "expected_rows": 3706,
    "checkpoint": "$A2_CHECKPOINT",
    "base_model": "$MODEL",
    "yaml": "$EVAL_YAML",
    "manifest": "$MANIFEST",
    "cache_root": "$CACHE_ROOT",
    "cache_namespace": "$CACHE_NAMESPACE",
    "inner_sgd": False,
    "transient_fast_state": False,
    "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
}
Path("$RUN_ROOT/run_config.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8"
)
PY

echo "stage=prepare_train3706 method=a2_static started_at=$(date -Iseconds)"
"$PYTHON" "$PROJECT_ROOT/scripts/prepare_svcbench_train3706_eval.py" \
  --sft-data "$SOURCE_SFT" \
  --raw-annotations "$RAW_ANNOTATIONS" \
  --manifest "$MANIFEST" \
  --output-dir "$DATASET_OUT" \
  --source-dataset-name "$SOURCE_DATASET_NAME" \
  --output-dataset-name "$EVAL_DATASET_NAME" \
  --expected-sft-sha256 "$EXPECTED_SFT_SHA256"

MASTER_PORT="${MASTER_PORT:-$("$PYTHON" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)}"
START_SECONDS="$(date +%s)"
echo "stage=generate method=a2_static checkpoint=$A2_CHECKPOINT master_port=$MASTER_PORT"
cd "$PROJECT_ROOT"
"$PYTHON" -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  -m ttt_svcbench_qwen.a2_static_eval \
  --yaml "$EVAL_YAML" \
  --checkpoint "$A2_CHECKPOINT" \
  --dataset-manifest "$MANIFEST" \
  --selection "$DATASET_OUT/selection.jsonl" \
  --sft-data "$DATASET_OUT/$EVAL_DATASET_NAME.json" \
  --output-dir "$PREDICT_OUT" \
  --llamafactory-root "$LF_ROOT" \
  --max-new-tokens 8

PREDICTIONS="$PREDICT_OUT/generated_predictions.jsonl"
[[ -s "$PREDICTIONS" ]] || { echo "missing A2-static predictions: $PREDICTIONS" >&2; exit 1; }
PREDICTION_COUNT="$(wc -l < "$PREDICTIONS" | tr -d ' ')"
[[ "$PREDICTION_COUNT" -eq 3706 ]] || {
  echo "A2-static prediction count is $PREDICTION_COUNT, expected 3706" >&2
  exit 1
}

echo "stage=score method=a2_static scorer=$SCORER"
"$PYTHON" "$SCORER" \
  --predictions "$PREDICTIONS" \
  --sft-data "$DATASET_OUT/$EVAL_DATASET_NAME.json" \
  --annotations "$SCORE_ANNOTATIONS" \
  --output-dir "$SCORE_OUT" \
  --evaluation-scope train_set \
  --elapsed-seconds "$(( $(date +%s) - START_SECONDS ))"

"$PYTHON" "$PROJECT_ROOT/scripts/stamp_svcbench_eval_metrics.py" \
  --metrics "$SCORE_OUT/metrics.json" \
  --method a2_static \
  --selection "$DATASET_OUT/selection.jsonl" \
  --prepared-sft "$DATASET_OUT/$EVAL_DATASET_NAME.json" \
  --score-annotations "$SCORE_ANNOTATIONS" \
  --scorer "$SCORER" \
  --manifest "$MANIFEST" \
  --evaluation-config "$EVAL_YAML" \
  --model-identity "$A2_CHECKPOINT"

echo "status=completed method=a2_static score=$SCORE_OUT/metrics.json finished_at=$(date -Iseconds)"
