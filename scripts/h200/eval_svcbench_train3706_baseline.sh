#!/usr/bin/env bash
set -euo pipefail

# Independently evaluate the four-epoch Qwen3-VL baseline on the exact production
# A2 train split. This is intentionally a train-set/convergence evaluation.

if [[ $# -gt 2 ]]; then
  echo "usage: bash scripts/h200/eval_svcbench_train3706_baseline.sh [baseline_model] [dataset_manifest.json]" >&2
  exit 2
fi

test "$(id -un)" = niujunbo || {
  echo "refusing to evaluate as $(id -un); expected niujunbo" >&2
  exit 97
}

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
LF_ROOT="${LLAMAFACTORY_ROOT:-$PLAY_ROOT/LLaMA-Factory}"
PREP_VENV="${TTT_PREP_VENV:-${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-py312-torch28}}"
EVAL_VENV="${QWEN_EVAL_VENV:-/mnt/shared-storage-user/mineru2-shared/niujunbo/miniconda3/envs/qwenvl}"
PREP_PYTHON="$PREP_VENV/bin/python"
PYTHON="$EVAL_VENV/bin/python"
LF_CLI="$EVAL_VENV/bin/llamafactory-cli"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-$PLAY_ROOT/datasets/qwensft-data/svcbench-part}"
SOURCE_SFT="${SOURCE_SFT:-$SOURCE_DATASET_DIR/svcbench_qwen3vl_sft.json}"
RAW_ANNOTATIONS="${RAW_ANNOTATIONS:-$SOURCE_DATASET_DIR/raw/data__vcbench_data.jsonl}"
SCORE_ANNOTATIONS="${SCORE_ANNOTATIONS:-$PLAY_ROOT/datasets/SVCBench/vcbench_eval.jsonl}"
SCORER="${SVCBENCH_SCORER:-$PLAY_ROOT/projects/qwen3vl_dist_train/scripts/evaluate_svcbench_predictions.py}"
DEFAULT_MANIFEST="$PROJECT_ROOT/runs/0719_215434_prepare_svcbench_k8/dataset_manifest.json"
MANIFEST="${2:-${SVCBENCH_DATASET_MANIFEST:-$DEFAULT_MANIFEST}}"
BASELINE_MODEL="${1:-${BASELINE_MODEL:-$PLAY_ROOT/projects/qwen3vl_dist_train/outputs/260712_224713_qwen3vl8b_full_svcbench_2fps_4epochs_b2_ga2}}"
EVAL_YAML="${SVCBENCH_EVAL_YAML:-$PROJECT_ROOT/configs/h200/qwen3vl8b_svcbench_train3706_eval_4gpu.yaml}"
EVAL_HF_HOME="${SVCBENCH_EVAL_HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
EXPECTED_SFT_SHA256="${EXPECTED_SFT_SHA256:-aae450f9d82ea067a28c294d2ab8c8dcde99be58c225651546fc62bde5a3d7eb}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)_qwen3vl8b_baseline_train3706_eval}"
SESSION="${SESSION:-qwen3vl8b_baseline_train3706_eval_${RUN_ID}}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/$RUN_ID}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/experiment.log}"
DATASET_OUT="$RUN_ROOT/dataset"
PREDICT_OUT="$RUN_ROOT/predict"
SCORE_OUT="$RUN_ROOT/score"
DATASET_NAME="svcbench_train3706"

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
    "TTT_PREP_VENV=$PREP_VENV"
    "QWEN_EVAL_VENV=$EVAL_VENV"
    "SOURCE_DATASET_DIR=$SOURCE_DATASET_DIR"
    "SOURCE_SFT=$SOURCE_SFT"
    "RAW_ANNOTATIONS=$RAW_ANNOTATIONS"
    "SCORE_ANNOTATIONS=$SCORE_ANNOTATIONS"
    "SVCBENCH_SCORER=$SCORER"
    "SVCBENCH_EVAL_YAML=$EVAL_YAML"
    "SVCBENCH_EVAL_HF_HOME=$EVAL_HF_HOME"
    "EXPECTED_SFT_SHA256=$EXPECTED_SFT_SHA256"
    "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    "RUN_ID=$RUN_ID"
    "SESSION=$SESSION"
    "RUN_ROOT=$RUN_ROOT"
    "LOG_DIR=$LOG_DIR"
    "LOG_FILE=$LOG_FILE"
  )
  for forwarded_name in NCCL_IB_DISABLE NCCL_DEBUG; do
    if [[ -n "${!forwarded_name:-}" ]]; then
      inner_env+=("$forwarded_name=${!forwarded_name}")
    fi
  done
  printf -v inner_command '%q ' "${inner_env[@]}" bash \
    "$PROJECT_ROOT/scripts/h200/eval_svcbench_train3706_baseline.sh" \
    "$BASELINE_MODEL" "$MANIFEST"
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

for path in "$PREP_PYTHON" "$PYTHON" "$SOURCE_SFT" "$RAW_ANNOTATIONS" "$SCORE_ANNOTATIONS" "$SCORER" "$MANIFEST" "$EVAL_YAML"; do
  [[ -e "$path" ]] || { echo "required path missing: $path" >&2; exit 1; }
done
[[ -f "$BASELINE_MODEL/config.json" ]] || {
  echo "baseline model is not evaluation-ready: $BASELINE_MODEL" >&2
  exit 1
}
[[ "$(git -C "$LF_ROOT" rev-parse --short HEAD)" == "523f801" ]] || {
  echo "LLaMA-Factory commit must be 523f801" >&2
  exit 1
}
GPU_COUNT="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
[[ "$GPU_COUNT" -eq 4 ]] || { echo "exactly four visible GPUs are required" >&2; exit 1; }
[[ "$(/usr/bin/nvidia-smi -L | wc -l)" -ge 4 ]] || { echo "four H200 GPUs are not visible" >&2; exit 1; }

export PYTHONNOUSERSITE=1
export PYTHONPATH="$LF_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$EVAL_VENV/bin:$PATH"
export HF_HOME="$EVAL_HF_HOME"
export HF_DATASETS_CACHE="$EVAL_HF_HOME/datasets"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/xdg"
if [[ -x "$LF_CLI" ]]; then
  LF_COMMAND=("$LF_CLI")
else
  "$PYTHON" -c "import llamafactory.cli"
  LF_COMMAND=("$PYTHON" -m llamafactory.cli)
fi
export CUDA_VISIBLE_DEVICES
export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-$("$PYTHON" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)}"
export NPROC_PER_NODE=4
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="/tmp/triton_niujunbo_${RUN_ID}"
mkdir -p "$TRITON_CACHE_DIR" "$SCORE_OUT" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

"$PREP_PYTHON" - "$RUN_ROOT/run_config.json" <<PY
import json
from pathlib import Path
payload = {
    "run_id": "$RUN_ID",
    "method": "qwen3vl_8b_baseline_4epoch",
    "evaluation_scope": "train_set",
    "expected_rows": 3706,
    "model": "$BASELINE_MODEL",
    "manifest": "$MANIFEST",
    "source_sft": "$SOURCE_SFT",
    "score_annotations": "$SCORE_ANNOTATIONS",
    "prep_python": "$PREP_PYTHON",
    "eval_python": "$PYTHON",
    "eval_yaml": "$EVAL_YAML",
    "hf_home": "$HF_HOME",
    "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
}
Path("$RUN_ROOT/run_config.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8"
)
PY

echo "stage=prepare_train3706 method=baseline started_at=$(date -Iseconds)"
PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH" "$PREP_PYTHON" \
  "$PROJECT_ROOT/scripts/prepare_svcbench_train3706_eval.py" \
  --sft-data "$SOURCE_SFT" \
  --raw-annotations "$RAW_ANNOTATIONS" \
  --manifest "$MANIFEST" \
  --output-dir "$DATASET_OUT" \
  --expected-sft-sha256 "$EXPECTED_SFT_SHA256"

START_SECONDS="$(date +%s)"
echo "stage=generate method=baseline model=$BASELINE_MODEL"
cd "$LF_ROOT"
"${LF_COMMAND[@]}" train "$EVAL_YAML" \
  model_name_or_path="$BASELINE_MODEL" \
  dataset_dir="$DATASET_OUT" \
  eval_dataset="$DATASET_NAME" \
  output_dir="$PREDICT_OUT" \
  run_name="$RUN_ID"

PREDICTIONS="$PREDICT_OUT/generated_predictions.jsonl"
[[ -s "$PREDICTIONS" ]] || { echo "missing baseline predictions: $PREDICTIONS" >&2; exit 1; }
PREDICTION_COUNT="$(wc -l < "$PREDICTIONS" | tr -d ' ')"
[[ "$PREDICTION_COUNT" -eq 3706 ]] || {
  echo "baseline prediction count is $PREDICTION_COUNT, expected 3706" >&2
  exit 1
}

echo "stage=score method=baseline scorer=$SCORER"
"$PYTHON" "$SCORER" \
  --predictions "$PREDICTIONS" \
  --sft-data "$DATASET_OUT/$DATASET_NAME.json" \
  --annotations "$SCORE_ANNOTATIONS" \
  --output-dir "$SCORE_OUT" \
  --evaluation-scope train_set \
  --elapsed-seconds "$(( $(date +%s) - START_SECONDS ))"

"$PYTHON" "$PROJECT_ROOT/scripts/stamp_svcbench_eval_metrics.py" \
  --metrics "$SCORE_OUT/metrics.json" \
  --method baseline \
  --selection "$DATASET_OUT/selection.jsonl" \
  --prepared-sft "$DATASET_OUT/$DATASET_NAME.json" \
  --score-annotations "$SCORE_ANNOTATIONS" \
  --scorer "$SCORER" \
  --manifest "$MANIFEST" \
  --evaluation-config "$EVAL_YAML" \
  --model-identity "$BASELINE_MODEL"

echo "status=completed method=baseline score=$SCORE_OUT/metrics.json finished_at=$(date -Iseconds)"
