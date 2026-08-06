#!/usr/bin/env bash
set -euo pipefail

# Evaluate one A5 warmup bundle on the fixed 3,706-row SVCBench train-set selection, with the
# per-video slot memory live: every Support chunk writes into M and the Answer Query reads M_T.
# The counterpart of eval_svcbench_train3706_a2_static.sh, which deliberately forbids the memory
# path, and it reuses that script's dataset preparation and scorer so the two numbers land on the
# identical rows and are directly comparable.

if [[ $# -lt 2 || $# -gt 3 ]]; then
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/eval_svcbench_train3706_a5.sh \
    <a2_checkpoint> <a5_warmup_bundle> [dataset_manifest.json]

The bundle overlay is provenance-gated: it must come from a full 256-step warmup started from
this same A2 checkpoint, the same dataset manifest, the same project config and seeds, and the
working tree must be clean.  Only code_commit is exempted (adding an evaluator is itself a
commit), and the exemption is recorded in the run's warmup_bundle_audit.json.
EOF
  exit 2
fi

test "$(id -un)" = niujunbo || {
  echo "refusing to evaluate as $(id -un); expected niujunbo" >&2
  exit 97
}

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
LF_ROOT="${LLAMAFACTORY_ROOT:-$PLAY_ROOT/LLaMA-Factory}"
# The A2/eval launchers default to a venv that does not exist; use the real one.
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-uv-py312-torch28}"
PYTHON="$VENV/bin/python"
MODEL="${MODEL:-$PLAY_ROOT/model/Qwen3-VL-8B-Instruct}"
SOURCE_DATASET_DIR="${SOURCE_DATASET_DIR:-$PLAY_ROOT/datasets/qwensft-data/svcbench-part}"
SOURCE_DATASET_NAME="${SOURCE_DATASET_NAME:-svcbench_qwen3vl_sft}"
SOURCE_SFT="${SOURCE_SFT:-$SOURCE_DATASET_DIR/$SOURCE_DATASET_NAME.json}"
RAW_ANNOTATIONS="${RAW_ANNOTATIONS:-$SOURCE_DATASET_DIR/raw/data__vcbench_data.jsonl}"
SCORE_ANNOTATIONS="${SCORE_ANNOTATIONS:-$PLAY_ROOT/datasets/SVCBench/vcbench_eval.jsonl}"
SOURCE_VIDEO_ROOT="${SVCBENCH_VIDEO_ROOT:-$PLAY_ROOT/datasets/SVCBench/videos}"
SCORER="${SVCBENCH_SCORER:-$PLAY_ROOT/projects/qwen3vl_dist_train/scripts/evaluate_svcbench_predictions.py}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-svcbench_train3706}"
EXPECTED_SFT_SHA256="${EXPECTED_SFT_SHA256:-aae450f9d82ea067a28c294d2ab8c8dcde99be58c225651546fc62bde5a3d7eb}"

A2_CHECKPOINT="$1"
WARMUP_BUNDLE="$2"
DEFAULT_MANIFEST="$PROJECT_ROOT/runs/260805_234502_prepare_svcbench_k8_w4/dataset_manifest.json"
MANIFEST="${3:-${SVCBENCH_DATASET_MANIFEST:-$DEFAULT_MANIFEST}}"
# The warmup YAML, not the A2 eval YAML: the bundle's parameter_allowlist and provenance are
# computed from the backbone this builds, so it has to be the config the bundle was trained under.
EVAL_YAML="${A5_EVAL_YAML:-$PROJECT_ROOT/configs/h200/a5_fast_state_warmup_256_4gpu.yaml}"
CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260726_a5_dense_querybundle_v4_fp16}"
CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-a5_dense_querybundle_train_support_statequery_fp16_v4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)_a5_bundle_train3706_eval}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/$RUN_ID}"
DATASET_OUT="$RUN_ROOT/dataset"
PREDICT_OUT="$RUN_ROOT/predict"
SCORE_OUT="$RUN_ROOT/score"

for path in "$PYTHON" "$MODEL/config.json" "$SOURCE_SFT" "$RAW_ANNOTATIONS" \
            "$SCORE_ANNOTATIONS" "$SCORER" "$MANIFEST" "$EVAL_YAML" "$CACHE_ROOT"; do
  [[ -e "$path" ]] || { echo "required path missing: $path" >&2; exit 1; }
done
if [[ ! -f "$A2_CHECKPOINT/model.safetensors" && ! -f "$A2_CHECKPOINT/model.safetensors.index.json" ]]; then
  echo "A2 checkpoint has no safetensors model: $A2_CHECKPOINT" >&2
  exit 1
fi
for required in model.safetensors manifest.json; do
  [[ -f "$WARMUP_BUNDLE/$required" ]] || {
    echo "warmup bundle is missing $required: $WARMUP_BUNDLE" >&2
    exit 1
  }
done
# The overlay recomputes the source manifest, which refuses a dirty tree outright.  Fail here
# with a clear message instead of deep inside the model build.
[[ -z "$(git -C "$PROJECT_ROOT" status --porcelain)" ]] || {
  echo "working tree must be clean to load a warmup bundle: $PROJECT_ROOT" >&2
  exit 1
}
[[ "$(git -C "$LF_ROOT" rev-parse --short HEAD)" == "523f801" ]] || {
  echo "LLaMA-Factory commit must be 523f801" >&2
  exit 1
}
GPU_COUNT="$(printf '%s' "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')"
[[ "$GPU_COUNT" -eq 4 ]] || { echo "exactly four visible GPUs are required" >&2; exit 1; }
[[ "$(/usr/bin/nvidia-smi -L | wc -l)" -ge 4 ]] || { echo "four H200 GPUs are not visible" >&2; exit 1; }

mkdir -p "$LOG_DIR" "$DATASET_OUT" "$PREDICT_OUT" "$SCORE_OUT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT/src:$LF_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES
export MODEL
export YAML="$EVAL_YAML"
export TTT_PROJECT_CONFIG="${TTT_PROJECT_CONFIG:-$PROJECT_ROOT/configs/model_state_ttt_8b.yaml}"
export TTT_A5_ADAPTATION_MODE="meta_ttt"
# The same two env vars train_a2_a5.sh uses to bind provenance into the ttt_config.
export A2_CHECKPOINT
export SVCBENCH_DATASET_MANIFEST="$MANIFEST"
export DATASET_DIR="$SOURCE_DATASET_DIR"
export DATASET_NAME="$SOURCE_DATASET_NAME"
export OUTPUT_DIR="$RUN_ROOT/_llamafactory_unused"
export RUN_ROOT
export SVCBENCH_VIDEO_ROOT="$SOURCE_VIDEO_ROOT"
export TTT_PREPROCESS_CACHE_ROOT="$CACHE_ROOT"
export TTT_PREPROCESS_CACHE_NAMESPACE="$CACHE_NAMESPACE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="/tmp/triton_niujunbo_${RUN_ID}"
mkdir -p "$TRITON_CACHE_DIR"

echo "stage=prepare_train3706 method=a5_bundle started_at=$(date -Iseconds)"
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
echo "stage=generate method=a5_bundle bundle=$WARMUP_BUNDLE master_port=$MASTER_PORT"
cd "$PROJECT_ROOT"
"$PYTHON" -m torch.distributed.run \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  -m ttt_svcbench_qwen.a5_eval \
  --yaml "$EVAL_YAML" \
  --checkpoint "$A2_CHECKPOINT" \
  --warmup-bundle "$WARMUP_BUNDLE" \
  --dataset-manifest "$MANIFEST" \
  --selection "$DATASET_OUT/selection.jsonl" \
  --sft-data "$DATASET_OUT/$EVAL_DATASET_NAME.json" \
  --output-dir "$PREDICT_OUT" \
  --llamafactory-root "$LF_ROOT" \
  --max-new-tokens 8

PREDICTIONS="$PREDICT_OUT/generated_predictions.jsonl"
[[ -s "$PREDICTIONS" ]] || { echo "missing A5 predictions: $PREDICTIONS" >&2; exit 1; }
PREDICTION_COUNT="$(wc -l < "$PREDICTIONS" | tr -d ' ')"
[[ "$PREDICTION_COUNT" -eq 3706 ]] || {
  echo "A5 prediction count is $PREDICTION_COUNT, expected 3706" >&2
  exit 1
}

echo "stage=score method=a5_bundle scorer=$SCORER"
"$PYTHON" "$SCORER" \
  --predictions "$PREDICTIONS" \
  --sft-data "$DATASET_OUT/$EVAL_DATASET_NAME.json" \
  --annotations "$SCORE_ANNOTATIONS" \
  --output-dir "$SCORE_OUT" \
  --evaluation-scope train_set \
  --elapsed-seconds "$(( $(date +%s) - START_SECONDS ))"

"$PYTHON" "$PROJECT_ROOT/scripts/stamp_svcbench_eval_metrics.py" \
  --metrics "$SCORE_OUT/metrics.json" \
  --method a5_bundle_memory \
  --selection "$DATASET_OUT/selection.jsonl" \
  --prepared-sft "$DATASET_OUT/$EVAL_DATASET_NAME.json" \
  --score-annotations "$SCORE_ANNOTATIONS" \
  --scorer "$SCORER" \
  --manifest "$MANIFEST" \
  --evaluation-config "$EVAL_YAML" \
  --model-identity "$WARMUP_BUNDLE"

echo "status=completed method=a5_bundle score=$SCORE_OUT/metrics.json finished_at=$(date -Iseconds)"
