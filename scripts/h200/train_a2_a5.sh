#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/train_a2_a5.sh a2 [dataset_manifest.json]
  bash scripts/h200/train_a2_a5.sh a5 <a2_final_checkpoint> [dataset_manifest.json]
  bash scripts/h200/launch_qwen3vl8b_ttt_a2_full4.sh [dataset_manifest.json]
  bash scripts/h200/launch_qwen3vl8b_ttt_a5_k8_full4.sh <a2_final_checkpoint> [dataset_manifest.json]

If dataset_manifest.json is omitted, the script builds the fixed fold0/K=8 manifest from the
existing H200 SVCBench conversion. Environment overrides: TTT_PROJECT_ROOT,
SVCBENCH_DATASET_ROOT, SVCBENCH_DATASET_MANIFEST, RUN_ID, SESSION,
CUDA_VISIBLE_DEVICES, TTT_RESUME_CHECKPOINT, TTT_H200_VENV, TTT_PREPROCESS_CACHE_ROOT,
TTT_DATALOADER_TRACE. The outer invocation starts a detached tmux;
set RUN_IN_TMUX=1 only when already running inside the intended tmux pane. One-step validation may
set TTT_SMOKE_MAX_STEPS=1, TTT_SKIP_FINAL_CHECKPOINT=1, and TTT_SMOKE_SHORTEST_FIRST=1.
EOF
  exit 2
}

[[ $# -ge 1 ]] || usage
STAGE="$1"
shift
[[ "$STAGE" == "a2" || "$STAGE" == "a5" ]] || usage

EXPECTED_USER="niujunbo"
PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
MODEL="${MODEL:-$PLAY_ROOT/model/Qwen3-VL-8B-Instruct}"
DATASET_ROOT="${DATASET_DIR:-${SVCBENCH_DATASET_ROOT:-$PLAY_ROOT/datasets/qwensft-data/svcbench-part}}"
DATASET_DIR="$DATASET_ROOT"
DATASET_NAME="${DATASET_NAME:-svcbench_qwen3vl_sft}"
SOURCE_VIDEO_ROOT="${SVCBENCH_VIDEO_ROOT:-$PLAY_ROOT/datasets/SVCBench/videos}"
ANNOTATION="$DATASET_ROOT/raw/data__vcbench_data.jsonl"
CONVERTED_DATASET="$DATASET_ROOT/$DATASET_NAME.json"
BOOTSTRAP_PYTHON="${TTT_BOOTSTRAP_PYTHON:-/mnt/shared-storage-user/mineru2-shared/niujunbo/miniconda3/envs/openclaw-rl/bin/python3.12}"
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-py312-torch28}"
PYTHON="$VENV/bin/python"
H200_TORCH_VERSION="2.8.0+cu128"
H200_TORCHVISION_VERSION="0.23.0+cu128"
H200_TORCHAUDIO_VERSION="2.8.0+cu128"
PYTORCH_INDEX_URL="${TTT_PYTORCH_INDEX_URL:-http://pypi.i.h.pjlab.org.cn/brain/dev/+simple}"
PYPI_INDEX_URL="${TTT_PYPI_INDEX_URL:-http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/}"

if [[ "$(id -un)" != "$EXPECTED_USER" ]]; then
  echo "refusing to train as $(id -un); expected $EXPECTED_USER" >&2
  exit 1
fi
if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
  echo "project checkout not found: $PROJECT_ROOT" >&2
  exit 1
fi
for path in "$ANNOTATION" "$CONVERTED_DATASET"; do
  if [[ ! -f "$path" ]]; then
    echo "required H200 SVCBench file not found: $path" >&2
    exit 1
  fi
done
if [[ ! -d "$SOURCE_VIDEO_ROOT" ]]; then
  echo "original SVCBench video root not found: $SOURCE_VIDEO_ROOT" >&2
  exit 1
fi
if [[ ! -d "$PLAY_ROOT/LLaMA-Factory/.git" ]]; then
  echo "LLaMA-Factory checkout not found: $PLAY_ROOT/LLaMA-Factory" >&2
  exit 1
fi
if [[ ! -d "$MODEL" ]]; then
  echo "Qwen3-VL checkpoint not found: $MODEL" >&2
  exit 1
fi

MANIFEST="${SVCBENCH_DATASET_MANIFEST:-}"
if [[ "$STAGE" == "a2" ]]; then
  if [[ $# -gt 1 ]]; then
    usage
  fi
  if [[ $# -eq 1 ]]; then
    MANIFEST="$1"
  fi
else
  A2_CHECKPOINT="${A2_CHECKPOINT:-}"
  if [[ $# -ge 1 ]]; then
    A2_CHECKPOINT="$1"
    shift
  fi
  [[ $# -le 1 ]] || usage
  if [[ $# -eq 1 ]]; then
    MANIFEST="$1"
  fi
  : "${A2_CHECKPOINT:?A5 requires the final A2 checkpoint as its first argument}"
  export A2_CHECKPOINT
fi

if [[ "$STAGE" == "a2" ]]; then
  TASK_NAME="qwen3vl8b_ttt_a2_full4"
  # Keep the default long-run profile below the observed 136 GB/card ZeRO-1
  # high-pixel peak. Visual tokens remain dynamic within each current chunk;
  # callers may explicitly select the 120g profile through YAML=... for stress tests.
  YAML="${YAML:-$PROJECT_ROOT/configs/h200/a2_qwen3vl8b_full_4gpu.yaml}"
else
  TASK_NAME="qwen3vl8b_ttt_a5_k8_full4"
  YAML="${YAML:-$PROJECT_ROOT/configs/h200/a5_meta_ttt_k8_4gpu.yaml}"
fi
RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_${TASK_NAME}}"
SESSION="${SESSION:-${TASK_NAME}_${RUN_ID}}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/$RUN_ID}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/experiment.log}"
export MODEL DATASET_DIR DATASET_NAME RUN_ID SESSION RUN_ROOT LOG_DIR LOG_FILE YAML

if [[ "${RUN_IN_TMUX:-0}" != "1" ]]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux is required on the H200 worker" >&2; exit 1; }
  mkdir -p "$LOG_DIR"
  inner_env=(
    env
    "RUN_IN_TMUX=1"
    "TTT_PROJECT_ROOT=$PROJECT_ROOT"
    "MODEL=$MODEL"
    "DATASET_DIR=$DATASET_DIR"
    "DATASET_NAME=$DATASET_NAME"
    "SVCBENCH_VIDEO_ROOT=$SOURCE_VIDEO_ROOT"
    "RUN_ID=$RUN_ID"
    "SESSION=$SESSION"
    "RUN_ROOT=$RUN_ROOT"
    "LOG_DIR=$LOG_DIR"
    "LOG_FILE=$LOG_FILE"
    "YAML=$YAML"
    "TTT_H200_VENV=$VENV"
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  )
  if [[ -n "$MANIFEST" ]]; then
    inner_env+=("SVCBENCH_DATASET_MANIFEST=$MANIFEST")
  fi
  if [[ "$STAGE" == "a5" ]]; then
    inner_env+=("A2_CHECKPOINT=$A2_CHECKPOINT")
  fi
  if [[ -n "${TTT_RESUME_CHECKPOINT:-}" ]]; then
    inner_env+=("TTT_RESUME_CHECKPOINT=$TTT_RESUME_CHECKPOINT")
  fi
  if [[ -n "${TTT_PREFLIGHT_ONLY:-}" ]]; then
    inner_env+=("TTT_PREFLIGHT_ONLY=$TTT_PREFLIGHT_ONLY")
  fi
  for forwarded_name in \
    TTT_SMOKE_MAX_STEPS \
    TTT_SKIP_FINAL_CHECKPOINT \
    TTT_CHECKPOINT_POLICY \
    TTT_SMOKE_SHORTEST_FIRST \
    TTT_RUN_TIMEOUT_SECONDS \
    TTT_A2_PROGRESS_TRACE \
    TTT_PREPROCESS_CACHE_ROOT \
    TTT_DATALOADER_TRACE \
    TTT_VISUAL_COST_PREFLIGHT \
    VISUAL_COST_INDEX \
    TTT_A2_SUPPORT_PREFETCH \
    TTT_SUPPORT_VISUAL_BATCH_SIZE \
    TTT_SKIP_ENV_SETUP \
    TTT_QUERY_ACTIVATION_OFFLOAD \
    NCCL_DEBUG \
    NCCL_DEBUG_SUBSYS \
    TORCH_DISTRIBUTED_DEBUG \
    TORCH_NCCL_TRACE_BUFFER_SIZE \
    TORCH_NCCL_DUMP_ON_TIMEOUT \
    TORCH_NCCL_DESYNC_DEBUG \
    TORCH_NCCL_ENABLE_MONITORING \
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC; do
    if [[ -n "${!forwarded_name:-}" ]]; then
      inner_env+=("$forwarded_name=${!forwarded_name}")
    fi
  done

  printf -v inner_command '%q ' "${inner_env[@]}" bash \
    "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" "$STAGE"
  printf -v project_q '%q' "$PROJECT_ROOT"
  printf -v log_q '%q' "$LOG_FILE"
  tmux_command="set -o pipefail; cd $project_q && $inner_command 2>&1 | tee -a $log_q"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1, tmux command would be:"
    printf '%s\n' "$tmux_command"
    exit 0
  fi

  tmux new-session -d -s "$SESSION" "$tmux_command"
  echo "session=$SESSION"
  echo "run_root=$RUN_ROOT"
  echo "log=$LOG_FILE"
  echo "tail -f $LOG_FILE"
  exit 0
fi

cd "$PROJECT_ROOT"
if [[ "${TTT_SKIP_ENV_SETUP:-0}" == "1" && ! -x "$PYTHON" ]]; then
  echo "TTT_SKIP_ENV_SETUP=1 requires an existing project Python: $PYTHON" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  if [[ ! -x "$BOOTSTRAP_PYTHON" ]]; then
    echo "Python 3.12 bootstrap interpreter not found: $BOOTSTRAP_PYTHON" >&2
    exit 1
  fi
  "$BOOTSTRAP_PYTHON" -m venv --copies "$VENV"
fi
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL="$PYPI_INDEX_URL"
export PIP_EXTRA_INDEX_URL=
export PIP_TRUSTED_HOST="mirrors.i.h.pjlab.org.cn pypi.i.h.pjlab.org.cn"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PLAY_ROOT/.cache/pip}"
mkdir -p "$PIP_CACHE_DIR"

# Qwen3-VL contains Conv3D modules. LLaMA-Factory rejects torch 2.9.x for this
# model because of the upstream Conv3D regression. Keep the CUDA runtime pin
# separate from the portable lock file and use PJLAB's internal mirrors only.
if [[ "${TTT_SKIP_ENV_SETUP:-0}" != "1" ]]; then
RUNTIME_VERSIONS="$("$PYTHON" - <<'PY' 2>/dev/null || true
try:
    import torch
    import torchaudio
    import torchvision
except Exception:
    pass
else:
    print(f"{torch.__version__}|{torchvision.__version__}|{torchaudio.__version__}")
PY
)"
EXPECTED_RUNTIME_VERSIONS="$H200_TORCH_VERSION|$H200_TORCHVISION_VERSION|$H200_TORCHAUDIO_VERSION"
if [[ "$RUNTIME_VERSIONS" != "$EXPECTED_RUNTIME_VERSIONS" ]]; then
  "$PYTHON" -m pip install --disable-pip-version-check \
    --index-url "$PYTORCH_INDEX_URL" \
    --extra-index-url "$PYPI_INDEX_URL" \
    --trusted-host pypi.i.h.pjlab.org.cn \
    --trusted-host mirrors.i.h.pjlab.org.cn \
    --prefer-binary \
    "torch==$H200_TORCH_VERSION" \
    "torchvision==$H200_TORCHVISION_VERSION" \
    "torchaudio==$H200_TORCHAUDIO_VERSION"
fi

REQUIREMENTS="$PROJECT_ROOT/configs/h200/requirements-h200.lock.txt"
REQUIREMENTS_HASH="$(sha256sum "$REQUIREMENTS" | awk '{print $1}')"
REQUIREMENTS_STAMP="$VENV/.requirements-h200.sha256"
if [[ ! -f "$REQUIREMENTS_STAMP" ]] \
   || [[ "$(<"$REQUIREMENTS_STAMP")" != "$REQUIREMENTS_HASH" ]]; then
  "$PYTHON" -m pip install --disable-pip-version-check \
    --index-url "$PIP_INDEX_URL" \
    --prefer-binary \
    --requirement "$REQUIREMENTS"
  printf '%s\n' "$REQUIREMENTS_HASH" > "$REQUIREMENTS_STAMP"
fi
else
  "$PYTHON" -c 'import torch, torchvision, torchaudio, transformers, accelerate, deepspeed, av'
fi

LF_COMMIT="$(git -C "$PLAY_ROOT/LLaMA-Factory" rev-parse --short HEAD)"
if [[ "$LF_COMMIT" != "523f801" ]]; then
  echo "LLaMA-Factory commit drift: expected 523f801, got $LF_COMMIT" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src:$PLAY_ROOT/LLaMA-Factory/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "$MANIFEST" ]]; then
  PREP_OUTPUT="$("$PYTHON" scripts/prepare_svcbench_episodes.py \
    --annotation "$ANNOTATION" \
    --converted-dataset "$CONVERTED_DATASET" \
    --video-root "$SOURCE_VIDEO_ROOT" \
    --dataset-name svcbench-part \
    --dataset-revision h200-20260710 \
    --output-root "$PROJECT_ROOT/runs")"
  PREP_DIR="$(printf '%s\n' "$PREP_OUTPUT" | tail -n 1)"
  MANIFEST="$PREP_DIR/dataset_manifest.json"
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "dataset manifest not found: $MANIFEST" >&2
  exit 1
fi

export TTT_PROJECT_ROOT="$PROJECT_ROOT"
export TTT_H200_PLAY_ROOT="$PLAY_ROOT"
export MODEL DATASET_DIR DATASET_NAME YAML RUN_ID SESSION RUN_ROOT LOG_DIR LOG_FILE
export SVCBENCH_DATASET_MANIFEST="$(cd "$(dirname "$MANIFEST")" && pwd)/$(basename "$MANIFEST")"
export SVCBENCH_VIDEO_ROOT="$SOURCE_VIDEO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "stage=$STAGE"
echo "manifest=$SVCBENCH_DATASET_MANIFEST"
echo "video_root=$SVCBENCH_VIDEO_ROOT"
echo "python=$PYTHON"
if [[ "$STAGE" == "a5" ]]; then
  echo "a2_checkpoint=$A2_CHECKPOINT"
fi

exec bash scripts/h200/launch_4gpu.sh "$STAGE"
