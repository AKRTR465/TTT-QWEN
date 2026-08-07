#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ "$1" != "a2" && "$1" != "a5" ]]; then
  echo "usage: $0 a2|a5" >&2
  exit 2
fi

STAGE="$1"
EXPECTED_USER="niujunbo"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-/mnt/shared-storage-user/mineru2-shared/niujunbo/play/projects/ttt_qwen}"
PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
export TTT_H200_PLAY_ROOT="${TTT_H200_PLAY_ROOT:-$PLAY_ROOT}"
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-py312-torch28}"
PYTHON="$VENV/bin/python"
WORLD_SIZE="${TTT_WORLD_SIZE:-4}"
case "$WORLD_SIZE" in
  4)
    DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3"
    ;;
  8)
    DEFAULT_CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    ;;
  *)
    echo "TTT_WORLD_SIZE must be 4 or 8, got: $WORLD_SIZE" >&2
    exit 2
    ;;
esac

if [[ "$(id -un)" != "$EXPECTED_USER" ]]; then
  echo "refusing to train as $(id -un); expected $EXPECTED_USER" >&2
  exit 1
fi
if [[ ! -e "$PROJECT_ROOT/.git" ]]; then
  echo "project checkout not found: $PROJECT_ROOT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "project Python is missing; run scripts/h200/train_a2_a5.sh first" >&2
  exit 1
fi
: "${SVCBENCH_VIDEO_ROOT:?set SVCBENCH_VIDEO_ROOT to the converted SVCBench dataset root}"
: "${SVCBENCH_DATASET_MANIFEST:?set SVCBENCH_DATASET_MANIFEST to dataset_manifest.json}"
if [[ ! -f "$SVCBENCH_DATASET_MANIFEST" ]]; then
  echo "dataset manifest not found: $SVCBENCH_DATASET_MANIFEST" >&2
  exit 1
fi
if [[ ! -d "$SVCBENCH_VIDEO_ROOT" ]]; then
  echo "SVCBench video root not found: $SVCBENCH_VIDEO_ROOT" >&2
  exit 1
fi
if [[ -n "${TTT_RESUME_CHECKPOINT:-}" ]]; then
  if [[ ! -d "$TTT_RESUME_CHECKPOINT" \
     || ! -f "$TTT_RESUME_CHECKPOINT/trainer_state.json" \
     || ! -f "$TTT_RESUME_CHECKPOINT/scheduler.pt" ]]; then
    echo "TTT_RESUME_CHECKPOINT must be a standard Trainer checkpoint: $TTT_RESUME_CHECKPOINT" >&2
    exit 1
  fi
  if [[ ! -f "$TTT_RESUME_CHECKPOINT/optimizer.pt" ]] \
     && ! find "$TTT_RESUME_CHECKPOINT" -maxdepth 1 -type d -name 'global_step*' -print -quit | grep -q .; then
    echo "TTT_RESUME_CHECKPOINT has no Trainer/DeepSpeed optimizer state: $TTT_RESUME_CHECKPOINT" >&2
    exit 1
  fi
fi
if [[ "$STAGE" == "a5" ]]; then
  if [[ -n "${TTT_RESUME_CHECKPOINT:-}" ]]; then
    export A2_CHECKPOINT="${A2_CHECKPOINT:-$TTT_RESUME_CHECKPOINT}"
  else
    : "${A2_CHECKPOINT:?set A2_CHECKPOINT to the final A2 checkpoint}"
    if [[ ! -d "$A2_CHECKPOINT" ]]; then
      echo "A2 checkpoint directory not found: $A2_CHECKPOINT" >&2
      exit 1
    fi
    if [[ ! -f "$A2_CHECKPOINT/model.safetensors" \
       && ! -f "$A2_CHECKPOINT/model.safetensors.index.json" \
       && ! -f "$A2_CHECKPOINT/pytorch_model.bin" \
       && ! -f "$A2_CHECKPOINT/pytorch_model.bin.index.json" ]]; then
      echo "A2 checkpoint has no loadable outer-model weights: $A2_CHECKPOINT" >&2
      exit 1
    fi
  fi
fi

GPU_COUNT="$(/usr/bin/nvidia-smi -L | wc -l | tr -d ' ')"
if (( GPU_COUNT < WORLD_SIZE )); then
  echo "$WORLD_SIZE GPUs are required; nvidia-smi reported $GPU_COUNT" >&2
  exit 1
fi
FREE_KB="$(df -Pk /mnt/shared-storage-user/mineru2-shared | awk 'NR==2 {print $4}')"
if (( FREE_KB < 209715200 )); then
  echo "shared storage has less than 200 GiB free; aborting before checkpoint smoke" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PLAY_ROOT/LLaMA-Factory/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
VISIBLE_GPU_COUNT="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
if (( VISIBLE_GPU_COUNT != WORLD_SIZE )); then
  echo "TTT_WORLD_SIZE=$WORLD_SIZE requires exactly $WORLD_SIZE CUDA_VISIBLE_DEVICES entries, got: $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/mnt/shared-storage-user/mineru2-shared/niujunbo/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PLAY_ROOT/.cache/pip}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_${USER:-niujunbo}_${RUN_ID:-ttt}}"
mkdir -p "$TRITON_CACHE_DIR"
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if command -v ip >/dev/null 2>&1 && ip link show bond0 >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME=bond0
  elif command -v ip >/dev/null 2>&1 && ip link show eth0 >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME=eth0
  else
    export NCCL_SOCKET_IFNAME=lo
  fi
fi
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTHONUNBUFFERED=1
export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE="$WORLD_SIZE"

"$PYTHON" - <<'PY'
import importlib.metadata as metadata
import sys

required = {
    "accelerate": "1.11.0",
    "av": "16.0.0",
    "deepspeed": "0.18.8",
    "peft": "0.18.1",
    "transformers": "4.57.1",
}
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"production runtime requires Python 3.12, got {sys.version}")
actual = {name: metadata.version(name) for name in required}
drift = {name: (required[name], value) for name, value in actual.items() if value != required[name]}
if drift:
    raise RuntimeError(f"H200 environment version drift: {drift}")

import torch
import torchaudio
import torchvision

if torch.__version__ != "2.8.0+cu128":
    raise RuntimeError(f"H200 runtime requires torch 2.8.0+cu128, got {torch.__version__}")
if torchvision.__version__ != "0.23.0+cu128":
    raise RuntimeError(
        f"H200 runtime requires torchvision 0.23.0+cu128, got {torchvision.__version__}"
    )
if torchaudio.__version__ != "2.8.0+cu128":
    raise RuntimeError(
        f"H200 runtime requires torchaudio 2.8.0+cu128, got {torchaudio.__version__}"
    )
if torch.version.cuda != "12.8":
    raise RuntimeError(f"H200 runtime requires a CUDA 12.8 torch wheel, got {torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("torch cannot see CUDA on the H200 worker")
PY

if [[ "$STAGE" == "a2" ]]; then
  TASK_NAME="a2_fullprefix256_8b_${WORLD_SIZE}h200"
  CONFIG="${YAML:-configs/h200/a2_qwen3vl8b_fullprefix256_4gpu.yaml}"
else
  TASK_NAME="a5_k8_fullprefix256_8b_${WORLD_SIZE}h200"
  CONFIG="${YAML:-configs/h200/a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml}"
fi
if [[ -n "${TTT_RESUME_CHECKPOINT:-}" ]]; then
  TASK_NAME="${TASK_NAME}_resume"
fi
RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_${TASK_NAME}}"
export RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/$RUN_ID}"
export OUTPUT_DIR="$RUN_ROOT/checkpoints"
if [[ -e "$RUN_ROOT" ]]; then
  echo "refusing to overwrite an existing run: $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR" "$RUN_ROOT/samples"
cp "$SVCBENCH_DATASET_MANIFEST" "$RUN_ROOT/dataset_manifest.json"
: > "$RUN_ROOT/succeeded.jsonl"
: > "$RUN_ROOT/failed.jsonl"
export LAUNCHER_PID="$$"

"$PYTHON" - "$RUN_ROOT/run_config.json" "$STAGE" "$CONFIG" <<'PY'
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

path, stage, config = sys.argv[1:]
adaptation_mode = None
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.episode_data import EpisodeSplit, load_production_episode_manifest
from ttt_svcbench_qwen.fast_ttt import ASSOCIATIVE_CONTRACT
from ttt_svcbench_qwen.production_factory import load_training_yaml

native, extension = load_training_yaml(config)
project = load_config(extension.project_config)
world_size = int(os.environ["NPROC_PER_NODE"])
if stage == "a5":
    adaptation_mode = extension.a5_adaptation_mode
    manifest = load_production_episode_manifest(os.environ["SVCBENCH_DATASET_MANIFEST"])
    active_bucket_world_sizes = {
        bucket.world_size for bucket in manifest.buckets if bucket.split is EpisodeSplit.TRAIN
    }
    if active_bucket_world_sizes != {world_size}:
        raise ValueError(
            "A5 dataset manifest bucket world_size does not match the active launcher: "
            f"manifest={sorted(active_bucket_world_sizes)}, active={world_size}; "
            f"regenerate it with scripts/prepare_svcbench_episodes.py --world-size {world_size}"
        )
payload = {
    "stage": stage,
    "project_spec_version": project.spec_version,
    "config_schema_version": project.config_schema_version,
    "associative_ttt_contract": ASSOCIATIVE_CONTRACT,
    "a5_adaptation_mode": adaptation_mode,
    "a5_phase": extension.a5_phase if stage == "a5" else None,
    "warmup_bundle": extension.warmup_bundle if stage == "a5" else None,
    "memory_write_rule": project.fast_memory.write_rule,
    "memory_dtype": project.fast_memory.memory_dtype,
    "query_meta_gradient": {
        "mode": project.a5.query_meta_gradient.mode,
        "max_norm": float(project.a5.query_meta_gradient.max_norm),
        "epsilon": float(project.a5.query_meta_gradient.epsilon),
    },
    "config": config,
    "working_directory": os.getcwd(),
    "host": socket.gethostname(),
    "pid": int(os.environ["LAUNCHER_PID"]),
    "started_at": datetime.now(timezone.utc).isoformat(),
    "model": os.environ["MODEL"],
    "world_size": world_size,
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "dataset_manifest": os.environ["SVCBENCH_DATASET_MANIFEST"],
    "runtime_factory": "ttt_svcbench_qwen.production_runtime:build_runtime",
    "initialize_from_a2": os.environ.get("A2_CHECKPOINT"),
    "same_stage_resume_from": os.environ.get("TTT_RESUME_CHECKPOINT"),
    "progress_command": f"tail -f {os.environ['RUN_ROOT']}/train.log",
    "launcher_log": os.environ.get("LOG_FILE"),
    "video_root": os.environ["SVCBENCH_VIDEO_ROOT"],
    "time_limit_seconds": (
        None
        if not os.environ.get("TTT_RUN_TIMEOUT_SECONDS")
        else int(os.environ["TTT_RUN_TIMEOUT_SECONDS"])
    ),
    "launch_command": f"python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node={world_size} -m ttt_svcbench_qwen.llamafactory_trainer {config}",
}
Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY

"$PYTHON" - "$RUN_ROOT/dataset_manifest.json" "$RUN_ROOT/failed.jsonl" <<'PY'
import json
import sys
from pathlib import Path

from ttt_svcbench_qwen.episode_data import load_production_episode_manifest

manifest_path, failed_path = map(Path, sys.argv[1:])
manifest = load_production_episode_manifest(manifest_path)
Path(failed_path).write_text(
    "".join(
        json.dumps(
            {
                "query_id": failure.query_id,
                "video_id": failure.video_id,
                "reason": failure.reason,
                "query_time": failure.query_time,
                "video_duration": failure.video_duration,
            },
            ensure_ascii=False,
        )
        + "\n"
        for failure in manifest.failures
    ),
    encoding="utf-8",
)
PY

"$PYTHON" - "$RUN_ROOT" <<'PY'
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

root = Path(sys.argv[1])
smoke = root / ".checkpoint-save-smoke.safetensors"
expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
save_file({"checkpoint_smoke": expected}, str(smoke))
actual = load_file(str(smoke))["checkpoint_smoke"]
if not torch.equal(actual, expected):
    raise RuntimeError("shared-storage checkpoint smoke roundtrip changed tensor values")
smoke.unlink()
PY

START_EPOCH="$(date +%s)"
{
  echo "start stage=$STAGE run_id=$RUN_ID"
  echo "checkpoint_smoke=passed"
  echo "python=$PYTHON"
  echo "torch=$($PYTHON -c 'import torch; print(torch.__version__)')"
  echo "torchvision=$($PYTHON -c 'import torchvision; print(torchvision.__version__)')"
  echo "torchaudio=$($PYTHON -c 'import torchaudio; print(torchaudio.__version__)')"
  echo "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
  echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
  echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
  if [[ -n "${TTT_RESUME_CHECKPOINT:-}" ]]; then
    echo "same_stage_resume_from=$TTT_RESUME_CHECKPOINT"
  fi
  if [[ "$STAGE" == "a5" ]]; then
    echo "a5_warmup_bundle=${A5_WARMUP_BUNDLE:-none}"
  fi
  echo "launch world_size=$WORLD_SIZE config=$CONFIG"
} > "$RUN_ROOT/experiment.log"

set +e
TRAIN_COMMAND=(
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$WORLD_SIZE"
  -m ttt_svcbench_qwen.llamafactory_trainer "$CONFIG"
)
if [[ -n "${TTT_RUN_TIMEOUT_SECONDS:-}" ]]; then
  if [[ ! "$TTT_RUN_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TTT_RUN_TIMEOUT_SECONDS must be a positive integer" >&2
    exit 2
  fi
  timeout --signal=TERM --kill-after=30s "$TTT_RUN_TIMEOUT_SECONDS" \
    "${TRAIN_COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/train.log"
else
  "${TRAIN_COMMAND[@]}" 2>&1 | tee "$RUN_ROOT/train.log"
fi
STATUS="${PIPESTATUS[0]}"
set -e

ELAPSED="$(( $(date +%s) - START_EPOCH ))"
if [[ "$STATUS" -eq 0 ]]; then
  echo "complete stage=$STAGE elapsed_seconds=$ELAPSED" >> "$RUN_ROOT/experiment.log"
else
  echo "failed stage=$STAGE exit_code=$STATUS elapsed_seconds=$ELAPSED" >> "$RUN_ROOT/experiment.log"
fi

"$PYTHON" - "$RUN_ROOT" "$STAGE" "$STATUS" "$ELAPSED" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
stage = sys.argv[2]
status = int(sys.argv[3])
elapsed = int(sys.argv[4])
manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
summary_path = root / "run_summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
else:
    summary = {}
summary.update(
    {
        "status": "completed" if status == 0 else "failed",
        "stage": stage,
        "exit_code": status,
        "elapsed_seconds": elapsed,
        "failed_query_count": len(manifest.get("failures", [])),
        "successful_run_count": 1 if status == 0 else 0,
    }
)
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

if status == 0:
    if stage == "a2":
        manifest_units = sum(row["split"] == "train" for row in manifest["a2_queries"])
        unit_name = "train_queries"
    else:
        manifest_units = sum(
            row["split"] == "train" and row["loss_weight"] == 1.0
            for row in manifest["episodes"]
        )
        unit_name = "train_episodes"
    (root / "succeeded.jsonl").write_text(
        json.dumps(
            {
                "scope": "run",
                "stage": stage,
                "status": "completed",
                f"manifest_{unit_name}": manifest_units,
                "note": "per-record materialization belongs in samples/ runtime logs",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
else:
    with (root / "failed.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "scope": "run",
                    "stage": stage,
                    "status": "failed",
                    "exit_code": status,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
PY
exit "$STATUS"
