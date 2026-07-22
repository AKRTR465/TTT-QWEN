#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/benchmark_fullprefix256_8step.sh baseline
  bash scripts/h200/benchmark_fullprefix256_8step.sh a2 [dataset_manifest.json]
  bash scripts/h200/benchmark_fullprefix256_8step.sh a5 <a2_checkpoint> [dataset_manifest.json]
EOF
  exit 2
}

[[ $# -ge 1 ]] || usage
MODE="$1"
[[ "$MODE" == "baseline" || "$MODE" == "a2" || "$MODE" == "a5" ]] || usage
shift

EXPECTED_USER="niujunbo"
PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-py312-torch28}"
MODEL="${MODEL:-$PLAY_ROOT/model/Qwen3-VL-8B-Instruct}"
DATASET_DIR="${DATASET_DIR:-$PLAY_ROOT/datasets/qwensft-data/svcbench-part}"
DATASET_NAME="${BENCHMARK_DATASET_NAME:-svcbench_qwen3vl_sft}"
RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_${MODE}_fullprefix256_8step_4h200}"
SESSION="${SESSION:-${MODE}_fullprefix256_8step_${RUN_ID}}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/runs/$RUN_ID}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs/$RUN_ID}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/experiment.log}"
VISUAL_COST_INDEX="${VISUAL_COST_INDEX:-$PROJECT_ROOT/artifacts/a2_state16_answer256_ema_visual_cost_index.json}"
TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"
if [[ "$TTT_SMOKE_SHORTEST_FIRST" != "0" && "$TTT_SMOKE_SHORTEST_FIRST" != "1" ]]; then
  echo "TTT_SMOKE_SHORTEST_FIRST must be 0 or 1, got: $TTT_SMOKE_SHORTEST_FIRST" >&2
  exit 2
fi
export TTT_SMOKE_SHORTEST_FIRST

if [[ "$(id -un)" != "$EXPECTED_USER" ]]; then
  echo "refusing to benchmark as $(id -un); expected $EXPECTED_USER" >&2
  exit 1
fi

if [[ "${RUN_IN_TMUX:-0}" != "1" ]]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 1; }
  command=(
    env RUN_IN_TMUX=1 TTT_PROJECT_ROOT="$PROJECT_ROOT" TTT_H200_VENV="$VENV"
    MODEL="$MODEL" DATASET_DIR="$DATASET_DIR" DATASET_NAME="$DATASET_NAME"
    RUN_ID="$RUN_ID" SESSION="$SESSION" RUN_ROOT="$RUN_ROOT"
    LOG_DIR="$LOG_DIR" LOG_FILE="$LOG_FILE" VISUAL_COST_INDEX="$VISUAL_COST_INDEX"
    TTT_SMOKE_SHORTEST_FIRST="$TTT_SMOKE_SHORTEST_FIRST"
    bash "$PROJECT_ROOT/scripts/h200/benchmark_fullprefix256_8step.sh" "$MODE"
  )
  command+=("$@")
  printf -v inner '%q ' "${command[@]}"
  tmux new-session -d -s "$SESSION" "cd '$PROJECT_ROOT' && $inner"
  echo "session=$SESSION"
  echo "run_root=$RUN_ROOT"
  echo "tail -f $LOG_FILE"
  exit 0
fi

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if (( GPU_COUNT < 4 )); then
  echo "four GPUs are required; nvidia-smi reported $GPU_COUNT" >&2
  exit 1
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "existing H200 environment not found: $VENV" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
GPU_LOG="$LOG_DIR/gpu_samples.csv"
nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
  --format=csv,noheader,nounits -l 1 > "$GPU_LOG" 2>&1 &
MONITOR_PID="$!"
cleanup_monitor() {
  kill "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup_monitor EXIT INT TERM

START_SECONDS="$(date +%s)"
STATUS=0
cd "$PROJECT_ROOT"
if [[ "$MODE" == "baseline" ]]; then
  [[ $# -eq 0 ]] || usage
  mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/samples"
  export MODEL DATASET_DIR DATASET_NAME
  export OUTPUT_DIR="$RUN_ROOT/checkpoints"
  export PYTHONNOUSERSITE=1
  export PYTHONPATH="$PROJECT_ROOT/src:$PLAY_ROOT/LLaMA-Factory/src${PYTHONPATH:+:$PYTHONPATH}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 NPROC_PER_NODE=4
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  CONFIG="$PROJECT_ROOT/configs/h200/baseline_qwen3vl8b_svcbench_256_4gpu.yaml"
  set +e
  "$VENV/bin/llamafactory-cli" train "$CONFIG" 2>&1 | tee "$RUN_ROOT/train.log"
  STATUS="${PIPESTATUS[0]}"
  set -e
else
  export RUN_IN_TMUX=1 RUN_ID RUN_ROOT LOG_DIR LOG_FILE MODEL DATASET_DIR DATASET_NAME
  export TTT_SKIP_ENV_SETUP=1 TTT_SMOKE_MAX_STEPS=8 TTT_SKIP_FINAL_CHECKPOINT=0
  export TTT_DATALOADER_TRACE=1
  set +e
  bash "$PROJECT_ROOT/scripts/h200/train_fullprefix256.sh" "$MODE" "$@"
  STATUS="$?"
  set -e
fi
ELAPSED="$(( $(date +%s) - START_SECONDS ))"
cleanup_monitor
trap - EXIT INT TERM

mkdir -p "$RUN_ROOT"
cp "$GPU_LOG" "$RUN_ROOT/gpu_samples.csv"
"$VENV/bin/python" - "$RUN_ROOT" "$MODE" "$STATUS" "$ELAPSED" <<'PY'
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
status = int(sys.argv[3])
elapsed = int(sys.argv[4])
rows = defaultdict(lambda: {"util": [], "memory_mib": [], "power_w": []})
with (root / "gpu_samples.csv").open(encoding="utf-8", errors="replace") as handle:
    for row in csv.reader(handle):
        if len(row) != 5:
            continue
        try:
            index = int(row[1].strip())
            util = float(row[2].strip())
            memory = float(row[3].strip())
            power = float(row[4].strip())
        except ValueError:
            continue
        rows[index]["util"].append(util)
        rows[index]["memory_mib"].append(memory)
        rows[index]["power_w"].append(power)

def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1)]

per_gpu = {}
for index, values in sorted(rows.items()):
    per_gpu[str(index)] = {
        "samples": len(values["util"]),
        "utilization_mean_percent": statistics.fmean(values["util"]),
        "utilization_p50_percent": percentile(values["util"], 0.50),
        "utilization_p95_percent": percentile(values["util"], 0.95),
        "memory_peak_mib": max(values["memory_mib"]),
        "power_mean_w": statistics.fmean(values["power_w"]),
    }

summary = {
    "status": "completed" if status == 0 else "failed",
    "exit_code": status,
    "mode": mode,
    "optimizer_steps": 8,
    "elapsed_seconds": elapsed,
    "wall_seconds_per_step": elapsed / 8.0,
    "per_gpu": per_gpu,
    "memory_gate_mib": 136 * 1024,
    "memory_gate_passed": bool(per_gpu) and all(
        item["memory_peak_mib"] <= 136 * 1024 for item in per_gpu.values()
    ),
}
(root / "benchmark_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "benchmark_summary=$RUN_ROOT/benchmark_summary.json"
exit "$STATUS"
