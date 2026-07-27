#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/train_a5_ttt_effect_ablation.sh \
    <A|B|C|D|E> <fixed|learned> <a2_checkpoint> <v4_manifest>

Builds a run-local, immutable project config and starts one four-GPU A5 TTT-effect
ablation. The fixed variants change exactly one strength parameter:
  A baseline
  B Inner SGD LR 1e-4 -> 2e-4
  C Predictor outer LR 5e-5 -> 1e-4
  D Outer TTT coefficient 0.1 -> 0.2
  E W0 gradient cap 0.1 -> 0.15

The learned step controller is a separate, explicit ablation layer. Normal A5
training remains fixed-step and does not instantiate controller parameters.
Set TTT_SMOKE_MAX_STEPS and TTT_SKIP_FINAL_CHECKPOINT=1 for acceptance runs.
EOF
  exit 2
}

[[ $# -eq 4 ]] || usage

FIXED_VARIANT="$1"
STEP_CONTROLLER_MODE="$2"
A2_CHECKPOINT="$3"
MANIFEST="$4"
[[ "$FIXED_VARIANT" =~ ^[A-E]$ ]] || usage
[[ "$STEP_CONTROLLER_MODE" == "fixed" || "$STEP_CONTROLLER_MODE" == "learned" ]] || usage

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-uv-py312-torch28}"
PYTHON="$VENV/bin/python"
BASE_CONFIG="$PROJECT_ROOT/configs/model_state_ttt_8b.yaml"
MODE_TAG="$STEP_CONTROLLER_MODE"
RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)_a5_ttt_effect_${FIXED_VARIANT}_${MODE_TAG}}"
SESSION="${SESSION:-a5_ttt_effect_${FIXED_VARIANT}_${MODE_TAG}_${RUN_ID}}"
CONFIG_DIR="${TTT_ABLATION_CONFIG_DIR:-$PROJECT_ROOT/logs/${RUN_ID}/config}"
PROJECT_CONFIG="$CONFIG_DIR/project_config.yaml"
CONFIG_SUMMARY="$CONFIG_DIR/config_summary.json"

if [[ "$(id -un)" != "niujunbo" ]]; then
  echo "refusing to train as $(id -un); expected niujunbo" >&2
  exit 1
fi
for path in "$PYTHON" "$BASE_CONFIG" "$A2_CHECKPOINT" "$MANIFEST"; do
  if [[ ! -e "$path" ]]; then
    echo "required ablation input does not exist: $path" >&2
    exit 1
  fi
done
if [[ -e "$PROJECT_CONFIG" || -e "$CONFIG_SUMMARY" ]]; then
  echo "refusing to overwrite existing ablation config artifacts: $CONFIG_DIR" >&2
  exit 1
fi

mkdir -p "$CONFIG_DIR"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$PROJECT_ROOT/scripts/build_a5_ttt_effect_config.py" \
  --base "$BASE_CONFIG" \
  --output "$PROJECT_CONFIG" \
  --fixed-variant "$FIXED_VARIANT" \
  --step-controller "$STEP_CONTROLLER_MODE" \
  > "$CONFIG_SUMMARY"

export TTT_PROJECT_ROOT="$PROJECT_ROOT"
export TTT_H200_VENV="$VENV"
export TTT_PROJECT_CONFIG="$PROJECT_CONFIG"
export TTT_GPU_SAMPLE_LOG="${TTT_GPU_SAMPLE_LOG:-$PROJECT_ROOT/logs/${RUN_ID}/gpu_samples.csv}"
export TTT_SKIP_ENV_SETUP="${TTT_SKIP_ENV_SETUP:-1}"
export TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"
export RUN_ID
export SESSION

exec bash "$PROJECT_ROOT/scripts/h200/train_a5_vithalf_decoder8.sh" \
  "$A2_CHECKPOINT" "$MANIFEST"
