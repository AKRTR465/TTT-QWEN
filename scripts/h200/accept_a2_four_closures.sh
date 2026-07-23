#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: bash scripts/h200/accept_a2_four_closures.sh prepare|8|32 [source_manifest]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
MODE="$1"
[[ "$MODE" == "prepare" || "$MODE" == "8" || "$MODE" == "32" ]] || usage

EXPECTED_USER="niujunbo"
PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-py312-torch28}"
SOURCE_MANIFEST="${2:-${SVCBENCH_DATASET_MANIFEST:-$PROJECT_ROOT/runs/0719_215434_prepare_svcbench_k8/dataset_manifest.json}}"
ACCEPTANCE_ROOT="${TTT_ACCEPTANCE_MANIFEST_ROOT:-$PROJECT_ROOT/artifacts/a2_four_closures_acceptance}"
ACCEPTANCE_MANIFEST="$ACCEPTANCE_ROOT/dataset_manifest.json"
ACCEPTANCE_VISUAL_COST_INDEX="$ACCEPTANCE_ROOT/visual_cost_index.json"
ACCEPTANCE_YAML="$PROJECT_ROOT/configs/h200/a2_qwen3vl8b_four_closures_acceptance_4gpu.yaml"
PREPROCESS_CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260720_ttt8_benchmark}"
SOURCE_VISUAL_COST_INDEX="${SOURCE_VISUAL_COST_INDEX:-$PROJECT_ROOT/artifacts/a2_trainsplit_state16_answer256_ema_cost_index.json}"

[[ "$(id -un)" == "$EXPECTED_USER" ]] || {
  echo "refusing to run as $(id -un); expected $EXPECTED_USER" >&2
  exit 1
}
[[ -x "$VENV/bin/python" ]] || { echo "H200 environment not found: $VENV" >&2; exit 1; }
[[ -f "$SOURCE_MANIFEST" ]] || { echo "source manifest not found: $SOURCE_MANIFEST" >&2; exit 1; }
[[ -d "$PREPROCESS_CACHE_ROOT" ]] || {
  echo "preprocess cache not found: $PREPROCESS_CACHE_ROOT" >&2
  exit 1
}
[[ -f "$SOURCE_VISUAL_COST_INDEX" ]] || {
  echo "source visual cost index not found: $SOURCE_VISUAL_COST_INDEX" >&2
  exit 1
}
[[ -f "$ACCEPTANCE_YAML" ]] || { echo "acceptance YAML not found: $ACCEPTANCE_YAML" >&2; exit 1; }
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

prepare_manifest() {
  if [[ -e "$ACCEPTANCE_ROOT" ]]; then
    "$VENV/bin/python" - "$ACCEPTANCE_MANIFEST" "$ACCEPTANCE_VISUAL_COST_INDEX" <<'PY'
import sys
from collections import Counter
from pathlib import Path

from ttt_svcbench_qwen.episode_data import load_production_episode_manifest

path = Path(sys.argv[1])
cost_path = Path(sys.argv[2])
manifest = load_production_episode_manifest(path)
counts = Counter((row.split.value, row.query.weak.operator) for row in manifest.a2_queries)
if len(counts) != 16 or min(counts.values()) < 1:
    raise RuntimeError("existing acceptance manifest does not cover all split/operator pairs")
if not cost_path.is_file():
    raise RuntimeError("existing acceptance manifest has no matching visual cost index")
print(path)
PY
    return
  fi
  "$VENV/bin/python" "$PROJECT_ROOT/scripts/prepare_a2_acceptance_manifest.py" \
      --manifest "$SOURCE_MANIFEST" \
      --output "$ACCEPTANCE_ROOT" \
      --visual-cost-index "$SOURCE_VISUAL_COST_INDEX" \
      --train-per-subtype 4 \
      --validation-per-subtype 1
}

prepare_manifest
if [[ "$MODE" == "prepare" ]]; then
  exit 0
fi

RUN_ID="${RUN_ID:-$(date +%m%d_%H%M%S)_a2_four_closures_${MODE}step_4h200}"
SESSION="${SESSION:-a2_four_closures_${MODE}step_${RUN_ID}}"
export RUN_ID SESSION
export TTT_PROJECT_ROOT="$PROJECT_ROOT"
export TTT_H200_VENV="$VENV"
export YAML="$ACCEPTANCE_YAML"
export TTT_SKIP_ENV_SETUP=1
export TTT_SMOKE_MAX_STEPS="$MODE"
export TTT_SKIP_FINAL_CHECKPOINT=1
export TTT_SMOKE_SHORTEST_FIRST=0
export TTT_A2_PROGRESS_TRACE=1
export TTT_DATALOADER_TRACE=1
export TTT_PREPROCESS_CACHE_ROOT="$PREPROCESS_CACHE_ROOT"
if [[ "$MODE" == "8" ]]; then
  export VISUAL_COST_INDEX="$ACCEPTANCE_VISUAL_COST_INDEX"
  export SVCBENCH_DATASET_MANIFEST="$ACCEPTANCE_MANIFEST"
else
  export VISUAL_COST_INDEX="$SOURCE_VISUAL_COST_INDEX"
  export SVCBENCH_DATASET_MANIFEST="$SOURCE_MANIFEST"
fi

exec bash "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" a2 "$SVCBENCH_DATASET_MANIFEST"
