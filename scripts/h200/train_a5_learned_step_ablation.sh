#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/train_a5_learned_step_ablation.sh <a2_checkpoint> <v4_manifest>

Runs the four-GPU A5 learned Inner-step ablation with the same fixed training
parameters as Variant A: dense-Query v4 data order, partial Qwen trainability,
seed/data_seed 42, FP16 Support+State-Query cache, Query weight 1.0, and four
epochs. The only model-side change is fast_ttt.step_controller.mode=learned:
the bounded 7->32->1 controller starts at the fixed 1e-4 Inner-SGD step and is
optimized by its independent Outer parameter group.

Only final-checkpoint is published after all four epochs. The canonical fixed
step A5 launcher is unchanged. Set TTT_SMOKE_MAX_STEPS together with
TTT_SKIP_FINAL_CHECKPOINT=1 for a smoke run; DRY_RUN=1 only prints the tmux
command.
EOF
  exit 2
}

[[ $# -eq 2 ]] || usage

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
ABLATION_LAUNCHER="$PROJECT_ROOT/scripts/h200/train_a5_ttt_effect_ablation.sh"

if [[ ! -f "$ABLATION_LAUNCHER" ]]; then
  echo "A5 TTT-effect ablation launcher not found: $ABLATION_LAUNCHER" >&2
  exit 1
fi

export TTT_PROJECT_ROOT="$PROJECT_ROOT"
export TTT_H200_VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-uv-py312-torch28}"
export TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"
export TTT_PREPROCESS_CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260726_a5_dense_querybundle_v4_fp16}"
export TTT_PREPROCESS_CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-a5_dense_querybundle_train_support_statequery_fp16_v4}"
export TTT_CHECKPOINT_POLICY="atomic_final_only"
if [[ -n "${TTT_SMOKE_MAX_STEPS:-}" ]]; then
  export TTT_DATALOADER_TRACE="${TTT_DATALOADER_TRACE:-1}"
fi
export RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_a5_learned_step_dense_querybundle_v4_4epoch_finalonly}"
export SESSION="${SESSION:-a5_learned_step_${RUN_ID}}"

exec bash "$ABLATION_LAUNCHER" A learned "$@"
