#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/train_a5_no_write_ablation.sh <a2_checkpoint> <v4_manifest>

Runs the four-GPU A5 no-write ablation with the same dense-Query v4 data order,
partial Qwen trainability, seed 42, FP16 Support+State-Query cache, and four epochs
as the Meta-TTT experiment. The slot memory stays zero: memory-interface parameters,
Support writes, and Support-to-Query meta gradients are disabled. Only final-checkpoint
is published after all four epochs.

The script does not rebuild the cache. Set TTT_SMOKE_MAX_STEPS plus
TTT_SKIP_FINAL_CHECKPOINT=1 for an explicit smoke run. DRY_RUN=1 prints the tmux
command without starting training.
EOF
  exit 2
}

[[ $# -eq 2 ]] || usage

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
STATIC_YAML="$PROJECT_ROOT/configs/h200/a5_no_write_k8_vithalf_decoder8_4gpu.yaml"

if [[ ! -f "$STATIC_YAML" ]]; then
  echo "no-write A5 config not found: $STATIC_YAML" >&2
  exit 1
fi

export YAML="$STATIC_YAML"
export TTT_A5_ADAPTATION_MODE="no_write"
export TTT_PROJECT_CONFIG="${TTT_PROJECT_CONFIG:-$PROJECT_ROOT/configs/model_state_ttt_8b.yaml}"
export TTT_H200_VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-uv-py312-torch28}"
export TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"
export TTT_PREPROCESS_CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260726_a5_dense_querybundle_v4_fp16}"
export TTT_PREPROCESS_CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-a5_dense_querybundle_train_support_statequery_fp16_v4}"
export TTT_CHECKPOINT_POLICY="atomic_final_only"
if [[ -n "${TTT_SMOKE_MAX_STEPS:-}" ]]; then
  export TTT_DATALOADER_TRACE="${TTT_DATALOADER_TRACE:-1}"
fi
export RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_a5_no_write_dense_querybundle_v4_4epoch_finalonly}"
export SESSION="${SESSION:-a5_no_write_${RUN_ID}}"

exec bash "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" a5 "$@"
