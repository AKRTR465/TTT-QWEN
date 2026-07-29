#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/train_a5_associative_lttt_finalonly.sh \
    <a2_final_checkpoint> <a5_warmup_bundle> <dataset_manifest.json>

Runs four-GPU A5 main training with the state-write Associative LTTT design:
  - initialize the complete Outer model from the supplied A2 final checkpoint
  - strictly overlay the verified 128-step Fast/State warmup bundle
  - use Meta-TTT with normalized FP32 active-head state-write targets
  - use the partial Qwen policy (ViT upper half and Decoder last eight layers)
  - train for exactly four epochs with seed/data_seed 42
  - disable periodic Trainer checkpoints and atomically publish final-checkpoint only

The manifest must match the prewarmed dense-Query v4 FP16 Support + State-Query cache.
The final checkpoint includes model weights, trainer metadata, and resume state. Set
TTT_SMOKE_MAX_STEPS together with TTT_SKIP_FINAL_CHECKPOINT=1 only for an explicit
smoke run. DRY_RUN=1 prints the tmux command without starting training.
EOF
  exit 2
}

[[ $# -eq 3 ]] || usage

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
ASSOCIATIVE_YAML="$PROJECT_ROOT/configs/h200/a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml"

if [[ ! -f "$ASSOCIATIVE_YAML" ]]; then
  echo "associative A5 config not found: $ASSOCIATIVE_YAML" >&2
  exit 1
fi

export YAML="$ASSOCIATIVE_YAML"
export TTT_A5_ADAPTATION_MODE="meta_ttt"
export A5_WARMUP_BUNDLE="$2"
export TTT_PROJECT_CONFIG="${TTT_PROJECT_CONFIG:-$PROJECT_ROOT/configs/model_state_ttt_8b.yaml}"
export TTT_H200_VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-uv-py312-torch28}"
export TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"
export TTT_PREPROCESS_CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260726_a5_dense_querybundle_v4_fp16}"
export TTT_PREPROCESS_CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-a5_dense_querybundle_train_support_statequery_fp16_v4}"
export TTT_CHECKPOINT_POLICY="atomic_final_only"
if [[ -n "${TTT_SMOKE_MAX_STEPS:-}" ]]; then
  export TTT_DATALOADER_TRACE="${TTT_DATALOADER_TRACE:-1}"
fi
export RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_a5_statewrite_lttt_warmup128_v4_4epoch_finalonly}"
export SESSION="${SESSION:-a5_associative_lttt_${RUN_ID}}"

exec bash "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" a5 "$1" "$3"
