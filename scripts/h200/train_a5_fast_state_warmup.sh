#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/train_a5_fast_state_warmup.sh \
    <a2_final_checkpoint> <dataset_manifest.json>

Runs the independent four-GPU 128-step A5 Fast/State warmup:
  - Qwen is fully frozen and excluded from AdamW
  - all persistent Fast parameters and non-Qwen state modules are trainable
  - only the existing answer + balanced state Query objective is used
  - counterfactual evaluation remains read-only at interval 8
  - success atomically publishes a5_warmup_bundle only; no full checkpoint is saved

Set TTT_SMOKE_MAX_STEPS=32 and TTT_SKIP_FINAL_CHECKPOINT=1 for the required
32-step distributed smoke. An interrupted formal warmup never publishes a bundle.
EOF
  exit 2
}

[[ $# -eq 2 ]] || usage

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
WARMUP_YAML="$PROJECT_ROOT/configs/h200/a5_fast_state_warmup_128_4gpu.yaml"

if [[ ! -f "$WARMUP_YAML" ]]; then
  echo "A5 Fast/State warmup config not found: $WARMUP_YAML" >&2
  exit 1
fi

export YAML="$WARMUP_YAML"
export TTT_A5_ADAPTATION_MODE="meta_ttt"
export TTT_PROJECT_CONFIG="${TTT_PROJECT_CONFIG:-$PROJECT_ROOT/configs/model_state_ttt_8b.yaml}"
export TTT_H200_VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-uv-py312-torch28}"
export TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"
export TTT_PREPROCESS_CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260726_a5_dense_querybundle_v4_fp16}"
export TTT_PREPROCESS_CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-a5_dense_querybundle_train_support_statequery_fp16_v4}"
export TTT_CHECKPOINT_POLICY="atomic_final_only"
if [[ -n "${TTT_SMOKE_MAX_STEPS:-}" ]]; then
  export TTT_DATALOADER_TRACE="${TTT_DATALOADER_TRACE:-1}"
fi
export RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_a5_fast_state_warmup128_statewrite_cf8_v4}"
export SESSION="${SESSION:-a5_fast_state_warmup_${RUN_ID}}"

exec bash "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" a5 "$@"
