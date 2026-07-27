#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/train_a5_vithalf_decoder8.sh <a2_final_checkpoint> <dataset_manifest.json>

Starts four-GPU A5 Meta-TTT with this Qwen Outer policy:
  - freeze ViT patch embedding and blocks 0-12
  - train ViT blocks 13-26, Main Merger, and all DeepStack mergers
  - freeze Decoder layers 0-27 and input/output embeddings
  - train Decoder layers 28-35 and the final language-model norm
  - train for 4 epochs and retain complete checkpoints at epochs 2 and 4 only

The A2 checkpoint must contain the complete Outer model. The v3 manifest must be the same
Support-aligned manifest used to prewarm the strict A5 cache. Environment overrides accepted by
scripts/h200/train_a2_a5.sh remain available, including TTT_PROJECT_ROOT,
TTT_PREPROCESS_CACHE_ROOT, TTT_RESUME_CHECKPOINT, RUN_ID, SESSION, and DRY_RUN.
EOF
  exit 2
}

[[ $# -eq 2 ]] || usage

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
PARTIAL_YAML="$PROJECT_ROOT/configs/h200/a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml"

if [[ ! -f "$PARTIAL_YAML" ]]; then
  echo "partial A5 config not found: $PARTIAL_YAML" >&2
  exit 1
fi

export YAML="$PARTIAL_YAML"
export TTT_PROJECT_CONFIG="${TTT_PROJECT_CONFIG:-$PROJECT_ROOT/configs/model_state_ttt_8b.yaml}"
export TTT_H200_VENV="${TTT_H200_VENV:-$PLAY_ROOT/projects/ttt_qwen/.venv-h200-uv-py312-torch28}"
export TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"
export TTT_PREPROCESS_CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PLAY_ROOT/projects/ttt_qwen/.cache/preprocess/260726_a5_support_aligned_v3_fp16}"
export TTT_PREPROCESS_CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-a5_support_aligned_train_support_statequery_fp16_v3}"
if [[ -n "${TTT_SMOKE_MAX_STEPS:-}" ]]; then
  # Max-step acceptance runs use the atomic smoke path and do not publish epoch checkpoints.
  export TTT_CHECKPOINT_POLICY="atomic_final_only"
  # Acceptance needs per-rank segment/update evidence. CUDA-event tracing is buffered and
  # therefore does not synchronize the hot path per event.
  export TTT_DATALOADER_TRACE="${TTT_DATALOADER_TRACE:-1}"
else
  export TTT_CHECKPOINT_POLICY="${TTT_CHECKPOINT_POLICY:-epoch_2_and_epoch_4}"
fi
export RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_a5_k8_vithalf_decoder8_4h200}"
export SESSION="${SESSION:-a5_k8_vithalf_decoder8_${RUN_ID}}"

exec bash "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" a5 "$@"
