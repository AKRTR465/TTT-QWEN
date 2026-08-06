#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/train_a5_fast_state_warmup_8gpu.sh \
    <a2_final_checkpoint> [dataset_manifest.json]

Runs the independent eight-GPU 256-step A5 Memory/State warmup:
  - Qwen, W0, and RMSNorm/P_in/P_out are fully frozen and excluded from AdamW
  - only P_C + memory-interface parameters and the four state groups are trainable
  - per-device batch remains 1, so each optimizer step consumes eight episodes globally
  - one DataLoader worker/prefetch slot per rank limits startup-process pressure
  - success atomically publishes a5_warmup_bundle only; no full checkpoint is saved
  - omit dataset_manifest.json to generate a new world_size=8 manifest automatically

Set TTT_SMOKE_MAX_STEPS=32 and TTT_SKIP_FINAL_CHECKPOINT=1 for the required
32-step distributed smoke. An interrupted formal warmup never publishes a bundle.
EOF
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
WARMUP_YAML="$PROJECT_ROOT/configs/h200/a5_fast_state_warmup_256_8gpu.yaml"

if [[ ! -f "$WARMUP_YAML" ]]; then
  echo "A5 eight-GPU Memory/State warmup config not found: $WARMUP_YAML" >&2
  exit 1
fi

export YAML="$WARMUP_YAML"
export TTT_WORLD_SIZE=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TTT_LAUNCHER="$PROJECT_ROOT/scripts/h200/launch_8gpu.sh"
export TTT_A5_ADAPTATION_MODE="meta_ttt"
export TTT_PROJECT_CONFIG="${TTT_PROJECT_CONFIG:-$PROJECT_ROOT/configs/model_state_ttt_8b.yaml}"
export TTT_H200_VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-uv-py312-torch28}"
export TTT_SKIP_ENV_SETUP="${TTT_SKIP_ENV_SETUP:-1}"
export TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"
export TTT_PREPROCESS_CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260726_a5_dense_querybundle_v4_fp16}"
export TTT_PREPROCESS_CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-a5_dense_querybundle_train_support_statequery_fp16_v4}"
export TTT_CHECKPOINT_POLICY="atomic_final_only"
if [[ -n "${TTT_SMOKE_MAX_STEPS:-}" ]]; then
  export TTT_DATALOADER_TRACE="${TTT_DATALOADER_TRACE:-1}"
fi
export RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_a5_memory_state_warmup256_8gpu_statewrite_cf8_v4}"
export SESSION="${SESSION:-a5_fast_state_warmup_8gpu_${RUN_ID}}"

exec bash "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" a5 "$@"
