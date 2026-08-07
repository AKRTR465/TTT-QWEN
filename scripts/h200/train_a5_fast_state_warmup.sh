#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  bash scripts/h200/train_a5_fast_state_warmup.sh \
    <a2_final_checkpoint> <dataset_manifest.json>

Runs the independent four-GPU 256-step A5 Memory/State warmup:
  - Qwen, W0, and RMSNorm/P_in/P_out are fully frozen and excluded from AdamW
  - only P_C + memory-interface parameters and the four state groups are trainable
  - only the existing answer + balanced state Query objective is used
  - success atomically publishes a5_warmup_bundle only; no full checkpoint is saved

An interrupted formal warmup never publishes a bundle.
EOF
  exit 2
}

[[ $# -eq 2 ]] || usage

PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
WARMUP_YAML="$PROJECT_ROOT/configs/h200/a5_fast_state_warmup_256_4gpu.yaml"

if [[ ! -f "$WARMUP_YAML" ]]; then
  echo "A5 Memory/State warmup config not found: $WARMUP_YAML" >&2
  exit 1
fi

export YAML="$WARMUP_YAML"
export TTT_PROJECT_CONFIG="${TTT_PROJECT_CONFIG:-$PROJECT_ROOT/configs/model_state_ttt_8b.yaml}"
export TTT_H200_VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-uv-py312-torch28}"
export TTT_SKIP_ENV_SETUP="${TTT_SKIP_ENV_SETUP:-1}"
export TTT_PREPROCESS_CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260726_a5_dense_querybundle_v4_fp16}"
export RUN_ID="${RUN_ID:-$(date +%y%m%d_%H%M%S)_a5_memory_state_warmup256_v4}"
export SESSION="${SESSION:-a5_fast_state_warmup_${RUN_ID}}"

exec bash "$PROJECT_ROOT/scripts/h200/train_a2_a5.sh" a5 "$@"
