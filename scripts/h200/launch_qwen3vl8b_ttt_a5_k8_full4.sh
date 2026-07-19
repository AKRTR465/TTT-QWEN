#!/usr/bin/env bash
set -euo pipefail

test "$(id -un)" = niujunbo || { echo "Refusing non-niujunbo shell"; exit 97; }

BASE=${BASE:-/mnt/shared-storage-user/mineru2-shared/niujunbo/play/projects/ttt_qwen}
export TTT_PROJECT_ROOT=${TTT_PROJECT_ROOT:-$BASE}
export YAML=${YAML:-$BASE/configs/h200/a5_meta_ttt_k8_4gpu.yaml}
export MODEL=${MODEL:-/mnt/shared-storage-user/mineru2-shared/niujunbo/play/model/Qwen3-VL-8B-Instruct}
export DATASET_DIR=${DATASET_DIR:-/mnt/shared-storage-user/mineru2-shared/niujunbo/play/datasets/qwensft-data/svcbench-part}
export DATASET_NAME=${DATASET_NAME:-svcbench_qwen3vl_sft}
export RUN_ID=${RUN_ID:-$(date +%y%m%d_%H%M%S)_qwen3vl8b_ttt_a5_k8_full4}
export SESSION=${SESSION:-qwen3vl8b_ttt_a5_k8_full4_${RUN_ID}}

exec bash "$BASE/scripts/h200/train_a2_a5.sh" a5 "$@"
