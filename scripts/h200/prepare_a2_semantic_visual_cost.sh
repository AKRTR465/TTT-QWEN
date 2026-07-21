#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "usage: bash scripts/h200/prepare_a2_semantic_visual_cost.sh [manifest] [output]" >&2
  exit 2
fi

EXPECTED_USER="niujunbo"
PLAY_ROOT="/mnt/shared-storage-user/mineru2-shared/niujunbo/play"
PROJECT_ROOT="${TTT_PROJECT_ROOT:-$PLAY_ROOT/projects/ttt_qwen}"
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-py312-torch28}"
PYTHON="$VENV/bin/python"
MODEL="${MODEL:-$PLAY_ROOT/model/Qwen3-VL-8B-Instruct}"
MANIFEST="${1:-${SVCBENCH_DATASET_MANIFEST:-$PROJECT_ROOT/runs/0719_215434_prepare_svcbench_k8/dataset_manifest.json}}"
OUTPUT="${2:-${VISUAL_COST_INDEX:-$PROJECT_ROOT/artifacts/a2_trainsplit_state16_answer256_ema_cost_index.json}}"
VIDEO_ROOT="${SVCBENCH_VIDEO_ROOT:-$PLAY_ROOT/datasets/SVCBench/videos}"

if [[ "$(id -un)" != "$EXPECTED_USER" ]]; then
  echo "refusing to build H200 cost data as $(id -un); expected $EXPECTED_USER" >&2
  exit 1
fi
for path in "$PYTHON" "$MANIFEST"; do
  if [[ ! -e "$path" ]]; then
    echo "required input not found: $path" >&2
    exit 1
  fi
done
if [[ ! -d "$MODEL" || ! -d "$VIDEO_ROOT" ]]; then
  echo "model or SVCBench video root not found" >&2
  exit 1
fi

PROCESSOR_CLASS="$($PYTHON - "$MODEL" <<'PY'
import sys
import transformers

processor = transformers.AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True)
print(f"{type(processor).__module__}.{type(processor).__qualname__}")
PY
)"
GPU_MODEL="${GPU_MODEL:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)}"
mkdir -p "$(dirname "$OUTPUT")"

cd "$PROJECT_ROOT"
exec "$PYTHON" scripts/build_visual_cost_index.py \
  --manifest "$MANIFEST" \
  --stage a2 \
  --output "$OUTPUT" \
  --project-config configs/model_state_ttt_8b.yaml \
  --model-revision "${MODEL_REVISION:-$MODEL@main}" \
  --processor "$PROCESSOR_CLASS" \
  --minimum-pixels 256 \
  --maximum-pixels 131072 \
  --dtype bfloat16 \
  --visual-batch-size 1 \
  --cache-mode readonly \
  --gpu-model "$GPU_MODEL" \
  --state-query-visual-mode recent_chunk \
  --state-query-max-frames 16 \
  --answer-query-visual-mode causal_prefix \
  --answer-query-max-frames 256 \
  --query-sample-fps 2.0 \
  --query-decode-strategy grouped_seek \
  --query-decode-max-groups 16 \
  --video-root "$VIDEO_ROOT"
