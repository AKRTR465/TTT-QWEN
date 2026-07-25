#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${TTT_PROJECT_ROOT:-/mnt/shared-storage-user/mineru2-shared/niujunbo/play/projects/ttt_qwen}"
PLAY_ROOT="${TTT_H200_PLAY_ROOT:-/mnt/shared-storage-user/mineru2-shared/niujunbo/play}"
MANIFEST="${1:?usage: bash scripts/h200/prewarm_a5_train_support_state_cache.sh MANIFEST}"
VIDEO_ROOT="${SVCBENCH_VIDEO_ROOT:-$PLAY_ROOT/datasets/SVCBench/videos}"
CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-$PROJECT_ROOT/.cache/preprocess/260725_a5_half_support_statequery_fp16}"
CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-a5_half_seed42_support_statequery_fp16_v2}"
CACHE_DTYPE="${TTT_PREPROCESS_CACHE_DTYPE:-float16}"
TRAINING_CONFIG="${TTT_TRAINING_CONFIG:-$PROJECT_ROOT/configs/h200/a5_meta_ttt_k8_vithalf_decoder8_4gpu.yaml}"
PROJECT_CONFIG="${TTT_PROJECT_CONFIG:-$PROJECT_ROOT/configs/model_state_ttt_8b.yaml}"
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-py312-torch28}"
PYTHON="${TTT_PYTHON:-$VENV/bin/python}"
SHARD_COUNT="${TTT_CACHE_SHARD_COUNT:-16}"
RUN_ID="${TTT_CACHE_RUN_ID:-$(date +%m%d_%H%M%S)_a5_train_support_state_cache}"
RUN_ROOT="${TTT_CACHE_RUN_ROOT:-$PROJECT_ROOT/runs/$RUN_ID}"
LOCK_DIR="$CACHE_ROOT/.a5_support_state_query_train_prewarm.lock"

[[ "$(id -un)" == "niujunbo" ]] || { echo "refusing non-niujunbo shell" >&2; exit 97; }
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }
[[ -d "$VIDEO_ROOT" ]] || { echo "video root not found: $VIDEO_ROOT" >&2; exit 2; }
[[ -f "$TRAINING_CONFIG" ]] || { echo "training config not found: $TRAINING_CONFIG" >&2; exit 2; }
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid shard count: $SHARD_COUNT" >&2; exit 2; }

mkdir -p "$CACHE_ROOT" "$RUN_ROOT/shards"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another A5 Support + State Query prewarm owns $LOCK_DIR" >&2
  exit 3
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SVCBENCH_VIDEO_ROOT="$VIDEO_ROOT"

{
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'project_root=%s\n' "$PROJECT_ROOT"
  printf 'manifest=%s\n' "$MANIFEST"
  printf 'video_root=%s\n' "$VIDEO_ROOT"
  printf 'cache_root=%s\n' "$CACHE_ROOT"
  printf 'cache_namespace=%s\n' "$CACHE_NAMESPACE"
  printf 'cache_dtype=%s\n' "$CACHE_DTYPE"
  printf 'split=train\nroles=support,state_query\nstage=a5\nshard_count=%s\n' "$SHARD_COUNT"
} > "$RUN_ROOT/command.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_ROOT/git_state.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD >> "$RUN_ROOT/git_state.txt"
"$PYTHON" -VV > "$RUN_ROOT/environment.txt" 2>&1

declare -a pids=()
for ((index = 0; index < SHARD_COUNT; index++)); do
  shard="$(printf '%02d' "$index")"
  (
    set +e
    "$PYTHON" "$PROJECT_ROOT/scripts/preprocess_cache.py" prewarm \
      --root "$CACHE_ROOT" \
      --max-gb 800 \
      --namespace "$CACHE_NAMESPACE" \
      --storage-dtype "$CACHE_DTYPE" \
      --manifest "$MANIFEST" \
      --project-config "$PROJECT_CONFIG" \
      --training-config "$TRAINING_CONFIG" \
      --video-root "$VIDEO_ROOT" \
      --stage a5 \
      --minimum-pixels 256 \
      --maximum-pixels 131072 \
      --split train \
      --roles support state_query \
      --shard-index "$index" \
      --shard-count "$SHARD_COUNT" \
      --summary "$RUN_ROOT/shards/shard_${shard}_summary.json" \
      > "$RUN_ROOT/shards/shard_${shard}.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$RUN_ROOT/shards/shard_${shard}.exit"
    exit "$status"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done

"$PYTHON" - "$RUN_ROOT" "$SHARD_COUNT" "$failed" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
shard_count, failed = map(int, sys.argv[2:])
statuses = {
    f"{index:02d}": int((root / "shards" / f"shard_{index:02d}.exit").read_text())
    for index in range(shard_count)
}
summaries = [
    json.loads((root / "shards" / f"shard_{index:02d}_summary.json").read_text())
    for index in range(shard_count)
    if (root / "shards" / f"shard_{index:02d}_summary.json").is_file()
]
ok = failed == 0 and not any(statuses.values()) and len(summaries) == shard_count
summary = {
    "status": "complete" if ok else "failed",
    "stage": "a5",
    "split": "train",
    "roles": ["support", "state_query"],
    "failed_shards": failed,
    "shard_exit_codes": statuses,
    "candidate_chunk_count": max(
        (item["candidate_chunk_count"] for item in summaries), default=0
    ),
    "unique_chunk_count": max((item["unique_chunk_count"] for item in summaries), default=0),
    "selected_chunk_count": sum(item["selected_chunk_count"] for item in summaries),
    "written_bytes": sum(item["written_bytes"] for item in summaries),
}
(root / "run_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY

"$PYTHON" "$PROJECT_ROOT/scripts/preprocess_cache.py" verify-inputs \
  --root "$CACHE_ROOT" \
  --max-gb 800 \
  --namespace "$CACHE_NAMESPACE" \
  --storage-dtype "$CACHE_DTYPE" \
  --manifest "$MANIFEST" \
  --project-config "$PROJECT_CONFIG" \
  --training-config "$TRAINING_CONFIG" \
  --video-root "$VIDEO_ROOT" \
  --stage a5 \
  --minimum-pixels 256 \
  --maximum-pixels 131072 \
  --split train \
  --roles support state_query \
  > "$RUN_ROOT/cache_verify.json"

"$PYTHON" "$PROJECT_ROOT/scripts/preprocess_cache.py" inspect \
  --root "$CACHE_ROOT" \
  --max-gb 800 \
  --namespace "$CACHE_NAMESPACE" \
  --storage-dtype "$CACHE_DTYPE" \
  > "$RUN_ROOT/cache_inspect.json"
