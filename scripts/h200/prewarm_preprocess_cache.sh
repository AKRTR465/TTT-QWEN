#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLAY_ROOT="${TTT_H200_PLAY_ROOT:-/mnt/shared-storage-user/mineru2-shared/niujunbo/play}"

usage() {
  cat <<'USAGE'
usage: bash scripts/h200/prewarm_preprocess_cache.sh [options] MANIFEST
  --stage {a2,a5}         preprocess stage (default: a2)
  --roles "ROLE ..."      space separated roles (default: state_query)
  --cache-root PATH       preprocess cache root (required)
  --cache-namespace NAME  preprocess cache namespace (required)
  --storage-dtype DTYPE   cache storage dtype (default: unset)
  --training-config PATH  training YAML (required)
  --lock-name NAME        lock directory name under the cache root
  --run-tag TAG           run id suffix under runs/
  --verify {none,inputs}  verify-inputs pass after the shards (default: none)
  --inspect {0,1}         inspect pass after the shards (default: 0)
USAGE
}

STAGE=a2
ROLES=state_query
CACHE_ROOT="${TTT_PREPROCESS_CACHE_ROOT:-}"
CACHE_NAMESPACE="${TTT_PREPROCESS_CACHE_NAMESPACE:-}"
CACHE_DTYPE="${TTT_PREPROCESS_CACHE_DTYPE:-}"
TRAINING_CONFIG="${TTT_TRAINING_CONFIG:-}"
LOCK_NAME=""
RUN_TAG=""
VERIFY=none
INSPECT=0
MANIFEST="${SVCBENCH_DATASET_MANIFEST:-}"

while (($# > 0)); do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --roles) ROLES="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --cache-namespace) CACHE_NAMESPACE="$2"; shift 2 ;;
    --storage-dtype) CACHE_DTYPE="$2"; shift 2 ;;
    --training-config) TRAINING_CONFIG="$2"; shift 2 ;;
    --lock-name) LOCK_NAME="$2"; shift 2 ;;
    --run-tag) RUN_TAG="$2"; shift 2 ;;
    --verify) VERIFY="$2"; shift 2 ;;
    --inspect) INSPECT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) MANIFEST="$1"; shift ;;
  esac
done

VIDEO_ROOT="${SVCBENCH_VIDEO_ROOT:-$PLAY_ROOT/datasets/SVCBench/videos}"
PROJECT_CONFIG="${TTT_PROJECT_CONFIG:-$PROJECT_ROOT/configs/model_state_ttt_8b.yaml}"
VENV="${TTT_H200_VENV:-$PROJECT_ROOT/.venv-h200-py312-torch28}"
PYTHON="${TTT_PYTHON:-$VENV/bin/python}"
CACHE_CLI="$PROJECT_ROOT/scripts/preprocess_cache.py"
SHARD_COUNT="${TTT_CACHE_SHARD_COUNT:-16}"
RUN_ID="${TTT_CACHE_RUN_ID:-$(date +%m%d_%H%M%S)_${RUN_TAG:-${STAGE}_preprocess_cache}}"
RUN_ROOT="${TTT_CACHE_RUN_ROOT:-$PROJECT_ROOT/runs/$RUN_ID}"
LOCK_DIR="$CACHE_ROOT/${LOCK_NAME:-.${STAGE}_prewarm.lock}"

[[ "$STAGE" == "a2" || "$STAGE" == "a5" ]] || { echo "invalid stage: $STAGE" >&2; exit 2; }
[[ "$VERIFY" == "none" || "$VERIFY" == "inputs" ]] || { echo "invalid verify: $VERIFY" >&2; exit 2; }
[[ "$INSPECT" == "0" || "$INSPECT" == "1" ]] || { echo "invalid inspect: $INSPECT" >&2; exit 2; }
[[ -n "$CACHE_ROOT" ]] || { echo "missing --cache-root" >&2; exit 2; }
[[ -n "$CACHE_NAMESPACE" ]] || { echo "missing --cache-namespace" >&2; exit 2; }
[[ -n "$ROLES" ]] || { echo "missing --roles" >&2; exit 2; }
[[ "$(id -un)" == "niujunbo" ]] || { echo "refusing non-niujunbo shell" >&2; exit 97; }
[[ -x "$PYTHON" ]] || { echo "Python not found: $PYTHON" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }
[[ -d "$VIDEO_ROOT" ]] || { echo "video root not found: $VIDEO_ROOT" >&2; exit 2; }
[[ -f "$TRAINING_CONFIG" ]] || { echo "training config not found: $TRAINING_CONFIG" >&2; exit 2; }
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid shard count: $SHARD_COUNT" >&2; exit 2; }

read -r -a ROLE_ARGS <<<"$ROLES"
declare -a CACHE_ARGS=(--root "$CACHE_ROOT" --max-gb 800 --namespace "$CACHE_NAMESPACE")
if [[ -n "$CACHE_DTYPE" ]]; then
  CACHE_ARGS+=(--storage-dtype "$CACHE_DTYPE")
fi
declare -a INPUT_ARGS=(
  --manifest "$MANIFEST"
  --project-config "$PROJECT_CONFIG"
  --training-config "$TRAINING_CONFIG"
  --video-root "$VIDEO_ROOT"
  --stage "$STAGE"
  --minimum-pixels 256
  --maximum-pixels 131072
  --split train
  --roles "${ROLE_ARGS[@]}"
)

mkdir -p "$CACHE_ROOT" "$RUN_ROOT/shards"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another preprocess cache prewarm owns $LOCK_DIR" >&2
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
  if [[ -n "$CACHE_DTYPE" ]]; then printf 'cache_dtype=%s\n' "$CACHE_DTYPE"; fi
  printf 'split=train\nroles=%s\nstage=%s\nshard_count=%s\n' "${ROLES// /,}" "$STAGE" "$SHARD_COUNT"
} > "$RUN_ROOT/command.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_ROOT/git_state.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD >> "$RUN_ROOT/git_state.txt"
"$PYTHON" -VV > "$RUN_ROOT/environment.txt" 2>&1

declare -a pids=()
for ((index = 0; index < SHARD_COUNT; index++)); do
  shard="$(printf '%02d' "$index")"
  (
    set +e
    "$PYTHON" "$CACHE_CLI" prewarm "${CACHE_ARGS[@]}" "${INPUT_ARGS[@]}" \
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

verify_status=0
if [[ "$VERIFY" == "inputs" ]]; then
  set +e
  "$PYTHON" "$CACHE_CLI" verify-inputs "${CACHE_ARGS[@]}" "${INPUT_ARGS[@]}" \
    > "$RUN_ROOT/cache_verify.json"
  verify_status=$?
  set -e
fi

if [[ "$INSPECT" == "1" ]]; then
  "$PYTHON" "$CACHE_CLI" inspect "${CACHE_ARGS[@]}" > "$RUN_ROOT/cache_inspect.json"
fi

"$PYTHON" - "$RUN_ROOT" "$SHARD_COUNT" "$failed" "$verify_status" "$STAGE" "$ROLES" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
shard_count, failed, verify_status = (int(value) for value in sys.argv[2:5])
stage, roles = sys.argv[5], sys.argv[6].split()
statuses = {
    f"{index:02d}": int((root / "shards" / f"shard_{index:02d}.exit").read_text())
    for index in range(shard_count)
}
summaries = [
    json.loads((root / "shards" / f"shard_{index:02d}_summary.json").read_text())
    for index in range(shard_count)
    if (root / "shards" / f"shard_{index:02d}_summary.json").is_file()
]
ok = (
    failed == 0
    and not any(statuses.values())
    and len(summaries) == shard_count
    and verify_status == 0
)
summary = {
    "status": "complete" if ok else "failed",
    "stage": stage,
    "split": "train",
    "roles": roles,
    "shard_count": shard_count,
    "failed_shards": failed,
    "shard_exit_codes": statuses,
    "candidate_chunk_count": max(
        (item["candidate_chunk_count"] for item in summaries), default=0
    ),
    "unique_chunk_count": max((item["unique_chunk_count"] for item in summaries), default=0),
    "selected_chunk_count": sum(item["selected_chunk_count"] for item in summaries),
    "written_bytes": sum(item["written_bytes"] for item in summaries),
}
for key, path in (("verification", "cache_verify.json"), ("cache", "cache_inspect.json")):
    if (root / path).is_file():
        summary[key] = json.loads((root / path).read_text())
(root / "run_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, sort_keys=True))
raise SystemExit(0 if ok else 1)
PY
