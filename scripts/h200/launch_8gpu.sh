#!/usr/bin/env bash
set -euo pipefail

if [[ "${TTT_WORLD_SIZE:-8}" != "8" ]]; then
  echo "launch_8gpu.sh requires TTT_WORLD_SIZE=8" >&2
  exit 2
fi

export TTT_WORLD_SIZE=8
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/launch_4gpu.sh" "$@"
