"""Inspect or prune the shared TTT video-preprocessing cache.

Examples (PowerShell):

    python scripts/preprocess_cache.py --root $env:TTT_PREPROCESS_CACHE_ROOT --max-gb 200 --prune

The training workers perform the same eviction best-effort after each write.  This command is
useful before a long run or when a cache was populated by an older experiment namespace.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ttt_svcbench_qwen.preprocess_cache import PreprocessCache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--max-gb", type=float, default=200.0)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--prune", action="store_true")
    args = parser.parse_args()
    if args.max_gb <= 0.0:
        parser.error("--max-gb must be positive")
    cache = PreprocessCache(
        args.root,
        max_bytes=int(args.max_gb * 1024**3),
        memory_entries=0,
        enabled=True,
        namespace=args.namespace,
    )
    removed = cache.prune() if args.prune else 0
    print(
        {
            "root": str(args.root),
            "namespace": args.namespace,
            "size_bytes": cache.disk_size_bytes(),
            "max_bytes": cache.max_bytes,
            "removed_entries": removed,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
