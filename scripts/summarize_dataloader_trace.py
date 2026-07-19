"""Summarize buffered runtime JSONL events by rank, phase, and cross-rank skew."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="one JSONL file or a runtime-trace directory")
    args = parser.parse_args()
    durations: dict[tuple[int, str], list[float]] = defaultdict(list)
    paths = (
        (args.trace,)
        if args.trace.is_file()
        else tuple(sorted(args.trace.rglob("runtime_*.jsonl")))
    )
    if not paths:
        parser.error("no runtime JSONL files found")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            event = row.get("event")
            seconds = row.get("seconds")
            rank = row.get("rank", 0)
            if isinstance(event, str) and isinstance(seconds, (int, float)):
                durations[(int(rank), event)].append(float(seconds))
    rank_p90: dict[str, list[float]] = defaultdict(list)
    for (rank, event), values in sorted(durations.items()):
        ordered = sorted(values)
        if len(ordered) > 1:
            quantiles = statistics.quantiles(ordered, n=100, method="inclusive")
            p50, p90, p99 = quantiles[49], quantiles[89], quantiles[98]
        else:
            p50 = p90 = p99 = ordered[0]
        rank_p90[event].append(p90)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "event": event,
                    "count": len(values),
                    "mean_seconds": statistics.fmean(values),
                    "p50_seconds": p50,
                    "p90_seconds": p90,
                    "p99_seconds": p99,
                },
                sort_keys=True,
            )
        )
    for event, values in sorted(rank_p90.items()):
        minimum = min(values)
        print(
            json.dumps(
                {
                    "event": event,
                    "rank_count": len(values),
                    "rank_p90_max_min_ratio": None if minimum <= 0.0 else max(values) / minimum,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
