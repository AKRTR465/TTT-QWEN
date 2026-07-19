"""Summarize ``TTT_DATALOADER_TRACE`` JSONL events by rank and phase."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    durations: dict[tuple[int, str], list[float]] = defaultdict(list)
    for line in args.trace.read_text(encoding="utf-8").splitlines():
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
    for (rank, event), values in sorted(durations.items()):
        ordered = sorted(values)
        if len(ordered) > 1:
            quantiles = statistics.quantiles(ordered, n=100, method="inclusive")
            p50, p95 = quantiles[49], quantiles[94]
        else:
            p50 = p95 = ordered[0]
        print(
            json.dumps(
                {
                    "rank": rank,
                    "event": event,
                    "count": len(values),
                    "mean_seconds": statistics.fmean(values),
                    "p50_seconds": p50,
                    "p95_seconds": p95,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
