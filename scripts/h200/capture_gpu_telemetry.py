#!/usr/bin/env python3
"""Write fixed-interval, per-GPU NVIDIA telemetry for one training run."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path


def parse_nvidia_smi(output: str) -> list[tuple[int, float, int, float]]:
    rows: list[tuple[int, float, int, float]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = tuple(part.strip() for part in line.split(","))
        if len(parts) != 4:
            raise ValueError(f"unexpected nvidia-smi telemetry row: {line!r}")
        index, utilization, memory, power = parts
        rows.append((int(index), float(utilization), int(memory), float(power)))
    if not rows:
        raise ValueError("nvidia-smi returned no GPU telemetry rows")
    return rows


def sample_gpus() -> list[tuple[int, float, int, float]]:
    completed = subprocess.run(
        [
            "/usr/bin/nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval_seconds <= 0:
        raise ValueError("--seconds and --interval-seconds must be positive")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sentinel = output.with_suffix(".done")
    sentinel.unlink(missing_ok=True)
    deadline = time.monotonic() + args.seconds
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch_s", "gpu", "util_pct", "memory_mib", "power_w"))
        while True:
            epoch = time.time()
            for index, utilization, memory, power in sample_gpus():
                writer.writerow((f"{epoch:.3f}", index, utilization, memory, power))
            stream.flush()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(args.interval_seconds, remaining))
    sentinel.touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
