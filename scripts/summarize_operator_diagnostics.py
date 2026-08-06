#!/usr/bin/env python3
"""Summarize additive 8x9 Operator diagnostics from JSONL or TensorBoard logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

OFFICIAL_CLASSES = (
    "o1-snap",
    "o1-delta",
    "o2-unique",
    "o2-gain",
    "e1-action",
    "e1-transit",
    "e2-periodic",
    "e2-episode",
)
UNSUPPORTED = "unsupported"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _json_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        payload: Any = json.loads(line)
        if not isinstance(payload, dict):
            continue
        candidate = payload.get("metrics", payload)
        if not isinstance(candidate, dict):
            continue
        for name, value in candidate.items():
            if isinstance(name, str) and isinstance(value, (int, float)):
                metrics[name] = float(value)
    return metrics


def _tensorboard_metrics(path: Path) -> dict[str, float]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:  # pragma: no cover - production environment dependency
        raise RuntimeError("TensorBoard is required to read event files") from error
    accumulator = EventAccumulator(str(path))
    accumulator.Reload()
    metrics: dict[str, float] = {}
    for tag in accumulator.Tags().get("scalars", ()):
        values = accumulator.Scalars(tag)
        if values:
            metrics[tag] = float(values[-1].value)
    return metrics


def load_metrics(inputs: list[Path]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for path in inputs:
        if path.is_dir() or path.name.startswith("events.out.tfevents"):
            metrics.update(_tensorboard_metrics(path))
        else:
            metrics.update(_json_metrics(path))
    return metrics


def render(metrics: dict[str, float]) -> str:
    rows = [
        "| class | support | recall | mean_ce | predicted_as_unsupported |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in OFFICIAL_CLASSES:
        rows.append(
            "| {name} | {support:.0f} | {recall:.6f} | {mean_ce:.6f} | {unsupported:.0f} |".format(
                name=name,
                support=metrics.get(f"operator/support/{name}", 0.0),
                recall=metrics.get(f"operator/recall/{name}", float("nan")),
                mean_ce=metrics.get(f"operator/raw_loss/{name}", float("nan")),
                unsupported=metrics.get(
                    f"operator/effective_confusion/{name}/{UNSUPPORTED}", 0.0
                ),
            )
        )
    rows.extend(
        (
            "",
            f"micro_accuracy: {metrics.get('operator/micro_accuracy', float('nan')):.6f}",
            f"macro_recall: {metrics.get('operator/macro_recall', float('nan')):.6f}",
            "predicted_unsupported_rate: "
            f"{metrics.get('operator/predicted_unsupported_rate', float('nan')):.6f}",
            f"temperature: {metrics.get('operator/temperature', float('nan')):.6f}",
        )
    )
    return "\n".join(rows) + "\n"


def main() -> int:
    args = _parser().parse_args()
    output = render(load_metrics(args.inputs))
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
