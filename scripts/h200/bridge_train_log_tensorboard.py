#!/usr/bin/env python3
"""Expose structured TTT-QWEN ``train.log`` metrics to TensorBoard."""

from __future__ import annotations

import argparse
import ast
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def parse_row(line: str) -> dict[str, object] | None:
    start = line.find("{'loss':")
    if start < 0:
        return None
    try:
        value = ast.literal_eval(line[start:])
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def scalar_items(row: dict[str, object]) -> Iterator[tuple[str, float]]:
    prefixes = ("loss/", "retrieval/", "grad/", "outer_grad/", "operator/", "task/")
    for key, value in row.items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        if key in {"loss", "grad_norm", "learning_rate", "epoch"} or key.startswith(prefixes):
            yield key, numeric


def write_row(
    writer: Any,
    event_type: Any,
    summary_type: Any,
    row: dict[str, object],
    step: int,
) -> None:
    values = [
        summary_type.Value(tag=tag, simple_value=value) for tag, value in scalar_items(row)
    ]
    if values:
        writer.add_event(
            event_type(wall_time=time.time(), step=step, summary=summary_type(value=values))
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--history-log", type=Path)
    parser.add_argument(
        "--history-steps",
        type=int,
        default=0,
        help="Maximum history rows to publish; zero publishes all available rows.",
    )
    parser.add_argument("--step-offset", type=int, default=0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if args.history_steps < 0 or args.step_offset < 0 or args.poll_seconds <= 0:
        raise ValueError("history-steps/step-offset must be non-negative; poll-seconds positive")

    try:
        from tensorboard.compat.proto import event_pb2, summary_pb2
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorBoard is required; install the project tracking extra before running the bridge"
        ) from error

    args.logdir.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(args.logdir))
    try:
        if args.history_log is not None:
            published = 0
            with args.history_log.open(encoding="utf-8", errors="replace") as source:
                for line in source:
                    row = parse_row(line)
                    if row is None:
                        continue
                    if args.history_steps and published >= args.history_steps:
                        break
                    published += 1
                    write_row(writer, event_pb2.Event, summary_pb2.Summary, row, published)
            writer.flush()
            print(f"published_history_steps={published}", flush=True)

        offset = 0
        observed_steps = 0
        while True:
            try:
                size = args.train_log.stat().st_size
            except FileNotFoundError:
                time.sleep(args.poll_seconds)
                continue
            if size < offset:
                offset = 0
                observed_steps = 0
            with args.train_log.open(encoding="utf-8", errors="replace") as source:
                source.seek(offset)
                for line in source:
                    row = parse_row(line)
                    if row is None:
                        continue
                    observed_steps += 1
                    write_row(
                        writer,
                        event_pb2.Event,
                        summary_pb2.Summary,
                        row,
                        args.step_offset + observed_steps,
                    )
                offset = source.tell()
            writer.flush()
            print(f"published_current_steps={observed_steps}", flush=True)
            time.sleep(args.poll_seconds)
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
