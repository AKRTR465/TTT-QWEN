#!/usr/bin/env python3
"""Compare immutable tuple appends with the episode-local tensor ring."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.query_encoder import OPERATORS, Operator
from ttt_svcbench_qwen.state_bank import (
    HeadType,
    RetrievalHistoryAppendBatch,
    TensorizedRetrievalHistory,
    build_state_bank,
)


def _batch(slots: int, *, dtype: torch.dtype, device: torch.device) -> RetrievalHistoryAppendBatch:
    count = slots + 3
    sources = torch.randn((count, 768), dtype=dtype, device=device)
    heads = torch.tensor((0, *(1 for _ in range(slots)), 2, 3), dtype=torch.int64, device=device)
    operators = torch.tensor(
        (
            OPERATORS.index(Operator.O1_SNAP),
            *(OPERATORS.index(Operator.O2_UNIQUE) for _ in range(slots)),
            OPERATORS.index(Operator.E1_ACTION),
            OPERATORS.index(Operator.E2_EPISODE),
        ),
        dtype=torch.int64,
        device=device,
    )
    timestamps = torch.cat(
        (
            torch.tensor((-1.0,), dtype=torch.float64, device=device),
            torch.arange(slots, dtype=torch.float64, device=device),
            torch.tensor((-1.0, -1.0), dtype=torch.float64, device=device),
        )
    )
    ranges = torch.full((count, 2), -1.0, dtype=torch.float64, device=device)
    ranges[0] = torch.tensor((0.0, 1.0), dtype=torch.float64, device=device)
    ranges[-2:] = torch.tensor((0.0, 1.0), dtype=torch.float64, device=device)
    valid = torch.ones(count, dtype=torch.bool, device=device)
    return RetrievalHistoryAppendBatch(
        sources=sources,
        head_codes=heads,
        operator_codes=operators,
        timestamps=timestamps,
        time_ranges=ranges,
        valid_mask=valid,
        eligible_mask=valid.clone(),
    )


def _legacy_seconds(supports: int, slots: int, repeats: int) -> float:
    bank = build_state_bank(load_config()).cpu()
    rows: list[float] = []
    for _ in range(repeats):
        state = bank.reset("video", "trajectory")
        started = time.perf_counter()
        for support in range(supports):
            values = (
                (HeadType.O1, Operator.O1_SNAP, 1),
                (HeadType.O2, Operator.O2_UNIQUE, slots),
                (HeadType.E1, Operator.E1_ACTION, 1),
                (HeadType.E2, Operator.E2_EPISODE, 1),
            )
            for head, operator, count in values:
                for slot in range(count):
                    timestamp = float(support * 64 + slot)
                    state = bank.append_retrieval_history(
                        state,
                        head_type=head,
                        operator=operator,
                        semantic_source=torch.randn(768),
                        timestamp=timestamp,
                        time_range=None,
                    )
        rows.append(time.perf_counter() - started)
    return statistics.median(rows)


def _ring_seconds(
    supports: int,
    slots: int,
    repeats: int,
    *,
    device: torch.device,
) -> float:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    batch = _batch(slots, dtype=dtype, device=device)
    rows: list[float] = []
    for _ in range(repeats):
        ring = TensorizedRetrievalHistory(
            "video",
            "trajectory",
            capacity_per_head=512,
            source_dim=768,
            dtype=dtype,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _support in range(supports):
            ring.append_many(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rows.append(time.perf_counter() - started)
    return statistics.median(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    devices = [torch.device("cpu")]
    if args.device in {"cuda", "both"} and torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested without CUDA")
    rows: list[dict[str, object]] = []
    for supports in (1, 8, 16, 32):
        for slots in (1, 8, 32):
            legacy = _legacy_seconds(supports, slots, args.repeats)
            for device in devices:
                ring = _ring_seconds(supports, slots, args.repeats, device=device)
                rows.append(
                    {
                        "supports": supports,
                        "o2_slots": slots,
                        "device": device.type,
                        "legacy_seconds": legacy,
                        "ring_seconds": ring,
                        "speedup": legacy / ring,
                    }
                )
    payload = {"schema_version": 1, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
