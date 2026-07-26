#!/usr/bin/env python3
"""Benchmark row-wise and vectorized writes to the retrieval-history tensor ring."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ttt_svcbench_qwen.query_encoder import OPERATORS, Operator
from ttt_svcbench_qwen.state_bank import RetrievalHistoryAppendBatch, TensorizedRetrievalHistory


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


def _row_batches(batch: RetrievalHistoryAppendBatch) -> tuple[RetrievalHistoryAppendBatch, ...]:
    return tuple(
        RetrievalHistoryAppendBatch(
            sources=batch.sources[index : index + 1],
            head_codes=batch.head_codes[index : index + 1],
            operator_codes=batch.operator_codes[index : index + 1],
            timestamps=batch.timestamps[index : index + 1],
            time_ranges=batch.time_ranges[index : index + 1],
            valid_mask=batch.valid_mask[index : index + 1],
            eligible_mask=batch.eligible_mask[index : index + 1],
        )
        for index in range(batch.sources.shape[0])
    )


def _write_seconds(
    supports: int,
    slots: int,
    repeats: int,
    *,
    device: torch.device,
    vectorized: bool,
) -> float:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    batch = _batch(slots, dtype=dtype, device=device)
    writes = (batch,) if vectorized else _row_batches(batch)
    samples: list[float] = []
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
            for write in writes:
                ring.append_many(write)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")

    devices = [torch.device("cpu")]
    if args.device in {"cuda", "both"} and torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requested without CUDA")
        devices = [torch.device("cuda")]

    rows: list[dict[str, object]] = []
    for supports in (1, 8, 16, 32):
        for slots in (1, 8, 32):
            for device in devices:
                rowwise = _write_seconds(
                    supports,
                    slots,
                    args.repeats,
                    device=device,
                    vectorized=False,
                )
                batched = _write_seconds(
                    supports,
                    slots,
                    args.repeats,
                    device=device,
                    vectorized=True,
                )
                rows.append(
                    {
                        "supports": supports,
                        "o2_slots": slots,
                        "device": device.type,
                        "rowwise_seconds": rowwise,
                        "batched_seconds": batched,
                        "speedup": rowwise / batched,
                    }
                )

    payload = {"schema_version": 2, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
