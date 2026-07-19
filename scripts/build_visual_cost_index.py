"""Build a strict schema-2 A2/A5 visual and runtime cost index.

The default timing coefficients are zero, so this command can create a token-only preflight
index. H200 calibration should pass measured coefficients or replace rows from a trace summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import transformers

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    A5EpisodeRecord,
    ManifestStage,
    adaptive_support_schedule,
    load_production_manifest_views,
)
from ttt_svcbench_qwen.visual_cost import (
    VISUAL_COST_SCHEMA_VERSION,
    VisualCostRecord,
    make_visual_cost_fingerprint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("a2", "a5"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-config", type=Path, default=None)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--processor", required=True)
    parser.add_argument("--minimum-pixels", required=True, type=int)
    parser.add_argument("--maximum-pixels", required=True, type=int)
    parser.add_argument("--dtype", required=True, choices=("bfloat16", "float32"))
    parser.add_argument("--visual-batch-size", required=True, type=int)
    parser.add_argument(
        "--cache-mode",
        required=True,
        choices=("disabled", "read_write", "readonly"),
    )
    parser.add_argument("--gpu-model", required=True)
    parser.add_argument("--decode-seconds-per-chunk", type=float, default=0.0)
    parser.add_argument("--processor-seconds-per-chunk", type=float, default=0.0)
    parser.add_argument("--vit-seconds-per-token", type=float, default=0.0)
    parser.add_argument("--query-seconds-per-query", type=float, default=0.0)
    parser.add_argument("--loss-collective-seconds", type=float, default=0.0)
    args = parser.parse_args()

    project = load_config(args.project_config)
    balance = project.loss.official_weak_balance
    fingerprint = make_visual_cost_fingerprint(
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        model_revision=args.model_revision,
        transformers_version=transformers.__version__,
        processor=args.processor,
        minimum_pixels=args.minimum_pixels,
        maximum_pixels=args.maximum_pixels,
        dtype=args.dtype,
        visual_batch_size=args.visual_batch_size,
        cache_mode=args.cache_mode,
        loss_mode=balance.mode.value,
        loss_group_weight=balance.group_weight,
        loss_scale_min=balance.scale_min,
        loss_scale_max=balance.scale_max,
        loss_epsilon=balance.epsilon,
        gpu_model=args.gpu_model,
    )
    train, validation = load_production_manifest_views(
        args.manifest,
        stage=ManifestStage(args.stage),
    )
    records = tuple(train.records) + tuple(validation.records)
    rows = [
        _cost_record(
            record,
            decode_per_chunk=args.decode_seconds_per_chunk,
            processor_per_chunk=args.processor_seconds_per_chunk,
            vit_per_token=args.vit_seconds_per_token,
            query_per_query=args.query_seconds_per_query,
            loss_collective=args.loss_collective_seconds,
        )
        for record in records
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": VISUAL_COST_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "records": [asdict(row) for row in rows],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} strict visual-cost rows to {args.output}")
    return 0


def _cost_record(
    record: object,
    *,
    decode_per_chunk: float,
    processor_per_chunk: float,
    vit_per_token: float,
    query_per_query: float,
    loss_collective: float,
) -> VisualCostRecord:
    coefficients = (
        decode_per_chunk,
        processor_per_chunk,
        vit_per_token,
        query_per_query,
        loss_collective,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in coefficients):
        raise ValueError("cost coefficients must be finite and non-negative")
    if isinstance(record, A2QueryRecord):
        _, supports = adaptive_support_schedule(record.query.runtime.query_time)
        intervals = tuple(
            (chunk.start_time, chunk.end_time, chunk.maximum_frames) for chunk in supports
        ) + (
            (
                max(0.0, record.query.runtime.query_time - 8.0),
                record.query.runtime.query_time,
                16,
            ),
        )
        record_id = record.query.runtime.query_id
        support_count = len(supports)
        segment_lengths: tuple[int, ...] = ()
        query_count = 1
    elif isinstance(record, A5EpisodeRecord):
        intervals = (
            (record.prewarm.start_time, record.prewarm.end_time, record.prewarm.maximum_frames),
            *(
                (chunk.start_time, chunk.end_time, chunk.maximum_frames)
                for chunk in record.supports
            ),
            *(
                (
                    max(0.0, query.runtime.query_time - 8.0),
                    query.runtime.query_time,
                    16,
                )
                for query in record.queries
            ),
        )
        record_id = record.episode_id
        support_count = record.support_count
        segment_lengths = _segment_lengths(record)
        query_count = record.query_count
    else:
        raise TypeError("visual cost builder received an unknown manifest record")
    visual_tokens = tuple(_frame_budget(*interval) for interval in intervals)
    chunk_count = len(visual_tokens)
    total_tokens = sum(visual_tokens)
    decode_seconds = decode_per_chunk * chunk_count
    processor_seconds = processor_per_chunk * chunk_count
    vit_seconds = vit_per_token * total_tokens
    query_seconds = query_per_query * query_count
    predicted = (
        decode_seconds
        + processor_seconds
        + vit_seconds
        + query_seconds
        + loss_collective
    )
    return VisualCostRecord(
        record_id=record_id,
        support_count=support_count,
        segment_lengths=segment_lengths,
        query_count=query_count,
        visual_tokens=visual_tokens,
        total_visual_tokens=total_tokens,
        maximum_visual_tokens=max(visual_tokens),
        decode_seconds=decode_seconds,
        processor_seconds=processor_seconds,
        vit_seconds=vit_seconds,
        query_seconds=query_seconds,
        loss_collective_seconds=loss_collective,
        predicted_total_seconds=predicted,
    )


def _frame_budget(start: float, end: float, maximum: int) -> int:
    desired = min(maximum, max(2, int(math.floor((end - start) * 2.0)) + 1))
    return max(2, desired - desired % 2)


def _segment_lengths(record: A5EpisodeRecord) -> tuple[int, ...]:
    remaining = record.support_count
    values: list[int] = []
    while remaining:
        length = min(record.truncation_horizon, remaining)
        values.append(length)
        remaining -= length
    return tuple(values)


if __name__ == "__main__":
    raise SystemExit(main())
