#!/usr/bin/env python3
"""Build a deterministic A2 acceptance manifest with all eight operator subtypes.

The selected rows are also required to contain a label-side structural retrieval bag:
at least one Support interval overlaps an official causal occurrence and at least one
other Support interval does not.  This is a pre-forward guarantee only; the H200 run
still verifies the actual Bank candidates, masks, Projector gradient, and parameter
delta produced by the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    EpisodeSplit,
    ProductionEpisodeManifest,
    adaptive_support_schedule,
    load_production_episode_manifest,
    write_production_episode_manifest,
)
from ttt_svcbench_qwen.query_encoder import Operator
from ttt_svcbench_qwen.visual_cost import (
    VISUAL_COST_SCHEMA_VERSION,
    load_visual_cost_index,
    validate_visual_cost_fingerprint,
)

SUPPORTED_OPERATORS = tuple(
    operator.value for operator in Operator if operator is not Operator.UNSUPPORTED
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--visual-cost-index", required=True, type=Path)
    parser.add_argument("--train-per-subtype", type=int, default=4)
    parser.add_argument("--validation-per-subtype", type=int, default=1)
    return parser


def _occurrences(record: A2QueryRecord) -> tuple[tuple[float, float], ...]:
    query = record.query.weak
    ranges = [
        (point, point)
        for point in query.occurrence_points
        if point <= query.query_time
    ]
    ranges.extend(
        (start, min(end, query.query_time))
        for start, end in query.occurrence_intervals
        if start <= query.query_time
    )
    return tuple(ranges)


def _has_structural_retrieval_bag(record: A2QueryRecord) -> bool:
    occurrences = _occurrences(record)
    if not occurrences:
        return False
    prewarm, supports = adaptive_support_schedule(record.query.runtime.query_time)
    intervals = ((prewarm.start_time, prewarm.end_time),) + tuple(
        (chunk.start_time, chunk.end_time) for chunk in supports
    )
    positives = sum(
        any(
            left <= occurrence_end and occurrence_start <= right
            for occurrence_start, occurrence_end in occurrences
        )
        for left, right in intervals
    )
    return positives > 0 and positives < len(intervals)


def _select(
    manifest: ProductionEpisodeManifest,
    *,
    split: EpisodeSplit,
    per_subtype: int,
    allowed_record_ids: frozenset[str] | None,
) -> tuple[A2QueryRecord, ...]:
    if per_subtype <= 0:
        raise ValueError("per-subtype acceptance count must be positive")
    candidates: dict[str, list[A2QueryRecord]] = defaultdict(list)
    for record in manifest.a2_queries:
        if record.split is split and (
            allowed_record_ids is None
            or record.query.runtime.query_id in allowed_record_ids
        ):
            candidates[record.query.weak.operator].append(record)
    selected: list[A2QueryRecord] = []
    for operator in SUPPORTED_OPERATORS:
        rows = tuple(candidates.get(operator, ()))
        by_support_count: dict[int, list[A2QueryRecord]] = defaultdict(list)
        structurally_valid: dict[int, list[A2QueryRecord]] = defaultdict(list)
        for record in rows:
            support_count = len(
                adaptive_support_schedule(record.query.runtime.query_time)[1]
            )
            by_support_count[support_count].append(record)
            if _has_structural_retrieval_bag(record):
                structurally_valid[support_count].append(record)
        eligible = {
            count: values
            for count, values in structurally_valid.items()
            if len(values) >= per_subtype
        }
        if not eligible:
            # E2-periodic is expected here: its completed intervals tile the causal
            # prefix, so no official negative exists.  Keep the class for Router/Task
            # diagnostics while other batches provide Retrieval supervision.
            eligible = {
                count: values
                for count, values in by_support_count.items()
                if len(values) >= per_subtype
            }
        if not eligible:
            raise ValueError(
                f"{split.value}/{operator} has no single Support-count bucket with "
                f"{per_subtype} rows"
            )
        _, bucket = min(
            eligible.items(),
            key=lambda item: (item[0], tuple(row.query.runtime.query_id for row in item[1])),
        )
        selected.extend(
            sorted(bucket, key=lambda record: record.query.runtime.query_id)[:per_subtype]
        )
    return tuple(sorted(selected, key=lambda record: record.query.runtime.query_id))


def build_acceptance_manifest(
    manifest: ProductionEpisodeManifest,
    *,
    train_per_subtype: int,
    validation_per_subtype: int,
    measured_train_record_ids: frozenset[str] | None = None,
) -> ProductionEpisodeManifest:
    selected = _select(
        manifest,
        split=EpisodeSplit.TRAIN,
        per_subtype=train_per_subtype,
        allowed_record_ids=measured_train_record_ids,
    ) + _select(
        manifest,
        split=EpisodeSplit.VALIDATION,
        per_subtype=validation_per_subtype,
        allowed_record_ids=None,
    )
    task_counts = Counter(record.task_class for record in selected)
    return replace(
        manifest,
        a2_queries=selected,
        episodes=(),
        buckets=(),
        task_query_counts=tuple(sorted(task_counts.items())),
        failures=(),
    )


def _write_subset_visual_cost_index(
    source: Path,
    destination: Path,
    manifest_path: Path,
    manifest: ProductionEpisodeManifest,
) -> None:
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "fingerprint",
        "records",
    }:
        raise ValueError("source visual cost index must use strict schema 4")
    if raw["schema_version"] != VISUAL_COST_SCHEMA_VERSION:
        raise ValueError("source visual cost index schema is not supported")
    fingerprint = validate_visual_cost_fingerprint(raw["fingerprint"])
    source_records = load_visual_cost_index(source)
    selected_ids = {
        record.query.runtime.query_id
        for record in manifest.a2_queries
        if record.split is EpisodeSplit.TRAIN
    }
    missing = selected_ids - set(source_records)
    if missing:
        raise ValueError(
            "source visual cost index is missing acceptance rows: "
            + ", ".join(sorted(missing)[:8])
        )
    rows = raw["records"]
    if not isinstance(rows, list):
        raise ValueError("source visual cost records must be a list")
    selected_rows = [row for row in rows if row.get("record_id") in selected_ids]
    if len(selected_rows) != len(selected_ids):
        raise ValueError("subset visual cost rows are not one-to-one with the train manifest")
    fingerprint["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    destination.write_text(
        json.dumps(
            {
                "schema_version": VISUAL_COST_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "records": selected_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    load_visual_cost_index(destination, expected_fingerprint=fingerprint)


def main() -> None:
    args = _parser().parse_args()
    source = load_production_episode_manifest(args.manifest)
    output = build_acceptance_manifest(
        source,
        train_per_subtype=args.train_per_subtype,
        validation_per_subtype=args.validation_per_subtype,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / "dataset_manifest.json"
    write_production_episode_manifest(
        output,
        manifest_path=manifest_path,
        failed_path=args.output / "failed.jsonl",
    )
    visual_cost_path = args.output / "visual_cost_index.json"
    _write_subset_visual_cost_index(
        args.visual_cost_index,
        visual_cost_path,
        manifest_path,
        output,
    )
    counts = Counter(
        (record.split.value, record.query.weak.operator) for record in output.a2_queries
    )
    structural_counts = Counter(
        (record.split.value, record.query.weak.operator)
        for record in output.a2_queries
        if _has_structural_retrieval_bag(record)
    )
    summary = {
        "source_manifest": str(args.manifest.resolve()),
        "output_manifest": str(manifest_path.resolve()),
        "visual_cost_index": str(visual_cost_path.resolve()),
        "counts": {"/".join(key): value for key, value in sorted(counts.items())},
        "structural_retrieval_bag_counts": {
            "/".join(key): value for key, value in sorted(structural_counts.items())
        },
        "e2_periodic_has_no_structural_negative_by_definition": True,
    }
    (args.output / "acceptance_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"manifest={manifest_path}")
    for key, count in sorted(counts.items()):
        print(
            f"{key[0]}/{key[1]}={count} "
            f"structural_bag={structural_counts.get(key, 0)}"
        )


if __name__ == "__main__":
    main()
