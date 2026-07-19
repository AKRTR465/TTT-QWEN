"""Strict schema-2 visual/runtime cost index used for rank-aligned sampling."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch.distributed as dist

VISUAL_COST_SCHEMA_VERSION = 2
FINGERPRINT_FIELDS = frozenset(
    {
        "manifest_sha256",
        "model_revision",
        "transformers_version",
        "processor",
        "minimum_pixels",
        "maximum_pixels",
        "dtype",
        "visual_batch_size",
        "cache_mode",
        "loss_mode",
        "loss_group_weight",
        "loss_scale_min",
        "loss_scale_max",
        "loss_epsilon",
        "gpu_model",
    }
)


def make_visual_cost_fingerprint(
    *,
    manifest_sha256: str,
    model_revision: str,
    transformers_version: str,
    processor: str,
    minimum_pixels: int,
    maximum_pixels: int,
    dtype: str,
    visual_batch_size: int,
    cache_mode: str,
    loss_mode: str,
    loss_group_weight: float,
    loss_scale_min: float,
    loss_scale_max: float,
    loss_epsilon: float,
    gpu_model: str,
) -> dict[str, object]:
    return validate_visual_cost_fingerprint(locals())


@dataclass(frozen=True, slots=True)
class VisualCostRecord:
    record_id: str
    support_count: int
    segment_lengths: tuple[int, ...]
    query_count: int
    visual_tokens: tuple[int, ...]
    total_visual_tokens: int
    maximum_visual_tokens: int
    decode_seconds: float
    processor_seconds: float
    vit_seconds: float
    query_seconds: float
    loss_collective_seconds: float
    predicted_total_seconds: float

    def __post_init__(self) -> None:
        if not self.record_id:
            raise ValueError("visual cost record_id must be non-empty")
        if self.support_count < 0 or self.query_count <= 0:
            raise ValueError("visual cost Support/Query counts are invalid")
        if any(value <= 0 for value in self.segment_lengths):
            raise ValueError("visual cost segment lengths must be positive")
        if sum(self.segment_lengths) not in (0, self.support_count):
            raise ValueError("visual cost segment lengths must sum to support_count")
        if not self.visual_tokens or any(value <= 0 for value in self.visual_tokens):
            raise ValueError("visual cost per-chunk token counts must be positive")
        if self.total_visual_tokens != sum(self.visual_tokens):
            raise ValueError("visual cost total tokens must sum per-chunk tokens")
        if self.maximum_visual_tokens != max(self.visual_tokens):
            raise ValueError("visual cost maximum tokens must match per-chunk tokens")
        times = (
            self.decode_seconds,
            self.processor_seconds,
            self.vit_seconds,
            self.query_seconds,
            self.loss_collective_seconds,
            self.predicted_total_seconds,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in times):
            raise ValueError("visual cost times must be finite and non-negative")

    @property
    def sort_key(self) -> tuple[float, int, int]:
        return (
            self.predicted_total_seconds,
            self.total_visual_tokens,
            self.maximum_visual_tokens,
        )


class EpochBoundaryCostEMA:
    """Collect step costs during an epoch and publish alpha=0.2 updates at boundaries."""

    def __init__(self, initial: Mapping[str, float], *, alpha: float = 0.2) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("runtime cost EMA alpha must be in (0, 1]")
        self.alpha = alpha
        self.epoch = 0
        self._values = {key: float(value) for key, value in initial.items()}
        self._pending: dict[str, list[float]] = defaultdict(list)

    def observe(self, record_id: str, seconds: float) -> None:
        if record_id not in self._values:
            return
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError("runtime cost observation must be finite and non-negative")
        self._pending[record_id].append(seconds)

    def advance_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < self.epoch:
            raise ValueError("runtime cost EMA epoch must advance monotonically")
        if epoch == self.epoch:
            return
        local = {
            key: (sum(values), len(values)) for key, values in self._pending.items()
        }
        gathered: list[dict[str, tuple[float, int]]]
        if dist.is_available() and dist.is_initialized():
            gathered_objects: list[object] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_objects, local)
            gathered = [
                value
                for value in gathered_objects
                if isinstance(value, dict)
            ]
        else:
            gathered = [local]
        merged: dict[str, tuple[float, int]] = {}
        for rows in gathered:
            for key, (total, count) in rows.items():
                previous_total, previous_count = merged.get(key, (0.0, 0))
                merged[key] = previous_total + total, previous_count + count
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            for key, (total, count) in merged.items():
                if count:
                    measured = total / count
                    self._values[key] = (
                        (1.0 - self.alpha) * self._values[key]
                        + self.alpha * measured
                    )
        if dist.is_available() and dist.is_initialized():
            payload: list[object] = [self._values if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(payload, src=0)
            broadcast = payload[0]
            if not isinstance(broadcast, dict):
                raise RuntimeError("runtime cost EMA broadcast returned an invalid payload")
            self._values = {str(key): float(value) for key, value in broadcast.items()}
        self._pending.clear()
        self.epoch = epoch

    def value(self, record_id: str, fallback: float) -> float:
        return self._values.get(record_id, fallback)


def validate_visual_cost_fingerprint(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != FINGERPRINT_FIELDS:
        raise ValueError("visual cost fingerprint fields do not match schema 2")
    result = {str(key): item for key, item in value.items()}
    string_fields = (
        "manifest_sha256",
        "model_revision",
        "transformers_version",
        "processor",
        "dtype",
        "cache_mode",
        "loss_mode",
        "gpu_model",
    )
    for field in string_fields:
        item = result[field]
        if not isinstance(item, str) or not item:
            raise ValueError(f"visual cost fingerprint {field} must be non-empty")
    for field in ("minimum_pixels", "maximum_pixels", "visual_batch_size"):
        item = result[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"visual cost fingerprint {field} must be positive")
    if int(result["minimum_pixels"]) > int(result["maximum_pixels"]):
        raise ValueError("visual cost fingerprint pixel bounds are invalid")
    for field in ("loss_group_weight", "loss_scale_min", "loss_scale_max", "loss_epsilon"):
        item = result[field]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"visual cost fingerprint {field} must be numeric")
        if not math.isfinite(float(item)) or float(item) <= 0.0:
            raise ValueError(f"visual cost fingerprint {field} must be positive")
    return result


def load_visual_cost_index(
    path: str | Path,
    *,
    expected_fingerprint: Mapping[str, object] | None = None,
) -> dict[str, VisualCostRecord]:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "fingerprint",
        "records",
    }:
        raise ValueError("visual cost index must use strict schema 2")
    if raw["schema_version"] != VISUAL_COST_SCHEMA_VERSION:
        raise ValueError("visual cost index schema_version must be 2")
    fingerprint = validate_visual_cost_fingerprint(raw["fingerprint"])
    if expected_fingerprint is not None:
        expected = validate_visual_cost_fingerprint(expected_fingerprint)
        if fingerprint != expected:
            differing = sorted(
                key for key in FINGERPRINT_FIELDS if fingerprint[key] != expected[key]
            )
            raise ValueError(
                "visual cost fingerprint mismatch: " + ", ".join(differing)
            )
    rows = raw["records"]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("visual cost records must be a list")
    result: dict[str, VisualCostRecord] = {}
    for value in rows:
        record = _parse_visual_cost_record(value)
        if record.record_id in result:
            raise ValueError(f"duplicate visual cost record: {record.record_id}")
        result[record.record_id] = record
    if not result:
        raise ValueError("visual cost index must contain at least one record")
    return result


def _parse_visual_cost_record(value: object) -> VisualCostRecord:
    fields = set(VisualCostRecord.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("visual cost record fields do not match schema 2")
    record_id = value["record_id"]
    if not isinstance(record_id, str):
        raise ValueError("visual cost record_id must be a string")
    try:
        return VisualCostRecord(
            record_id=record_id,
            support_count=_integer(value["support_count"], "support_count"),
            segment_lengths=_integer_tuple(value["segment_lengths"], "segment_lengths"),
            query_count=_integer(value["query_count"], "query_count"),
            visual_tokens=_integer_tuple(value["visual_tokens"], "visual_tokens"),
            total_visual_tokens=_integer(value["total_visual_tokens"], "total_visual_tokens"),
            maximum_visual_tokens=_integer(
                value["maximum_visual_tokens"], "maximum_visual_tokens"
            ),
            decode_seconds=_number(value["decode_seconds"], "decode_seconds"),
            processor_seconds=_number(value["processor_seconds"], "processor_seconds"),
            vit_seconds=_number(value["vit_seconds"], "vit_seconds"),
            query_seconds=_number(value["query_seconds"], "query_seconds"),
            loss_collective_seconds=_number(
                value["loss_collective_seconds"], "loss_collective_seconds"
            ),
            predicted_total_seconds=_number(
                value["predicted_total_seconds"], "predicted_total_seconds"
            ),
        )
    except KeyError as error:  # pragma: no cover - exact field set above
        raise ValueError("visual cost record is incomplete") from error


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"visual cost {field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"visual cost {field} must be numeric")
    return float(value)


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"visual cost {field} must be a list")
    return tuple(_integer(item, field) for item in value)


__all__ = [
    "FINGERPRINT_FIELDS",
    "EpochBoundaryCostEMA",
    "VISUAL_COST_SCHEMA_VERSION",
    "VisualCostRecord",
    "load_visual_cost_index",
    "make_visual_cost_fingerprint",
    "validate_visual_cost_fingerprint",
]
