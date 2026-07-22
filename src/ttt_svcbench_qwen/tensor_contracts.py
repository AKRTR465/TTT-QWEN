"""Shared tensor identity and timestamp comparison contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass

import torch
from torch import Tensor


def tensor_storage_key(tensor: Tensor) -> tuple[str, int | None, int]:
    """Return a device-qualified storage key, preserving meta-tensor identity."""

    if tensor.device.type == "meta":
        return ("meta", None, id(tensor))
    return (
        tensor.device.type,
        tensor.device.index,
        int(tensor.untyped_storage().data_ptr()),
    )


def timestamps_match(left: Tensor, right: Tensor) -> bool:
    """Compare source-identical timestamps across legal FP32/FP64 handoffs."""

    if left.shape != right.shape:
        return False
    left_64 = left.to(dtype=torch.float64)
    right_64 = right.to(dtype=torch.float64)
    scale = torch.maximum(left_64.abs(), right_64.abs()).clamp_min(1.0)
    tolerance = 4.0 * torch.finfo(torch.float32).eps * scale
    return bool(torch.all((left_64 - right_64).abs() <= tolerance))


def validate_finite_tensor_tree(value: object, name: str) -> None:
    """Scan a typed runtime tree only at an explicit pack/commit/segment boundary."""

    tensors: list[Tensor] = []
    _collect_tensors(value, tensors, set())
    for tensor in tensors:
        if (
            tensor.device.type != "meta"
            and (torch.is_floating_point(tensor) or torch.is_complex(tensor))
            and not bool(torch.isfinite(tensor.detach()).all())
        ):
            raise ValueError(f"{name} contains a nonfinite tensor")


def _collect_tensors(value: object, found: list[Tensor], seen: set[int]) -> None:
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, Tensor):
        found.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            _collect_tensors(item, found, seen)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for item in value:
            _collect_tensors(item, found, seen)
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _collect_tensors(getattr(value, field.name), found, seen)
