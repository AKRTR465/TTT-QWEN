"""Shared tensor identity and timestamp comparison contracts."""

from __future__ import annotations

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
