"""Implement query-conditioned spatial slots and causal tubelet event states.

Inputs: adapted visual tokens, q_target, masks, grid metadata, and prior per-video state.
Outputs: recurrent spatial slots, causal temporal states, and functional runtime caches.
Forbidden: hard counting, semantic overflow inference, Bank mutation, or optimizer steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import (
    ProjectConfig,
    SpatialEncoderConfig,
    TemporalEncoderConfig,
)
from ttt_svcbench_qwen.qwen_adapter import MergedVideoMetadata


@dataclass(frozen=True, slots=True)
class RestoredMergedGrid:
    """Padded heterogeneous Main-Merger grids and their effective validity masks."""

    tokens: Tensor
    geometry_valid_mask: Tensor
    spatial_valid_mask: Tensor
    tubelet_valid_mask: Tensor
    grid_shapes: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if (
            self.tokens.ndim != 5
            or self.tokens.shape[-1] != 4096
            or not torch.is_floating_point(self.tokens)
        ):
            raise ValueError("restored grid tokens must be floating [B, T, H, W, 4096]")
        if (
            self.geometry_valid_mask.shape != self.tokens.shape[:4]
            or self.geometry_valid_mask.dtype != torch.bool
        ):
            raise ValueError("geometry_valid_mask must be bool [B, T, H, W]")
        if (
            self.spatial_valid_mask.shape != self.tokens.shape[:4]
            or self.spatial_valid_mask.dtype != torch.bool
        ):
            raise ValueError("spatial_valid_mask must be bool [B, T, H, W]")
        if (
            self.tubelet_valid_mask.shape != self.tokens.shape[:2]
            or self.tubelet_valid_mask.dtype != torch.bool
        ):
            raise ValueError("tubelet_valid_mask must be bool [B, T]")
        if len(self.grid_shapes) != self.tokens.shape[0]:
            raise ValueError("restored grid requires one shape per batch row")
        if any(len(shape) != 3 or min(shape) <= 0 for shape in self.grid_shapes):
            raise ValueError("restored grid shapes must contain positive (T, H, W) values")
        if (
            self.tokens.device != self.geometry_valid_mask.device
            or self.tokens.device != self.spatial_valid_mask.device
            or self.tokens.device != self.tubelet_valid_mask.device
        ):
            raise ValueError("restored grid tensors must share one device")
        if self.tokens.device.type != "meta":
            expected_spatial = self.geometry_valid_mask & self.tubelet_valid_mask[:, :, None, None]
            if not torch.equal(self.spatial_valid_mask, expected_spatial):
                raise ValueError("spatial mask must combine geometry and tubelet validity")


@dataclass(frozen=True, slots=True)
class SpatialSlotRuntimeState:
    """One video's functional recurrent slots; this state is never module-owned."""

    video_id: str
    slots: Tensor
    slot_valid_mask: Tensor
    slot_confidence: Tensor
    active_slot_overflow_count: int
    overflow_event_count: int
    processed_tubelets: int
    differentiable: bool = False

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("spatial slot runtime requires a non-empty video_id")
        if self.slots.ndim != 2 or not 1 <= self.slots.shape[0] <= 64 or self.slots.shape[1] != 768:
            raise ValueError("runtime slots must be [K_a, 768] with 1 <= K_a <= 64")
        if not torch.is_floating_point(self.slots):
            raise TypeError("runtime slots must use a floating dtype")
        if (
            self.slot_valid_mask.shape != self.slots.shape[:1]
            or self.slot_valid_mask.dtype != torch.bool
        ):
            raise ValueError("runtime slot_valid_mask must be bool [K_a]")
        if self.slot_confidence.shape != self.slots.shape[:1] or not torch.is_floating_point(
            self.slot_confidence
        ):
            raise ValueError("runtime slot_confidence must be floating [K_a]")
        if (
            self.slots.device != self.slot_valid_mask.device
            or self.slots.device != self.slot_confidence.device
            or self.slots.dtype != self.slot_confidence.dtype
        ):
            raise ValueError("runtime slot tensors must share dtype/device")
        if self.slots.device.type != "meta" and not bool(self.slot_valid_mask.any()):
            raise ValueError("spatial runtime requires at least one valid slot")
        if _shares_storage(self.slots, self.slot_confidence):
            raise ValueError("runtime slots and confidence must use distinct storage")
        if self.slots.device.type != "meta":
            if not bool(torch.isfinite(self.slots).all()):
                raise ValueError("runtime slots must be finite")
            if not bool(torch.isfinite(self.slot_confidence).all()):
                raise ValueError("runtime slot confidence must be finite")
            if bool(torch.any((self.slot_confidence < 0.0) | (self.slot_confidence > 1.0))):
                raise ValueError("runtime slot confidence must stay within [0, 1]")
            if bool(torch.any(self.slot_confidence[~self.slot_valid_mask] != 0.0)):
                raise ValueError("invalid runtime slots must have zero confidence")
        counters = (
            self.active_slot_overflow_count,
            self.overflow_event_count,
            self.processed_tubelets,
        )
        if any(type(value) is not int for value in counters):
            raise TypeError("spatial runtime counters must be exact integers")
        if min(counters) < 0:
            raise ValueError("spatial runtime counters must be non-negative")
        if type(self.differentiable) is not bool:
            raise TypeError("spatial runtime differentiable must be a bool")
        if not self.differentiable and (
            self.slots.requires_grad or self.slot_confidence.requires_grad
        ):
            raise ValueError("non-differentiable spatial runtime tensors must be detached")


@dataclass(frozen=True, slots=True)
class SpatialEncoderAudit:
    """Detached structural evidence for one spatial encoder call."""

    grid_shapes: tuple[tuple[int, int, int], ...]
    visual_token_counts: tuple[int, ...]
    valid_tubelet_counts: tuple[int, ...]
    required_slot_counts: tuple[int, ...]
    excess_slot_counts: tuple[int, ...]
    overflow_events: tuple[bool, ...]
    overflow_policy: str
    stage_refinement_calls: tuple[int, int]

    def __post_init__(self) -> None:
        batch_size = len(self.grid_shapes)
        fields = (
            self.visual_token_counts,
            self.valid_tubelet_counts,
            self.required_slot_counts,
            self.excess_slot_counts,
            self.overflow_events,
        )
        if batch_size <= 0 or any(len(field) != batch_size for field in fields):
            raise ValueError("spatial audit fields must align to one non-empty batch")
        counters = (
            *self.visual_token_counts,
            *self.valid_tubelet_counts,
            *self.required_slot_counts,
            *self.excess_slot_counts,
            *self.stage_refinement_calls,
        )
        if any(type(value) is not int for value in counters):
            raise TypeError("spatial audit counters must be exact integers")
        if min(counters) < 0:
            raise ValueError("spatial audit counters must be non-negative")
        if any(type(value) is not bool for value in self.overflow_events):
            raise TypeError("spatial audit overflow flags must be bool")
        if not self.overflow_policy:
            raise ValueError("spatial audit requires an overflow policy")


@dataclass(frozen=True, slots=True)
class SpatialEncoderOutput:
    slots: Tensor
    slot_valid_mask: Tensor
    active_slot_overflow_count: Tensor
    slot_confidence: Tensor | None = None
    next_states: tuple[SpatialSlotRuntimeState, ...] | None = None
    audit: SpatialEncoderAudit | None = None

    def __post_init__(self) -> None:
        if self.slots.ndim != 3 or not 1 <= self.slots.shape[1] <= 64 or self.slots.shape[2] != 768:
            raise ValueError("slots must be [B, K_a, 768] with 1 <= K_a <= 64")
        if not torch.is_floating_point(self.slots):
            raise TypeError("slots must use a floating dtype")
        expected_mask = self.slots.shape[:2]
        if self.slot_valid_mask.shape != expected_mask or self.slot_valid_mask.dtype != torch.bool:
            raise ValueError("slot_valid_mask must be bool [B, K_a]")
        if self.active_slot_overflow_count.shape != (self.slots.shape[0],):
            raise ValueError("active_slot_overflow_count must be [B]")
        if self.active_slot_overflow_count.dtype not in (torch.int32, torch.int64):
            raise TypeError("active_slot_overflow_count must use an integer dtype")
        if (
            self.slots.device != self.slot_valid_mask.device
            or self.slots.device != self.active_slot_overflow_count.device
        ):
            raise ValueError("spatial output tensors must share one device")
        if self.slots.device.type != "meta":
            if not bool(torch.isfinite(self.slots).all()):
                raise ValueError("spatial output slots must be finite")
            if bool(torch.any(self.active_slot_overflow_count < 0)):
                raise ValueError("spatial output overflow counts must be non-negative")
        if self.slot_confidence is not None:
            if (
                self.slot_confidence.shape != expected_mask
                or self.slot_confidence.dtype != self.slots.dtype
                or self.slot_confidence.device != self.slots.device
            ):
                raise ValueError("slot_confidence must match slots as floating [B, K_a]")
            if self.slots.device.type != "meta":
                if not bool(torch.isfinite(self.slot_confidence).all()):
                    raise ValueError("slot_confidence must be finite")
                if bool(torch.any((self.slot_confidence < 0.0) | (self.slot_confidence > 1.0))):
                    raise ValueError("slot_confidence must stay within [0, 1]")
                if bool(torch.any(self.slot_confidence[~self.slot_valid_mask] != 0.0)):
                    raise ValueError("invalid slots must have zero confidence")
        if self.next_states is not None:
            if len(self.next_states) != self.slots.shape[0]:
                raise ValueError("spatial output requires one next state per batch row")
            _assert_runtime_state_storage_isolated(self.next_states)


@dataclass(frozen=True, slots=True)
class TemporalCache:
    """Functional batched cache containing every layer's causal K/V state."""

    hidden: Tensor
    layer_keys: tuple[Tensor, ...]
    layer_values: tuple[Tensor, ...]
    replay_layer_keys: tuple[Tensor, ...]
    replay_layer_values: tuple[Tensor, ...]
    timestamps: Tensor
    replay_timestamps: Tensor
    position_ids: Tensor
    replay_position_ids: Tensor
    valid_mask: Tensor
    replay_valid_mask: Tensor
    video_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]
    query_signatures: Tensor
    total_seen: Tensor
    differentiable: bool = False

    def __post_init__(self) -> None:
        if (
            self.hidden.ndim != 3
            or self.hidden.shape[-1] != 768
            or not torch.is_floating_point(self.hidden)
        ):
            raise ValueError("temporal cache hidden must be [B, T_cache, 768]")
        if self.hidden.shape[1] > 64:
            raise ValueError("temporal cache cannot exceed 64 tubelets")
        batch_size, cache_length = self.hidden.shape[:2]
        shape = (batch_size, cache_length)
        if len(self.layer_keys) != 6 or len(self.layer_values) != 6:
            raise ValueError("temporal cache requires six independent K/V layers")
        if len(self.replay_layer_keys) != 6 or len(self.replay_layer_values) != 6:
            raise ValueError("temporal cache requires six replay-context K/V layers")
        expected_kv = (batch_size, 12, cache_length, 64)
        for layer_index, (keys, values) in enumerate(
            zip(self.layer_keys, self.layer_values, strict=True)
        ):
            if (
                keys.shape != expected_kv
                or values.shape != expected_kv
                or not torch.is_floating_point(keys)
                or not torch.is_floating_point(values)
            ):
                raise ValueError(
                    f"temporal cache layer {layer_index} K/V must be [B, 12, T_cache, 64]"
                )
            if (
                keys.dtype != self.hidden.dtype
                or values.dtype != self.hidden.dtype
                or keys.device != self.hidden.device
                or values.device != self.hidden.device
            ):
                raise ValueError("temporal cache hidden and all K/V must share dtype/device")
        if self.replay_position_ids.ndim != 2:
            raise ValueError("temporal replay position_ids must be [B, T_replay]")
        replay_length = self.replay_position_ids.shape[1]
        if replay_length > 3:
            raise ValueError("temporal replay context cannot exceed three tubelets")
        expected_replay_kv = (batch_size, 12, replay_length, 64)
        for layer_index, (keys, values) in enumerate(
            zip(self.replay_layer_keys, self.replay_layer_values, strict=True)
        ):
            if (
                keys.shape != expected_replay_kv
                or values.shape != expected_replay_kv
                or not torch.is_floating_point(keys)
                or not torch.is_floating_point(values)
            ):
                raise ValueError(
                    f"temporal replay layer {layer_index} K/V must be [B, 12, T_replay, 64]"
                )
            if (
                keys.dtype != self.hidden.dtype
                or values.dtype != self.hidden.dtype
                or keys.device != self.hidden.device
                or values.device != self.hidden.device
            ):
                raise ValueError("temporal replay K/V must share hidden dtype/device")
        if (
            self.timestamps.shape != shape
            or self.timestamps.dtype != torch.float64
            or self.timestamps.device != self.hidden.device
        ):
            raise ValueError("temporal cache timestamps must be floating [B, T_cache]")
        replay_shape = (batch_size, replay_length)
        if (
            self.replay_timestamps.shape != replay_shape
            or self.replay_timestamps.dtype != torch.float64
            or self.replay_timestamps.device != self.hidden.device
        ):
            raise ValueError("temporal replay timestamps must match cache metadata")
        if (
            self.position_ids.shape != shape
            or self.position_ids.dtype != torch.int64
            or self.position_ids.device != self.hidden.device
        ):
            raise ValueError("temporal cache position_ids must be int64 [B, T_cache]")
        if (
            self.replay_position_ids.shape != replay_shape
            or self.replay_position_ids.dtype != torch.int64
            or self.replay_position_ids.device != self.hidden.device
        ):
            raise ValueError("temporal replay position_ids must be int64 [B, T_replay]")
        if (
            self.valid_mask.shape != shape
            or self.valid_mask.dtype != torch.bool
            or self.valid_mask.device != self.hidden.device
        ):
            raise ValueError("temporal cache valid_mask must be bool [B, T_cache]")
        if (
            self.replay_valid_mask.shape != replay_shape
            or self.replay_valid_mask.dtype != torch.bool
            or self.replay_valid_mask.device != self.hidden.device
        ):
            raise ValueError("temporal replay valid_mask must be bool [B, T_replay]")
        if len(self.video_ids) != batch_size or not all(self.video_ids):
            raise ValueError("temporal cache requires one non-empty video_id per batch item")
        if len(set(self.video_ids)) != batch_size:
            raise ValueError("temporal cache cannot contain duplicate video_ids")
        if len(self.trajectory_ids) != batch_size or not all(self.trajectory_ids):
            raise ValueError("temporal cache requires one non-empty trajectory_id per batch item")
        if (
            self.query_signatures.shape != (batch_size, 512)
            or not torch.is_floating_point(self.query_signatures)
            or self.query_signatures.dtype != self.hidden.dtype
            or self.query_signatures.device != self.hidden.device
        ):
            raise ValueError("temporal query_signatures must match hidden as [B, 512]")
        if (
            self.total_seen.shape != (batch_size,)
            or self.total_seen.dtype != torch.int64
            or self.total_seen.device != self.hidden.device
        ):
            raise ValueError("temporal total_seen must be int64 [B]")
        if type(self.differentiable) is not bool:
            raise TypeError("temporal cache differentiable must be a bool")
        floating_cache = (
            self.hidden,
            self.timestamps,
            self.replay_timestamps,
            self.query_signatures,
            *self.layer_keys,
            *self.layer_values,
            *self.replay_layer_keys,
            *self.replay_layer_values,
        )
        if not self.differentiable and any(tensor.requires_grad for tensor in floating_cache):
            raise ValueError("non-differentiable temporal cache tensors must be detached")
        if self.hidden.device.type == "meta":
            return
        cache_tensors = self._storage_tensors()
        for left_index, left in enumerate(cache_tensors):
            if left.numel() == 0:
                continue
            for right in cache_tensors[left_index + 1 :]:
                if right.numel() and _shares_storage(left, right):
                    raise ValueError("temporal cache fields must use independent storage")
        if not bool(torch.isfinite(self.query_signatures).all()):
            raise ValueError("temporal query signatures must be finite")
        if bool(torch.any(self.total_seen < 0)):
            raise ValueError("temporal total_seen must be non-negative")
        if cache_length > 1 and bool(torch.any(self.valid_mask[:, 1:] & ~self.valid_mask[:, :-1])):
            raise ValueError("temporal cache valid_mask must be a valid prefix")
        if replay_length > 1 and bool(
            torch.any(self.replay_valid_mask[:, 1:] & ~self.replay_valid_mask[:, :-1])
        ):
            raise ValueError("temporal replay valid_mask must be a valid prefix")
        for tensor in (
            self.hidden,
            *self.layer_keys,
            *self.layer_values,
            *self.replay_layer_keys,
            *self.replay_layer_values,
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("temporal cache hidden and K/V must be finite")
        for row in range(batch_size):
            valid_count = int(self.valid_mask[row].sum().item())
            replay_count = int(self.replay_valid_mask[row].sum().item())
            valid_timestamps = self.timestamps[row, :valid_count]
            valid_positions = self.position_ids[row, :valid_count]
            replay_timestamps = self.replay_timestamps[row, :replay_count]
            replay_positions = self.replay_position_ids[row, :replay_count]
            if valid_count:
                if not bool(torch.isfinite(valid_timestamps).all()) or bool(
                    torch.any(valid_timestamps < 0.0)
                ):
                    raise ValueError(
                        "valid temporal cache timestamps must be finite and non-negative"
                    )
                if valid_count > 1 and (
                    bool(torch.any(valid_timestamps[1:] <= valid_timestamps[:-1]))
                    or bool(torch.any(valid_positions[1:] != valid_positions[:-1] + 1))
                ):
                    raise ValueError(
                        "temporal cache timestamps and positions must increase strictly"
                    )
                expected_seen = int(valid_positions[-1].item()) + 1
                if int(self.total_seen[row].item()) != expected_seen:
                    raise ValueError(
                        "temporal total_seen must equal last absolute position plus one"
                    )
                if bool(torch.any(valid_positions < 0)):
                    raise ValueError("valid temporal cache position_ids must be non-negative")
            elif int(self.total_seen[row].item()) != 0:
                raise ValueError("an empty temporal cache must have total_seen=0")
            expected_replay_count = (
                min(3, int(valid_positions[0].item())) if valid_count == 64 else 0
            )
            if replay_count != expected_replay_count:
                raise ValueError(
                    "temporal replay context must preserve the available three-position margin"
                )
            if replay_count:
                if valid_count != 64:
                    raise ValueError("replay context is legal only beside a full 64-token cache")
                if (
                    not bool(torch.isfinite(replay_timestamps).all())
                    or bool(torch.any(replay_timestamps < 0.0))
                    or bool(torch.any(replay_positions < 0))
                ):
                    raise ValueError("valid replay metadata must be finite and non-negative")
                if replay_count > 1 and (
                    bool(torch.any(replay_timestamps[1:] <= replay_timestamps[:-1]))
                    or bool(torch.any(replay_positions[1:] != replay_positions[:-1] + 1))
                ):
                    raise ValueError("temporal replay metadata must increase strictly")
                if int(replay_positions[-1].item()) + 1 != int(
                    valid_positions[0].item()
                ) or not bool(replay_timestamps[-1] < valid_timestamps[0]):
                    raise ValueError("temporal replay context must immediately precede main cache")
            invalid = slice(valid_count, cache_length)
            if cache_length > valid_count:
                if (
                    bool(torch.any(self.hidden[row, invalid] != 0.0))
                    or bool(torch.any(self.timestamps[row, invalid] != -1.0))
                    or bool(torch.any(self.position_ids[row, invalid] != -1))
                ):
                    raise ValueError("temporal cache padding must use zero hidden and -1 metadata")
                for keys, values in zip(self.layer_keys, self.layer_values, strict=True):
                    if bool(torch.any(keys[row, :, invalid] != 0.0)) or bool(
                        torch.any(values[row, :, invalid] != 0.0)
                    ):
                        raise ValueError("temporal cache padding K/V must be zero")
            replay_invalid = slice(replay_count, replay_length)
            if replay_length > replay_count:
                if bool(torch.any(self.replay_timestamps[row, replay_invalid] != -1.0)) or bool(
                    torch.any(self.replay_position_ids[row, replay_invalid] != -1)
                ):
                    raise ValueError("temporal replay padding metadata must use -1")
                for keys, values in zip(
                    self.replay_layer_keys, self.replay_layer_values, strict=True
                ):
                    if bool(torch.any(keys[row, :, replay_invalid] != 0.0)) or bool(
                        torch.any(values[row, :, replay_invalid] != 0.0)
                    ):
                        raise ValueError("temporal replay padding K/V must be zero")

    @property
    def batch_size(self) -> int:
        return int(self.hidden.shape[0])

    @property
    def cache_length(self) -> int:
        return int(self.hidden.shape[1])

    @property
    def replay_length(self) -> int:
        return int(self.replay_position_ids.shape[1])

    def _storage_tensors(self) -> tuple[Tensor, ...]:
        return (
            self.hidden,
            self.timestamps,
            self.replay_timestamps,
            self.position_ids,
            self.replay_position_ids,
            self.valid_mask,
            self.replay_valid_mask,
            self.query_signatures,
            self.total_seen,
            *self.layer_keys,
            *self.layer_values,
            *self.replay_layer_keys,
            *self.replay_layer_values,
        )

    def split(self) -> tuple[TemporalCache, ...]:
        """Return storage-isolated singleton cache states for each batch row."""

        states: list[TemporalCache] = []
        for row in range(self.batch_size):
            count = int(self.valid_mask[row].sum().item())
            replay_count = int(self.replay_valid_mask[row].sum().item())
            states.append(
                TemporalCache(
                    hidden=self.hidden[row : row + 1, :count].clone(),
                    layer_keys=tuple(
                        keys[row : row + 1, :, :count].clone() for keys in self.layer_keys
                    ),
                    layer_values=tuple(
                        values[row : row + 1, :, :count].clone() for values in self.layer_values
                    ),
                    replay_layer_keys=tuple(
                        keys[row : row + 1, :, :replay_count].clone()
                        for keys in self.replay_layer_keys
                    ),
                    replay_layer_values=tuple(
                        values[row : row + 1, :, :replay_count].clone()
                        for values in self.replay_layer_values
                    ),
                    timestamps=self.timestamps[row : row + 1, :count].clone(),
                    replay_timestamps=self.replay_timestamps[row : row + 1, :replay_count].clone(),
                    position_ids=self.position_ids[row : row + 1, :count].clone(),
                    replay_position_ids=self.replay_position_ids[
                        row : row + 1, :replay_count
                    ].clone(),
                    valid_mask=self.valid_mask[row : row + 1, :count].clone(),
                    replay_valid_mask=self.replay_valid_mask[row : row + 1, :replay_count].clone(),
                    video_ids=(self.video_ids[row],),
                    trajectory_ids=(self.trajectory_ids[row],),
                    query_signatures=self.query_signatures[row : row + 1].clone(),
                    total_seen=self.total_seen[row : row + 1].clone(),
                    differentiable=self.differentiable,
                )
            )
        return tuple(states)

    @classmethod
    def pack(cls, states: Sequence[TemporalCache]) -> TemporalCache:
        """Pack storage-isolated singleton states with valid-prefix padding."""

        normalized = tuple(states)
        if not normalized or any(not isinstance(state, cls) for state in normalized):
            raise TypeError("TemporalCache.pack requires at least one TemporalCache")
        if any(state.batch_size != 1 for state in normalized):
            raise ValueError("TemporalCache.pack accepts singleton states only")
        for left_index, left in enumerate(normalized):
            for right in normalized[left_index + 1 :]:
                for left_tensor in left._storage_tensors():
                    if left_tensor.numel() == 0:
                        continue
                    for right_tensor in right._storage_tensors():
                        if right_tensor.numel() and _shares_storage(left_tensor, right_tensor):
                            raise ValueError(
                                "packed temporal states must not share mutable storage"
                            )
        reference = normalized[0]
        if any(
            state.hidden.dtype != reference.hidden.dtype
            or state.hidden.device != reference.hidden.device
            or state.timestamps.dtype != reference.timestamps.dtype
            or state.differentiable != reference.differentiable
            for state in normalized[1:]
        ):
            raise ValueError("packed temporal states must share dtype/device/differentiability")
        video_ids = tuple(state.video_ids[0] for state in normalized)
        trajectory_ids = tuple(state.trajectory_ids[0] for state in normalized)
        if len(set(video_ids)) != len(video_ids):
            raise ValueError("packed temporal states cannot duplicate video_ids")
        max_length = max(state.cache_length for state in normalized)
        max_replay_length = max(state.replay_length for state in normalized)

        def pad_hidden(tensor: Tensor, value: float = 0.0) -> Tensor:
            return F.pad(tensor, (0, 0, 0, max_length - tensor.shape[1]), value=value)

        def pad_vector(tensor: Tensor, value: float | int) -> Tensor:
            return F.pad(tensor, (0, max_length - tensor.shape[1]), value=value)

        hidden = torch.cat([pad_hidden(state.hidden) for state in normalized], dim=0)
        layer_keys = tuple(
            torch.cat(
                [
                    F.pad(
                        state.layer_keys[layer],
                        (0, 0, 0, max_length - state.cache_length),
                    )
                    for state in normalized
                ],
                dim=0,
            )
            for layer in range(6)
        )
        layer_values = tuple(
            torch.cat(
                [
                    F.pad(
                        state.layer_values[layer],
                        (0, 0, 0, max_length - state.cache_length),
                    )
                    for state in normalized
                ],
                dim=0,
            )
            for layer in range(6)
        )
        replay_layer_keys = tuple(
            torch.cat(
                [
                    F.pad(
                        state.replay_layer_keys[layer],
                        (0, 0, 0, max_replay_length - state.replay_length),
                    )
                    for state in normalized
                ],
                dim=0,
            )
            for layer in range(6)
        )
        replay_layer_values = tuple(
            torch.cat(
                [
                    F.pad(
                        state.replay_layer_values[layer],
                        (0, 0, 0, max_replay_length - state.replay_length),
                    )
                    for state in normalized
                ],
                dim=0,
            )
            for layer in range(6)
        )
        timestamps = torch.cat([pad_vector(state.timestamps, -1.0) for state in normalized], dim=0)
        replay_timestamps = torch.cat(
            [
                F.pad(
                    state.replay_timestamps,
                    (0, max_replay_length - state.replay_length),
                    value=-1.0,
                )
                for state in normalized
            ],
            dim=0,
        )
        position_ids = torch.cat(
            [pad_vector(state.position_ids, -1) for state in normalized], dim=0
        )
        replay_position_ids = torch.cat(
            [
                F.pad(
                    state.replay_position_ids,
                    (0, max_replay_length - state.replay_length),
                    value=-1,
                )
                for state in normalized
            ],
            dim=0,
        )
        valid_mask = torch.cat([pad_vector(state.valid_mask, False) for state in normalized], dim=0)
        replay_valid_mask = torch.cat(
            [
                F.pad(
                    state.replay_valid_mask,
                    (0, max_replay_length - state.replay_length),
                    value=False,
                )
                for state in normalized
            ],
            dim=0,
        )
        return cls(
            hidden=hidden,
            layer_keys=layer_keys,
            layer_values=layer_values,
            replay_layer_keys=replay_layer_keys,
            replay_layer_values=replay_layer_values,
            timestamps=timestamps,
            replay_timestamps=replay_timestamps,
            position_ids=position_ids,
            replay_position_ids=replay_position_ids,
            valid_mask=valid_mask,
            replay_valid_mask=replay_valid_mask,
            video_ids=video_ids,
            trajectory_ids=trajectory_ids,
            query_signatures=torch.cat([state.query_signatures for state in normalized], dim=0),
            total_seen=torch.cat([state.total_seen for state in normalized], dim=0),
            differentiable=reference.differentiable,
        )


@dataclass(frozen=True, slots=True)
class TemporalEncoderAudit:
    grid_shapes: tuple[tuple[int, int, int], ...]
    valid_tubelet_counts: tuple[int, ...]
    overlap_replay_counts: tuple[int, ...]
    evicted_counts: tuple[int, ...]
    cache_lengths: tuple[int, ...]
    causal_window: int = 64

    def __post_init__(self) -> None:
        batch_size = len(self.grid_shapes)
        fields = (
            self.valid_tubelet_counts,
            self.overlap_replay_counts,
            self.evicted_counts,
            self.cache_lengths,
        )
        if batch_size <= 0 or any(len(field) != batch_size for field in fields):
            raise ValueError("temporal audit fields must align to one non-empty batch")
        if any(type(value) is not int or value < 0 for field in fields for value in field):
            raise ValueError("temporal audit counters must be non-negative integers")
        if self.causal_window != 64:
            raise ValueError("P7 temporal audit requires a 64-token causal window")


@dataclass(frozen=True, slots=True)
class TemporalEncoderOutput:
    hidden: Tensor
    timestamps: Tensor
    position_ids: Tensor
    valid_mask: Tensor
    cache: TemporalCache
    audit: TemporalEncoderAudit | None = None

    def __post_init__(self) -> None:
        if (
            self.hidden.ndim != 3
            or self.hidden.shape[-1] != 768
            or not torch.is_floating_point(self.hidden)
        ):
            raise ValueError("temporal hidden must be [B, T, 768]")
        shape = self.hidden.shape[:2]
        if (
            self.timestamps.shape != shape
            or self.timestamps.dtype not in (torch.float32, torch.float64)
            or self.timestamps.device != self.hidden.device
        ):
            raise ValueError("temporal timestamps must be floating [B, T]")
        if (
            self.position_ids.shape != shape
            or self.position_ids.dtype != torch.int64
            or self.position_ids.device != self.hidden.device
        ):
            raise ValueError("temporal position_ids must be int64 [B, T]")
        if (
            self.valid_mask.shape != shape
            or self.valid_mask.dtype != torch.bool
            or self.valid_mask.device != self.hidden.device
        ):
            raise ValueError("temporal valid_mask must be bool [B, T]")
        if self.cache.batch_size != self.hidden.shape[0]:
            raise ValueError("temporal output and cache batch sizes must match")
        if (
            self.cache.hidden.dtype != self.hidden.dtype
            or self.cache.hidden.device != self.hidden.device
        ):
            raise ValueError("temporal output and cache must share dtype/device")
        if self.hidden.device.type != "meta":
            if not bool(torch.isfinite(self.hidden).all()):
                raise ValueError("temporal hidden must be finite")
            if shape[1] > 1 and bool(torch.any(self.valid_mask[:, 1:] & ~self.valid_mask[:, :-1])):
                raise ValueError("temporal output valid_mask must be a valid prefix")
            if bool(torch.any(self.hidden[~self.valid_mask] != 0.0)):
                raise ValueError("invalid temporal outputs must be zero")
            if bool(torch.any(self.timestamps[~self.valid_mask] != -1.0)) or bool(
                torch.any(self.position_ids[~self.valid_mask] != -1)
            ):
                raise ValueError("invalid temporal metadata must use -1 sentinels")
            for row in range(shape[0]):
                count = int(self.valid_mask[row].sum().item())
                timestamps = self.timestamps[row, :count]
                positions = self.position_ids[row, :count]
                if count and (
                    not bool(torch.isfinite(timestamps).all())
                    or bool(torch.any(timestamps < 0.0))
                    or bool(torch.any(positions < 0))
                ):
                    raise ValueError(
                        "valid temporal output metadata must be finite and non-negative"
                    )
                if count > 1 and (
                    bool(torch.any(timestamps[1:] <= timestamps[:-1]))
                    or bool(torch.any(positions[1:] != positions[:-1] + 1))
                ):
                    raise ValueError("temporal output metadata must increase strictly")
            if _shares_storage(self.hidden, self.cache.hidden):
                raise ValueError("temporal output and next cache must not share mutable storage")


class QueryConditionedSpatialPool(nn.Module):  # type: ignore[misc]
    """Pool every tubelet's merger grid with one query-conditioned attention token."""

    def __init__(self, config: TemporalEncoderConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.input_norm = nn.LayerNorm(config.input_dim, eps=config.layer_norm_eps)
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.query_projection = nn.Linear(config.query_dim, config.hidden_dim, bias=True)
        self.q_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.k_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.v_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)

    def forward(self, restored: RestoredMergedGrid, q_target: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, time_count, height, width, _ = restored.tokens.shape
        spatial_count = height * width
        spatial_mask = restored.spatial_valid_mask.flatten(2, 3)
        safe_tokens = torch.where(
            restored.spatial_valid_mask.unsqueeze(-1),
            restored.tokens,
            0.0,
        )
        projected = self.input_projection(self.input_norm(safe_tokens))
        projected = torch.where(restored.spatial_valid_mask.unsqueeze(-1), projected, 0.0)
        flattened = projected.flatten(2, 3)
        query_condition = self.query_projection(q_target)
        query = self.q_projection(query_condition).reshape(
            batch_size, self.num_heads, 1, self.head_dim
        )
        query = query[:, None].expand(-1, time_count, -1, -1, -1)
        keys = self.k_projection(flattened).reshape(
            batch_size,
            time_count,
            spatial_count,
            self.num_heads,
            self.head_dim,
        )
        values = self.v_projection(flattened).reshape_as(keys)
        keys = keys.permute(0, 1, 3, 2, 4)
        values = values.permute(0, 1, 3, 2, 4)
        logits = torch.matmul(query.float(), keys.float().transpose(-1, -2))
        logits = logits / math.sqrt(self.head_dim)
        expanded_mask = spatial_mask[:, :, None, None, :]
        logits = logits.masked_fill(~expanded_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * expanded_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        context = torch.matmul(weights.to(dtype=values.dtype), values)
        context = context.squeeze(-2).reshape(batch_size, time_count, self.hidden_dim)
        pooled = self.output_projection(context)
        pooled = torch.where(restored.tubelet_valid_mask.unsqueeze(-1), pooled, 0.0)
        return pooled, weights.squeeze(-2)


class CachedCausalTransformerLayer(nn.Module):  # type: ignore[misc]
    """One Pre-LN causal layer that consumes and returns its own projected K/V."""

    def __init__(self, config: TemporalEncoderConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dropout = float(config.dropout)
        self.norm_1 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.q_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.k_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.v_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.norm_2 = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.ffn_in = nn.Linear(config.hidden_dim, config.ffn_dim, bias=True)
        self.ffn_out = nn.Linear(config.ffn_dim, config.hidden_dim, bias=True)

    def forward(
        self,
        current: Tensor,
        prior_keys: Tensor,
        prior_values: Tensor,
        prior_position_ids: Tensor,
        current_position_ids: Tensor,
        *,
        causal_window: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        current_length = current.shape[1]
        normalized = self.norm_1(current)
        queries = self._split_heads(self.q_projection(normalized), current_length)
        current_keys = self._split_heads(self.k_projection(normalized), current_length)
        current_values = self._split_heads(self.v_projection(normalized), current_length)
        all_keys = torch.cat((prior_keys, current_keys), dim=2)
        all_values = torch.cat((prior_values, current_values), dim=2)
        all_positions = torch.cat((prior_position_ids, current_position_ids), dim=0)
        allowed = (all_positions.unsqueeze(0) <= current_position_ids.unsqueeze(1)) & (
            all_positions.unsqueeze(0) >= current_position_ids.unsqueeze(1) - (causal_window - 1)
        )
        logits = torch.matmul(queries.float(), all_keys.float().transpose(-1, -2))
        logits = logits / math.sqrt(self.head_dim)
        logits = logits.masked_fill(~allowed[None, None], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1).to(dtype=current.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        attention = torch.matmul(weights, all_values)
        attention = self.output_projection(self._merge_heads(attention))
        current = current + F.dropout(attention, p=self.dropout, training=self.training)
        feed_forward = self.ffn_in(self.norm_2(current))
        feed_forward = F.gelu(feed_forward)
        feed_forward = F.dropout(feed_forward, p=self.dropout, training=self.training)
        feed_forward = self.ffn_out(feed_forward)
        current = current + F.dropout(feed_forward, p=self.dropout, training=self.training)
        return current, current_keys, current_values

    def _split_heads(self, tensor: Tensor, length: int) -> Tensor:
        return tensor.reshape(1, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        return tensor.transpose(1, 2).reshape(1, tensor.shape[2], self.hidden_dim)


class RecurrentSlotAttentionStage(nn.Module):  # type: ignore[misc]
    """One independently-parameterized recurrent Slot Attention stage."""

    def __init__(self, config: SpatialEncoderConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.refinements = config.refinements_per_stage
        self.attention_epsilon = config.attention_epsilon
        self.token_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.slot_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.q_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.k_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.v_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.output_projection = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)
        self.gru = nn.GRUCell(config.hidden_dim, config.hidden_dim, bias=True)
        self.ffn_in = nn.Linear(config.hidden_dim, config.ffn_dim, bias=True)
        self.ffn_out = nn.Linear(config.ffn_dim, config.hidden_dim, bias=True)

    def forward(
        self,
        tokens: Tensor,
        slots: Tensor,
        query_condition: Tensor,
        token_valid_mask: Tensor,
        slot_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, token_count, _ = tokens.shape
        slot_count = slots.shape[1]
        normalized_tokens = self.token_norm(tokens)
        keys = self._split_heads(self.k_projection(normalized_tokens), token_count)
        values = self._split_heads(self.v_projection(normalized_tokens), token_count)
        current = slots
        confidence = torch.zeros(
            (batch_size, slot_count),
            dtype=slots.dtype,
            device=slots.device,
        )
        row_has_tokens = token_valid_mask.any(dim=1)
        effective_slot_mask = slot_valid_mask & row_has_tokens.unsqueeze(1)

        for _ in range(self.refinements):
            conditioned = self.slot_norm(current) + query_condition.unsqueeze(1)
            queries = self._split_heads(self.q_projection(conditioned), slot_count)
            logits = torch.einsum("bhkd,bhsd->bhks", queries, keys)
            logits = logits / math.sqrt(self.head_dim)
            logits = logits.masked_fill(
                ~slot_valid_mask[:, None, :, None],
                torch.finfo(logits.dtype).min,
            )
            normalization_logits = (
                logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
            )
            assignments = torch.softmax(normalization_logits, dim=2)
            valid_pairs = slot_valid_mask[:, None, :, None] & token_valid_mask[:, None, None, :]
            assignments = torch.where(valid_pairs, assignments, 0.0)
            valid_token_counts = token_valid_mask.sum(dim=1).clamp_min(1).to(assignments.dtype)
            confidence = assignments.sum(dim=-1) / valid_token_counts[:, None, None]
            confidence = confidence.mean(dim=1).to(slots.dtype)
            confidence = torch.where(effective_slot_mask, confidence, 0.0)
            denominator = assignments.sum(dim=-1, keepdim=True) + self.attention_epsilon
            weights = (assignments / denominator).to(values.dtype)
            updates = torch.einsum("bhks,bhsd->bhkd", weights, values)
            updates = self._merge_heads(updates)
            updates = self.output_projection(updates)
            updated = self.gru(
                updates.reshape(batch_size * slot_count, self.hidden_dim),
                current.reshape(batch_size * slot_count, self.hidden_dim),
            ).reshape(batch_size, slot_count, self.hidden_dim)
            updated = updated + self.ffn_out(F.silu(self.ffn_in(self.ffn_norm(updated))))
            current = torch.where(effective_slot_mask.unsqueeze(-1), updated, current)

        return current, confidence

    def _split_heads(self, values: Tensor, item_count: int) -> Tensor:
        return values.reshape(
            values.shape[0],
            item_count,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _merge_heads(self, values: Tensor) -> Tensor:
        return values.transpose(1, 2).reshape(values.shape[0], values.shape[2], self.hidden_dim)


class SpatialObjectEncoder(nn.Module):  # type: ignore[misc]
    """Two-stage query-conditioned recurrent Slot Attention over merger grids."""

    slot_codes: Tensor

    def __init__(self, config: SpatialEncoderConfig) -> None:
        super().__init__()
        _validate_spatial_config(config)
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim, eps=config.layer_norm_eps)
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim, bias=True)
        self.query_projection = nn.Linear(config.query_dim, config.hidden_dim, bias=True)
        self.shared_slot_seed = nn.Parameter(torch.zeros(config.hidden_dim))
        self.register_buffer(
            "slot_codes",
            _sinusoidal_slot_codes(config.max_active_slots, config.hidden_dim),
            persistent=False,
        )
        self.stage_1 = RecurrentSlotAttentionStage(config)
        self.stage_2 = RecurrentSlotAttentionStage(config)

    def forward(
        self,
        adapted_embeddings: Tensor,
        visual_valid_mask: Tensor,
        metadata: MergedVideoMetadata,
        tubelet_valid_mask: Tensor,
        q_target: Tensor,
        video_ids: Sequence[str],
        *,
        prior_states: Sequence[SpatialSlotRuntimeState | None] | None = None,
        query_valid_mask: Tensor | None = None,
        required_slot_counts: Tensor | None = None,
        detach_runtime_state: bool = True,
    ) -> SpatialEncoderOutput:
        """Process tubelets sequentially and return functional per-video next states."""

        if type(detach_runtime_state) is not bool:
            raise TypeError("detach_runtime_state must be a bool")
        self._validate_module_inputs(adapted_embeddings, q_target)
        batch_size = adapted_embeddings.shape[0]
        normalized_video_ids = _normalize_video_ids(video_ids, batch_size)
        restored = restore_merged_grid(
            adapted_embeddings,
            visual_valid_mask,
            metadata,
            tubelet_valid_mask,
        )
        query_mask = _normalize_query_valid_mask(q_target, query_valid_mask)
        valid_query = torch.where(query_mask.unsqueeze(1), q_target, 0.0)
        if q_target.device.type != "meta" and not bool(torch.isfinite(valid_query).all()):
            raise ValueError("valid q_target rows must be finite")
        query_condition = self.query_projection(valid_query)
        safe_tokens = torch.where(
            restored.spatial_valid_mask.unsqueeze(-1),
            restored.tokens,
            0.0,
        )
        if safe_tokens.device.type != "meta" and not bool(torch.isfinite(safe_tokens).all()):
            raise ValueError("valid adapted merger tokens must be finite")
        states = _normalize_prior_states(prior_states, batch_size)
        self._validate_prior_states(states, normalized_video_ids, adapted_embeddings)
        _assert_optional_runtime_state_storage_isolated(states)

        fresh_slots = self._initial_slots(query_condition)
        current_rows: list[Tensor] = []
        mask_rows: list[Tensor] = []
        confidence_rows: list[Tensor] = []
        prior_overflow: list[int] = []
        prior_events: list[int] = []
        prior_processed: list[int] = []
        for row, state in enumerate(states):
            if state is None:
                if not bool(restored.tubelet_valid_mask[row].any()):
                    raise ValueError("a fresh spatial runtime requires at least one valid tubelet")
                current_rows.append(fresh_slots[row])
                mask_rows.append(
                    torch.ones(
                        self.config.active_slots,
                        dtype=torch.bool,
                        device=adapted_embeddings.device,
                    )
                )
                confidence_rows.append(
                    torch.zeros(
                        self.config.active_slots,
                        dtype=adapted_embeddings.dtype,
                        device=adapted_embeddings.device,
                    )
                )
                prior_overflow.append(0)
                prior_events.append(0)
                prior_processed.append(0)
            else:
                current_rows.append(state.slots)
                mask_rows.append(state.slot_valid_mask)
                confidence_rows.append(state.slot_confidence)
                prior_overflow.append(state.active_slot_overflow_count)
                prior_events.append(state.overflow_event_count)
                prior_processed.append(state.processed_tubelets)

        current = torch.stack(current_rows)
        current_mask = torch.stack(mask_rows)
        current_confidence = torch.stack(confidence_rows)
        for tubelet_index in range(safe_tokens.shape[1]):
            raw_tubelet = safe_tokens[:, tubelet_index].flatten(1, 2)
            token_mask = restored.spatial_valid_mask[:, tubelet_index].flatten(1, 2)
            tubelet_tokens = self.input_projection(self.input_norm(raw_tubelet))
            tubelet_tokens = torch.where(
                token_mask.unsqueeze(-1),
                tubelet_tokens,
                0.0,
            )
            first, _ = self.stage_1(
                tubelet_tokens,
                current,
                query_condition,
                token_mask,
                current_mask,
            )
            second, confidence = self.stage_2(
                tubelet_tokens,
                first,
                query_condition,
                token_mask,
                current_mask,
            )
            row_has_tokens = token_mask.any(dim=1)
            current = torch.where(row_has_tokens[:, None, None], second, current)
            current_confidence = torch.where(
                row_has_tokens[:, None],
                confidence,
                current_confidence,
            )

        if current.device.type != "meta" and not bool(torch.isfinite(current).all()):
            raise ValueError("spatial encoder output slots must be finite")
        required = _normalize_required_slot_counts(
            required_slot_counts,
            batch_size,
            adapted_embeddings.device,
            self.config.active_slots,
        )
        excess = torch.clamp(required - self.config.active_slots, min=0)
        valid_tubelets = restored.tubelet_valid_mask.sum(dim=1, dtype=torch.int64)
        next_states = tuple(
            self._make_next_state(
                video_id=normalized_video_ids[row],
                slots=current[row],
                slot_valid_mask=current_mask[row],
                slot_confidence=current_confidence[row],
                overflow_count=prior_overflow[row] + int(excess[row].item()),
                overflow_event_count=prior_events[row] + int(excess[row].item() > 0),
                processed_tubelets=prior_processed[row] + int(valid_tubelets[row].item()),
                detach=detach_runtime_state,
            )
            for row in range(batch_size)
        )
        overflow_counts = torch.tensor(
            [state.active_slot_overflow_count for state in next_states],
            dtype=torch.int64,
            device=adapted_embeddings.device,
        )
        audit = SpatialEncoderAudit(
            grid_shapes=restored.grid_shapes,
            visual_token_counts=metadata.token_counts,
            valid_tubelet_counts=tuple(int(value) for value in valid_tubelets.tolist()),
            required_slot_counts=tuple(int(value) for value in required.tolist()),
            excess_slot_counts=tuple(int(value) for value in excess.tolist()),
            overflow_events=tuple(bool(value) for value in (excess > 0).tolist()),
            overflow_policy=self.config.overflow_policy,
            stage_refinement_calls=(
                self.config.refinements_per_stage * safe_tokens.shape[1],
                self.config.refinements_per_stage * safe_tokens.shape[1],
            ),
        )
        return SpatialEncoderOutput(
            slots=current,
            slot_valid_mask=current_mask,
            active_slot_overflow_count=overflow_counts,
            slot_confidence=current_confidence,
            next_states=next_states,
            audit=audit,
        )

    def reset_slot_state(
        self,
        video_id: str,
        q_target: Tensor,
        *,
        query_valid: bool = True,
        slot_valid_mask: Tensor | None = None,
        differentiable: bool = False,
    ) -> SpatialSlotRuntimeState:
        """Create a reproducible first-tubelet state from shared seed, query, and fixed codes."""

        if not video_id:
            raise ValueError("reset requires a non-empty video_id")
        if type(query_valid) is not bool or type(differentiable) is not bool:
            raise TypeError("reset query_valid and differentiable flags must be bool")
        if q_target.shape != (self.config.query_dim,) or not torch.is_floating_point(q_target):
            raise ValueError(f"reset q_target must be floating [{self.config.query_dim}]")
        if (
            q_target.dtype != self.shared_slot_seed.dtype
            or q_target.device != self.shared_slot_seed.device
        ):
            raise ValueError("reset q_target must share module dtype/device")
        safe_query = q_target if query_valid else torch.zeros_like(q_target)
        if safe_query.device.type != "meta" and not bool(torch.isfinite(safe_query).all()):
            raise ValueError("valid reset q_target must be finite")
        condition = self.query_projection(safe_query.unsqueeze(0))
        slots = self._initial_slots(condition)[0]
        if slot_valid_mask is None:
            valid_mask = torch.ones(
                self.config.active_slots,
                dtype=torch.bool,
                device=slots.device,
            )
        else:
            if (
                slot_valid_mask.shape != (self.config.active_slots,)
                or slot_valid_mask.dtype != torch.bool
                or slot_valid_mask.device != slots.device
            ):
                raise ValueError("reset slot_valid_mask must be bool [K_a] on the module device")
            if not bool(slot_valid_mask.any()):
                raise ValueError("reset requires at least one valid slot")
            valid_mask = slot_valid_mask
        confidence = torch.zeros(
            self.config.active_slots,
            dtype=slots.dtype,
            device=slots.device,
        )
        if differentiable:
            state_slots = slots.clone()
            state_confidence = confidence.clone()
        else:
            state_slots = slots.detach().clone()
            state_confidence = confidence.detach().clone()
        return SpatialSlotRuntimeState(
            video_id=video_id,
            slots=state_slots,
            slot_valid_mask=valid_mask.clone(),
            slot_confidence=state_confidence,
            active_slot_overflow_count=0,
            overflow_event_count=0,
            processed_tubelets=0,
            differentiable=differentiable,
        )

    def _initial_slots(self, query_condition: Tensor) -> Tensor:
        codes = self.slot_codes[: self.config.active_slots].to(
            dtype=query_condition.dtype,
            device=query_condition.device,
        )
        return (
            self.shared_slot_seed[None, None, :] + query_condition[:, None, :] + codes[None, :, :]
        )

    def _validate_module_inputs(self, adapted_embeddings: Tensor, q_target: Tensor) -> None:
        if (
            adapted_embeddings.ndim != 3
            or adapted_embeddings.shape[0] <= 0
            or adapted_embeddings.shape[1] <= 0
            or adapted_embeddings.shape[2] != self.config.input_dim
            or not torch.is_floating_point(adapted_embeddings)
        ):
            raise ValueError(
                f"adapted_embeddings must be floating non-empty [B, N, {self.config.input_dim}]"
            )
        if q_target.shape != (
            adapted_embeddings.shape[0],
            self.config.query_dim,
        ) or not torch.is_floating_point(q_target):
            raise ValueError(f"q_target must be floating [B, {self.config.query_dim}]")
        if (
            q_target.dtype != adapted_embeddings.dtype
            or q_target.device != adapted_embeddings.device
        ):
            raise ValueError("q_target and adapted embeddings must share dtype/device")
        for parameter in self.parameters():
            if (
                parameter.dtype != adapted_embeddings.dtype
                or parameter.device != adapted_embeddings.device
            ):
                raise ValueError("spatial encoder module and inputs must share dtype/device")

    def _validate_prior_states(
        self,
        states: tuple[SpatialSlotRuntimeState | None, ...],
        video_ids: tuple[str, ...],
        inputs: Tensor,
    ) -> None:
        for row, state in enumerate(states):
            if state is None:
                continue
            if state.video_id != video_ids[row]:
                raise ValueError("spatial runtime video_id must match its batch row")
            if state.slots.shape != (self.config.active_slots, self.config.hidden_dim):
                raise ValueError("stale spatial runtime has the wrong slot shape")
            if state.slots.dtype != inputs.dtype or state.slots.device != inputs.device:
                raise ValueError("spatial runtime and encoder inputs must share dtype/device")

    @staticmethod
    def _make_next_state(
        *,
        video_id: str,
        slots: Tensor,
        slot_valid_mask: Tensor,
        slot_confidence: Tensor,
        overflow_count: int,
        overflow_event_count: int,
        processed_tubelets: int,
        detach: bool,
    ) -> SpatialSlotRuntimeState:
        next_slots = slots.detach().clone() if detach else slots.clone()
        next_confidence = slot_confidence.detach().clone() if detach else slot_confidence.clone()
        return SpatialSlotRuntimeState(
            video_id=video_id,
            slots=next_slots,
            slot_valid_mask=slot_valid_mask.clone(),
            slot_confidence=next_confidence,
            active_slot_overflow_count=overflow_count,
            overflow_event_count=overflow_event_count,
            processed_tubelets=processed_tubelets,
            differentiable=not detach,
        )


@dataclass(frozen=True, slots=True)
class _PreparedTemporalHistory:
    layer_keys: tuple[Tensor, ...]
    layer_values: tuple[Tensor, ...]
    timestamps: Tensor
    position_ids: Tensor
    retained_hidden: Tensor
    retained_timestamps: Tensor
    retained_position_ids: Tensor
    overlap_count: int


class TemporalEventEncoder(nn.Module):  # type: ignore[misc]
    """Query-conditioned tubelet pooling followed by a six-layer causal Transformer."""

    def __init__(self, config: TemporalEncoderConfig) -> None:
        super().__init__()
        _validate_temporal_config(config)
        self.config = config
        self.spatial_pool = QueryConditionedSpatialPool(config)
        self.layers = nn.ModuleList(
            CachedCausalTransformerLayer(config) for _ in range(config.num_layers)
        )

    def forward(
        self,
        adapted_embeddings: Tensor,
        visual_valid_mask: Tensor,
        metadata: MergedVideoMetadata,
        tubelet_valid_mask: Tensor,
        tubelet_timestamps: Tensor,
        tubelet_position_ids: Tensor,
        query_time: Tensor,
        q_target: Tensor,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
        *,
        cache: TemporalCache | None = None,
        detach_cache: bool = True,
    ) -> TemporalEncoderOutput:
        """Encode one causal chunk and return a functional, overlap-safe K/V cache."""

        if type(detach_cache) is not bool:
            raise TypeError("detach_cache must be a bool")
        restored = restore_merged_grid(
            adapted_embeddings,
            visual_valid_mask,
            metadata,
            tubelet_valid_mask,
        )
        normalized_video_ids, normalized_trajectory_ids = self._validate_inputs(
            adapted_embeddings,
            restored,
            tubelet_timestamps,
            tubelet_position_ids,
            query_time,
            q_target,
            video_ids,
            trajectory_ids,
            cache,
        )
        pooled, _ = self.spatial_pool(restored, q_target)
        batch_size, max_time, _ = pooled.shape
        prior_rows = cache.split() if cache is not None else (None,) * batch_size
        hidden_rows: list[Tensor] = []
        next_rows: list[TemporalCache] = []
        overlap_counts: list[int] = []
        evicted_counts: list[int] = []
        cache_lengths: list[int] = []
        valid_counts: list[int] = []

        for row in range(batch_size):
            valid_count = int(restored.tubelet_valid_mask[row].sum().item())
            valid_counts.append(valid_count)
            prior = prior_rows[row]
            if prior is None:
                prior = _empty_temporal_cache(
                    normalized_video_ids[row],
                    normalized_trajectory_ids[row],
                    q_target[row],
                    num_layers=self.config.num_layers,
                    num_heads=self.config.num_heads,
                    head_dim=self.config.head_dim,
                    hidden_dim=self.config.hidden_dim,
                )
            if valid_count == 0:
                next_state = _clone_temporal_cache(prior, detach=detach_cache)
                hidden_rows.append(pooled.new_zeros((1, max_time, self.config.hidden_dim)))
                next_rows.append(next_state)
                overlap_counts.append(0)
                evicted_counts.append(0)
                cache_lengths.append(next_state.cache_length)
                continue

            current_positions = tubelet_position_ids[row, :valid_count]
            current_timestamps = tubelet_timestamps[row, :valid_count]
            history = self._prepare_prior_for_positions(
                prior,
                current_positions,
                current_timestamps,
            )
            current = pooled[row : row + 1, :valid_count]
            position_encoding = _temporal_sinusoidal_encoding(
                current_positions,
                self.config.hidden_dim,
                dtype=current.dtype,
            ).unsqueeze(0)
            current = current + position_encoding
            next_keys: list[Tensor] = []
            next_values: list[Tensor] = []
            next_replay_keys: list[Tensor] = []
            next_replay_values: list[Tensor] = []
            for layer_index, layer in enumerate(self.layers):
                current, current_keys, current_values = layer(
                    current,
                    history.layer_keys[layer_index],
                    history.layer_values[layer_index],
                    history.position_ids,
                    current_positions,
                    causal_window=self.config.cache_tubelets,
                )
                combined_keys = torch.cat((history.layer_keys[layer_index], current_keys), dim=2)
                combined_values = torch.cat(
                    (history.layer_values[layer_index], current_values), dim=2
                )
                main_start = max(0, combined_keys.shape[2] - self.config.cache_tubelets)
                replay_start = max(0, main_start - self.config.replay_context_tubelets)
                next_keys.append(combined_keys[:, :, main_start:])
                next_values.append(combined_values[:, :, main_start:])
                next_replay_keys.append(combined_keys[:, :, replay_start:main_start])
                next_replay_values.append(combined_values[:, :, replay_start:main_start])

            combined_hidden = torch.cat((history.retained_hidden, current), dim=1)
            combined_main_timestamps = torch.cat(
                (
                    history.retained_timestamps,
                    current_timestamps.to(dtype=torch.float64),
                ),
                dim=0,
            )
            combined_main_positions = torch.cat(
                (history.retained_position_ids, current_positions), dim=0
            )
            combined_context_timestamps = torch.cat(
                (history.timestamps, current_timestamps.to(dtype=torch.float64)), dim=0
            )
            combined_context_positions = torch.cat((history.position_ids, current_positions), dim=0)
            main_count = min(combined_context_positions.shape[0], self.config.cache_tubelets)
            context_main_start = combined_context_positions.shape[0] - main_count
            context_replay_start = max(0, context_main_start - self.config.replay_context_tubelets)
            evicted_count = max(0, combined_main_positions.shape[0] - self.config.cache_tubelets)
            cache_hidden = combined_hidden[:, -main_count:]
            cache_timestamps = combined_context_timestamps[-main_count:].unsqueeze(0)
            cache_positions = combined_context_positions[-main_count:].unsqueeze(0)
            replay_timestamps = combined_context_timestamps[
                context_replay_start:context_main_start
            ].unsqueeze(0)
            replay_positions = combined_context_positions[
                context_replay_start:context_main_start
            ].unsqueeze(0)
            if not torch.equal(
                combined_main_timestamps[-main_count:], cache_timestamps[0]
            ) or not torch.equal(combined_main_positions[-main_count:], cache_positions[0]):
                raise RuntimeError("temporal replay bookkeeping produced inconsistent main cache")
            cache_valid = torch.ones_like(cache_positions, dtype=torch.bool)
            replay_valid = torch.ones_like(replay_positions, dtype=torch.bool)
            next_state = TemporalCache(
                hidden=_cache_tensor(cache_hidden, detach_cache),
                layer_keys=tuple(_cache_tensor(value, detach_cache) for value in next_keys),
                layer_values=tuple(_cache_tensor(value, detach_cache) for value in next_values),
                replay_layer_keys=tuple(
                    _cache_tensor(value, detach_cache) for value in next_replay_keys
                ),
                replay_layer_values=tuple(
                    _cache_tensor(value, detach_cache) for value in next_replay_values
                ),
                timestamps=cache_timestamps.detach().clone(),
                replay_timestamps=replay_timestamps.detach().clone(),
                position_ids=cache_positions.clone(),
                replay_position_ids=replay_positions.clone(),
                valid_mask=cache_valid,
                replay_valid_mask=replay_valid,
                video_ids=(normalized_video_ids[row],),
                trajectory_ids=(normalized_trajectory_ids[row],),
                query_signatures=q_target[row : row + 1].detach().clone(),
                total_seen=torch.tensor(
                    [int(current_positions[-1].item()) + 1],
                    dtype=torch.int64,
                    device=current.device,
                ),
                differentiable=not detach_cache,
            )
            hidden_rows.append(F.pad(current, (0, 0, 0, max_time - valid_count), value=0.0))
            next_rows.append(next_state)
            overlap_counts.append(history.overlap_count)
            evicted_counts.append(evicted_count)
            cache_lengths.append(next_state.cache_length)

        next_cache = TemporalCache.pack(next_rows)
        output_hidden = torch.cat(hidden_rows, dim=0)
        output_timestamps = torch.where(
            restored.tubelet_valid_mask,
            tubelet_timestamps,
            torch.full_like(tubelet_timestamps, -1.0),
        )
        output_positions = torch.where(
            restored.tubelet_valid_mask,
            tubelet_position_ids,
            torch.full_like(tubelet_position_ids, -1),
        )
        audit = TemporalEncoderAudit(
            grid_shapes=restored.grid_shapes,
            valid_tubelet_counts=tuple(valid_counts),
            overlap_replay_counts=tuple(overlap_counts),
            evicted_counts=tuple(evicted_counts),
            cache_lengths=tuple(cache_lengths),
            causal_window=self.config.cache_tubelets,
        )
        return TemporalEncoderOutput(
            hidden=output_hidden,
            timestamps=output_timestamps,
            position_ids=output_positions,
            valid_mask=restored.tubelet_valid_mask,
            cache=next_cache,
            audit=audit,
        )

    def reset_cache(
        self,
        video_id: str,
        trajectory_id: str,
        q_target: Tensor,
    ) -> TemporalCache:
        """Create an empty singleton cache with explicit ownership and query signature."""

        if not video_id or not trajectory_id:
            raise ValueError("temporal cache reset requires non-empty owner identifiers")
        if q_target.shape != (self.config.query_dim,) or not torch.is_floating_point(q_target):
            raise ValueError(f"reset q_target must be floating [{self.config.query_dim}]")
        parameter = next(self.parameters())
        if q_target.dtype != parameter.dtype or q_target.device != parameter.device:
            raise ValueError("reset q_target must share module dtype/device")
        if q_target.device.type != "meta" and not bool(torch.isfinite(q_target).all()):
            raise ValueError("reset q_target must be finite")
        return _empty_temporal_cache(
            video_id,
            trajectory_id,
            q_target,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.hidden_dim,
        )

    @staticmethod
    def pack_cache(states: Sequence[TemporalCache]) -> TemporalCache:
        return TemporalCache.pack(states)

    @staticmethod
    def split_cache(cache: TemporalCache) -> tuple[TemporalCache, ...]:
        if not isinstance(cache, TemporalCache):
            raise TypeError("split_cache requires a TemporalCache")
        return cache.split()

    def _validate_inputs(
        self,
        adapted_embeddings: Tensor,
        restored: RestoredMergedGrid,
        tubelet_timestamps: Tensor,
        tubelet_position_ids: Tensor,
        query_time: Tensor,
        q_target: Tensor,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
        cache: TemporalCache | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        batch_size, max_time = restored.tubelet_valid_mask.shape
        if adapted_embeddings.shape[-1] != self.config.input_dim:
            raise ValueError("temporal adapted embeddings have the wrong input dimension")
        if q_target.shape != (batch_size, self.config.query_dim) or not torch.is_floating_point(
            q_target
        ):
            raise ValueError("temporal q_target must be floating [B, 512]")
        if (
            q_target.dtype != adapted_embeddings.dtype
            or q_target.device != adapted_embeddings.device
        ):
            raise ValueError("temporal q_target and embeddings must share dtype/device")
        if (
            tubelet_timestamps.shape != (batch_size, max_time)
            or tubelet_timestamps.dtype not in (torch.float32, torch.float64)
            or tubelet_timestamps.device != adapted_embeddings.device
        ):
            raise ValueError("tubelet_timestamps must match inputs as floating [B, T]")
        if (
            tubelet_position_ids.shape != (batch_size, max_time)
            or tubelet_position_ids.dtype != torch.int64
            or tubelet_position_ids.device != adapted_embeddings.device
        ):
            raise ValueError("tubelet_position_ids must be int64 [B, T] on the input device")
        if (
            query_time.shape != (batch_size,)
            or query_time.dtype not in (torch.float32, torch.float64)
            or query_time.device != adapted_embeddings.device
        ):
            raise ValueError("query_time must match inputs as floating [B]")
        if max_time > 1 and bool(
            torch.any(restored.tubelet_valid_mask[:, 1:] & ~restored.tubelet_valid_mask[:, :-1])
        ):
            raise ValueError("temporal tubelet_valid_mask must be a valid prefix")
        if adapted_embeddings.device.type != "meta":
            if not bool(torch.isfinite(q_target).all()):
                raise ValueError("temporal q_target must be finite")
            if not bool(torch.isfinite(query_time).all()) or bool(torch.any(query_time < 0.0)):
                raise ValueError("query_time must be finite and non-negative")
            if not bool(torch.isfinite(restored.tokens[restored.spatial_valid_mask]).all()):
                raise ValueError("valid adapted merger tokens must be finite")
            invalid_mask = ~restored.tubelet_valid_mask
            if bool(torch.any(tubelet_timestamps[invalid_mask] != -1.0)) or bool(
                torch.any(tubelet_position_ids[invalid_mask] != -1)
            ):
                raise ValueError("invalid tubelets must use -1 timestamp and position sentinels")
            for row in range(batch_size):
                count = int(restored.tubelet_valid_mask[row].sum().item())
                if not count:
                    continue
                timestamps = tubelet_timestamps[row, :count]
                positions = tubelet_position_ids[row, :count]
                if (
                    not bool(torch.isfinite(timestamps).all())
                    or bool(torch.any(timestamps < 0.0))
                    or bool(torch.any(timestamps > query_time[row]))
                ):
                    raise ValueError("valid tubelet timestamps must be legal at query_time")
                if count > 1 and (
                    bool(torch.any(timestamps[1:] <= timestamps[:-1]))
                    or bool(torch.any(positions[1:] != positions[:-1] + 1))
                ):
                    raise ValueError(
                        "valid tubelet timestamps and positions must increase strictly"
                    )
                if bool(torch.any(positions < 0)):
                    raise ValueError("valid tubelet position IDs must be non-negative")
        for parameter in self.parameters():
            if (
                parameter.dtype != adapted_embeddings.dtype
                or parameter.device != adapted_embeddings.device
            ):
                raise ValueError("temporal encoder module and inputs must share dtype/device")
        normalized_video_ids = tuple(video_ids)
        normalized_trajectory_ids = tuple(trajectory_ids)
        if (
            len(normalized_video_ids) != batch_size
            or any(not value for value in normalized_video_ids)
            or len(set(normalized_video_ids)) != batch_size
        ):
            raise ValueError("temporal encoder requires unique non-empty video_ids")
        if len(normalized_trajectory_ids) != batch_size or any(
            not value for value in normalized_trajectory_ids
        ):
            raise ValueError("temporal encoder requires one non-empty trajectory_id per row")
        if cache is not None:
            if not isinstance(cache, TemporalCache) or cache.batch_size != batch_size:
                raise ValueError("temporal cache batch size must match current inputs")
            if (
                cache.hidden.dtype != adapted_embeddings.dtype
                or cache.hidden.device != adapted_embeddings.device
            ):
                raise ValueError("temporal cache and inputs must share dtype/device")
            if cache.video_ids != normalized_video_ids:
                raise ValueError("temporal cache video owners must match exact batch order")
            if cache.trajectory_ids != normalized_trajectory_ids:
                raise ValueError("temporal cache trajectory owners must match exact batch order")
            if not torch.equal(cache.query_signatures, q_target.detach()):
                raise ValueError("temporal cache query signature drift requires reset")
            if cache.hidden.device.type != "meta":
                for row in range(batch_size):
                    count = int(cache.valid_mask[row].sum().item())
                    if count and bool(torch.any(cache.timestamps[row, :count] > query_time[row])):
                        raise ValueError("temporal cache contains content after query_time")
        return normalized_video_ids, normalized_trajectory_ids

    def _prepare_prior_for_positions(
        self,
        prior: TemporalCache,
        current_positions: Tensor,
        current_timestamps: Tensor,
    ) -> _PreparedTemporalHistory:
        if prior.batch_size != 1:
            raise ValueError("row processing requires a singleton temporal cache")
        if prior.cache_length == 0:
            if int(current_positions[0].item()) != 0:
                raise ValueError("a fresh temporal trajectory must start at position zero")
            return _PreparedTemporalHistory(
                layer_keys=prior.layer_keys,
                layer_values=prior.layer_values,
                timestamps=prior.timestamps[0],
                position_ids=prior.position_ids[0],
                retained_hidden=prior.hidden,
                retained_timestamps=prior.timestamps[0],
                retained_position_ids=prior.position_ids[0],
                overlap_count=0,
            )
        cached_positions = prior.position_ids[0]
        cached_timestamps = prior.timestamps[0]
        context_positions = torch.cat((prior.replay_position_ids[0], cached_positions), dim=0)
        context_timestamps = torch.cat((prior.replay_timestamps[0], cached_timestamps), dim=0)
        context_keys = tuple(
            torch.cat((replay, main), dim=2)
            for replay, main in zip(prior.replay_layer_keys, prior.layer_keys, strict=True)
        )
        context_values = tuple(
            torch.cat((replay, main), dim=2)
            for replay, main in zip(prior.replay_layer_values, prior.layer_values, strict=True)
        )
        first_position = int(current_positions[0].item())
        last_position = int(current_positions[-1].item())
        cached_first = int(cached_positions[0].item())
        cached_last = int(cached_positions[-1].item())
        context_first = int(context_positions[0].item())
        retain_context_count = context_positions.shape[0]
        retain_main_count = prior.cache_length
        overlap_count = 0
        if first_position <= cached_last:
            if first_position < cached_first:
                raise ValueError("overlap rewind reaches an already-evicted cache position")
            needed_first = max(0, first_position - (self.config.cache_tubelets - 1))
            if context_first > needed_first:
                raise ValueError("overlap replay lacks the complete 64-token causal context")
            if last_position < cached_last:
                raise ValueError("overlap replay cannot discard an unobserved cached suffix")
            retain_context_count = first_position - context_first
            retain_main_count = first_position - cached_first
            overlap_count = cached_last - first_position + 1
            overlap_timestamps = current_timestamps[:overlap_count].to(
                dtype=cached_timestamps.dtype
            )
            cached_overlap = cached_timestamps[retain_main_count:]
            if not _timestamps_match(overlap_timestamps, cached_overlap):
                raise ValueError("overlap position timestamps must match cached source tubelets")
        elif first_position != cached_last + 1:
            raise ValueError("temporal position IDs cannot contain gaps")
        if retain_context_count and not bool(
            current_timestamps[0].to(dtype=context_timestamps.dtype)
            > context_timestamps[retain_context_count - 1]
        ):
            raise ValueError("replayed temporal timestamps must remain strictly increasing")
        return _PreparedTemporalHistory(
            layer_keys=tuple(value[:, :, :retain_context_count] for value in context_keys),
            layer_values=tuple(value[:, :, :retain_context_count] for value in context_values),
            timestamps=context_timestamps[:retain_context_count],
            position_ids=context_positions[:retain_context_count],
            retained_hidden=prior.hidden[:, :retain_main_count],
            retained_timestamps=cached_timestamps[:retain_main_count],
            retained_position_ids=cached_positions[:retain_main_count],
            overlap_count=overlap_count,
        )


def restore_merged_grid(
    adapted_embeddings: Tensor,
    visual_valid_mask: Tensor,
    metadata: MergedVideoMetadata,
    tubelet_valid_mask: Tensor,
) -> RestoredMergedGrid:
    """Restore heterogeneous `[T,H_m,W_m]` grids without assuming 49 spatial tokens."""

    if not isinstance(metadata, MergedVideoMetadata):
        raise TypeError("metadata must be MergedVideoMetadata")
    if adapted_embeddings.ndim != 3 or not torch.is_floating_point(adapted_embeddings):
        raise ValueError("adapted_embeddings must be floating [B, N_max, D]")
    batch_size, width, hidden_dim = adapted_embeddings.shape
    if batch_size != len(metadata.token_counts) or width != max(metadata.token_counts):
        raise ValueError("adapted embedding padding must match metadata token counts")
    if visual_valid_mask.shape != (batch_size, width) or visual_valid_mask.dtype != torch.bool:
        raise ValueError("visual_valid_mask must be bool [B, N_max]")
    if visual_valid_mask.device != adapted_embeddings.device:
        raise ValueError("visual_valid_mask and adapted embeddings must share a device")
    grid_shapes = tuple(
        (int(row[0]), int(row[1]), int(row[2]))
        for row in metadata.merged_grid_thw.detach().cpu().tolist()
    )
    max_t = max(shape[0] for shape in grid_shapes)
    max_h = max(shape[1] for shape in grid_shapes)
    max_w = max(shape[2] for shape in grid_shapes)
    if (
        tubelet_valid_mask.shape != (batch_size, max_t)
        or tubelet_valid_mask.dtype != torch.bool
        or tubelet_valid_mask.device != adapted_embeddings.device
    ):
        raise ValueError("tubelet_valid_mask must be bool [B, T_max] on the input device")
    token_positions = torch.arange(width, device=adapted_embeddings.device).unsqueeze(0)
    expected_visual_mask = token_positions < torch.tensor(
        metadata.token_counts,
        device=adapted_embeddings.device,
    ).unsqueeze(1)
    if not torch.equal(visual_valid_mask, expected_visual_mask):
        raise ValueError("visual_valid_mask must be a metadata-aligned valid prefix")
    time_positions = torch.arange(max_t, device=adapted_embeddings.device).unsqueeze(0)
    geometric_tubelet_mask = time_positions < torch.tensor(
        [shape[0] for shape in grid_shapes],
        device=adapted_embeddings.device,
    ).unsqueeze(1)
    if bool(torch.any(tubelet_valid_mask & ~geometric_tubelet_mask)):
        raise ValueError("tubelet_valid_mask cannot enable padded time positions")
    tokens = adapted_embeddings.new_zeros((batch_size, max_t, max_h, max_w, hidden_dim))
    geometry_mask = torch.zeros(
        (batch_size, max_t, max_h, max_w),
        dtype=torch.bool,
        device=adapted_embeddings.device,
    )
    for row, (time_count, height, width_count) in enumerate(grid_shapes):
        token_count = metadata.token_counts[row]
        tokens[row, :time_count, :height, :width_count] = adapted_embeddings[
            row, :token_count
        ].reshape(time_count, height, width_count, hidden_dim)
        geometry_mask[row, :time_count, :height, :width_count] = True
    effective_tubelet_mask = tubelet_valid_mask & geometric_tubelet_mask
    spatial_mask = geometry_mask & effective_tubelet_mask[:, :, None, None]
    return RestoredMergedGrid(
        tokens=tokens,
        geometry_valid_mask=geometry_mask,
        spatial_valid_mask=spatial_mask,
        tubelet_valid_mask=effective_tubelet_mask,
        grid_shapes=grid_shapes,
    )


def build_spatial_encoder(config: ProjectConfig | None = None) -> SpatialObjectEncoder:
    if config is None:
        raise ValueError("build_spatial_encoder requires a validated ProjectConfig")
    return SpatialObjectEncoder(config.spatial_encoder)


def spatial_encoder_parameter_count(module: SpatialObjectEncoder) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def build_temporal_encoder(config: ProjectConfig | None = None) -> TemporalEventEncoder:
    if config is None:
        raise ValueError("build_temporal_encoder requires a validated ProjectConfig")
    return TemporalEventEncoder(config.temporal_encoder)


def temporal_encoder_parameter_count(module: TemporalEventEncoder) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class StateEncoders(nn.Module):  # type: ignore[misc]
    """Registered P6/P7 encoder pair without top-level model orchestration."""

    def __init__(self, spatial: SpatialObjectEncoder, temporal: TemporalEventEncoder) -> None:
        super().__init__()
        self.spatial = spatial
        self.temporal = temporal


def build_state_encoders(config: ProjectConfig | None = None) -> StateEncoders:
    if config is None:
        raise ValueError("build_state_encoders requires a validated ProjectConfig")
    return StateEncoders(
        spatial=build_spatial_encoder(config),
        temporal=build_temporal_encoder(config),
    )


def _validate_temporal_config(config: TemporalEncoderConfig) -> None:
    expected: dict[str, object] = {
        "input_dim": 4096,
        "hidden_dim": 768,
        "num_layers": 6,
        "num_heads": 12,
        "head_dim": 64,
        "ffn_dim": 3072,
        "dropout": 0.1,
        "position_encoding": "absolute_sinusoidal",
        "layer_norm_eps": 1.0e-5,
        "activation": "gelu",
        "pre_norm": True,
        "attention_projection_bias": True,
        "strict_causal": True,
        "causal_includes_self": True,
        "causal_window_includes_current": True,
        "cache_tubelets": 64,
        "cache_mode": "layerwise_kv",
        "position_id_mode": "explicit_global",
        "overlap_policy": "replay_replace",
        "overlap_tubelets": 4,
        "replay_context_tubelets": 3,
        "cache_owner_keys": ("video_id", "trajectory_id", "query_signature"),
        "detach_cache_default": True,
        "query_dim": 512,
        "parameter_count": 48_438_272,
    }
    for field, required in expected.items():
        if getattr(config, field) != required:
            raise ValueError(f"P7 requires temporal {field}={required!r}")
    if config.num_heads * config.head_dim != config.hidden_dim:
        raise ValueError("temporal num_heads * head_dim must equal hidden_dim")


def _temporal_sinusoidal_encoding(
    position_ids: Tensor,
    hidden_dim: int,
    *,
    dtype: torch.dtype,
) -> Tensor:
    positions = position_ids.to(dtype=torch.float64).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float64, device=position_ids.device)
        * (-math.log(10_000.0) / hidden_dim)
    )
    angles = positions * frequencies.unsqueeze(0)
    encoding = torch.zeros(
        position_ids.shape[0],
        hidden_dim,
        dtype=torch.float64,
        device=position_ids.device,
    )
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype)


def _timestamps_match(left: Tensor, right: Tensor) -> bool:
    """Compare source-identical timestamps across legal FP32/FP64 handoffs."""

    if left.shape != right.shape:
        return False
    left_64 = left.to(dtype=torch.float64)
    right_64 = right.to(dtype=torch.float64)
    scale = torch.maximum(left_64.abs(), right_64.abs()).clamp_min(1.0)
    tolerance = 4.0 * torch.finfo(torch.float32).eps * scale
    return bool(torch.all((left_64 - right_64).abs() <= tolerance))


def _empty_temporal_cache(
    video_id: str,
    trajectory_id: str,
    q_target: Tensor,
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
) -> TemporalCache:
    hidden = q_target.new_zeros((1, 0, hidden_dim))
    layer_keys = tuple(q_target.new_zeros((1, num_heads, 0, head_dim)) for _ in range(num_layers))
    layer_values = tuple(q_target.new_zeros((1, num_heads, 0, head_dim)) for _ in range(num_layers))
    replay_layer_keys = tuple(
        q_target.new_zeros((1, num_heads, 0, head_dim)) for _ in range(num_layers)
    )
    replay_layer_values = tuple(
        q_target.new_zeros((1, num_heads, 0, head_dim)) for _ in range(num_layers)
    )
    return TemporalCache(
        hidden=hidden,
        layer_keys=layer_keys,
        layer_values=layer_values,
        replay_layer_keys=replay_layer_keys,
        replay_layer_values=replay_layer_values,
        timestamps=torch.empty((1, 0), dtype=torch.float64, device=q_target.device),
        replay_timestamps=torch.empty((1, 0), dtype=torch.float64, device=q_target.device),
        position_ids=torch.empty((1, 0), dtype=torch.int64, device=q_target.device),
        replay_position_ids=torch.empty((1, 0), dtype=torch.int64, device=q_target.device),
        valid_mask=torch.empty((1, 0), dtype=torch.bool, device=q_target.device),
        replay_valid_mask=torch.empty((1, 0), dtype=torch.bool, device=q_target.device),
        video_ids=(video_id,),
        trajectory_ids=(trajectory_id,),
        query_signatures=q_target.detach().reshape(1, -1).clone(),
        total_seen=torch.zeros(1, dtype=torch.int64, device=q_target.device),
        differentiable=False,
    )


def _cache_tensor(tensor: Tensor, detach: bool) -> Tensor:
    return tensor.detach().clone() if detach else tensor.clone()


def _clone_temporal_cache(cache: TemporalCache, *, detach: bool) -> TemporalCache:
    return TemporalCache(
        hidden=_cache_tensor(cache.hidden, detach),
        layer_keys=tuple(_cache_tensor(value, detach) for value in cache.layer_keys),
        layer_values=tuple(_cache_tensor(value, detach) for value in cache.layer_values),
        replay_layer_keys=tuple(_cache_tensor(value, detach) for value in cache.replay_layer_keys),
        replay_layer_values=tuple(
            _cache_tensor(value, detach) for value in cache.replay_layer_values
        ),
        timestamps=cache.timestamps.detach().clone(),
        replay_timestamps=cache.replay_timestamps.detach().clone(),
        position_ids=cache.position_ids.clone(),
        replay_position_ids=cache.replay_position_ids.clone(),
        valid_mask=cache.valid_mask.clone(),
        replay_valid_mask=cache.replay_valid_mask.clone(),
        video_ids=cache.video_ids,
        trajectory_ids=cache.trajectory_ids,
        query_signatures=cache.query_signatures.detach().clone(),
        total_seen=cache.total_seen.clone(),
        differentiable=not detach,
    )


def _validate_spatial_config(config: SpatialEncoderConfig) -> None:
    if config.stages != 2:
        raise ValueError("P6 requires exactly two independent Slot Attention stages")
    if config.num_heads * config.head_dim != config.hidden_dim:
        raise ValueError("spatial num_heads * head_dim must equal hidden_dim")
    if config.active_slots > config.max_active_slots:
        raise ValueError("spatial active_slots cannot exceed max_active_slots")
    expected = {
        "slot_initialization": "shared_seed_plus_fixed_sinusoidal_codes",
        "attention_normalization": "softmax_slots_then_normalize_tokens",
        "confidence_mode": "attention_occupancy",
        "overflow_policy": "preserve_existing_reject_excess",
    }
    for field, required in expected.items():
        if getattr(config, field) != required:
            raise ValueError(f"P6 requires spatial {field}={required!r}")
    if not config.slot_valid_mask or not config.log_overflow:
        raise ValueError("P6 requires slot validity and overflow auditing")


def _sinusoidal_slot_codes(slot_count: int, hidden_dim: int) -> Tensor:
    positions = torch.arange(slot_count, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / hidden_dim)
    )
    codes = torch.zeros(slot_count, hidden_dim, dtype=torch.float32)
    codes[:, 0::2] = torch.sin(positions * frequencies)
    if hidden_dim > 1:
        codes[:, 1::2] = torch.cos(positions * frequencies[: codes[:, 1::2].shape[1]])
    return codes / math.sqrt(hidden_dim)


def _normalize_video_ids(video_ids: Sequence[str], batch_size: int) -> tuple[str, ...]:
    normalized = tuple(video_ids)
    if len(normalized) != batch_size or any(not video_id for video_id in normalized):
        raise ValueError("spatial encoder requires one non-empty video_id per batch row")
    if len(set(normalized)) != len(normalized):
        raise ValueError("one spatial batch cannot contain duplicate video_ids")
    return normalized


def _normalize_query_valid_mask(q_target: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return torch.ones(q_target.shape[0], dtype=torch.bool, device=q_target.device)
    if mask.shape != (q_target.shape[0],) or mask.dtype != torch.bool:
        raise ValueError("query_valid_mask must be bool [B]")
    if mask.device != q_target.device:
        raise ValueError("query_valid_mask and q_target must share a device")
    return mask


def _normalize_prior_states(
    states: Sequence[SpatialSlotRuntimeState | None] | None,
    batch_size: int,
) -> tuple[SpatialSlotRuntimeState | None, ...]:
    if states is None:
        return (None,) * batch_size
    normalized = tuple(states)
    if len(normalized) != batch_size:
        raise ValueError("spatial encoder requires one prior state entry per batch row")
    if any(
        state is not None and not isinstance(state, SpatialSlotRuntimeState) for state in normalized
    ):
        raise TypeError("prior_states must contain SpatialSlotRuntimeState or None")
    return normalized


def _normalize_required_slot_counts(
    counts: Tensor | None,
    batch_size: int,
    device: torch.device,
    default_count: int,
) -> Tensor:
    if counts is None:
        return torch.full((batch_size,), default_count, dtype=torch.int64, device=device)
    if counts.shape != (batch_size,) or counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("required_slot_counts must be integer [B]")
    if counts.device != device:
        raise ValueError("required_slot_counts and inputs must share a device")
    if bool(torch.any(counts < 0)):
        raise ValueError("required_slot_counts must be non-negative")
    return counts.to(dtype=torch.int64)


def _shares_storage(left: Tensor, right: Tensor) -> bool:
    if left.device.type == "meta" or right.device.type == "meta":
        return left is right
    return int(left.untyped_storage().data_ptr()) == int(right.untyped_storage().data_ptr())


def _assert_runtime_state_storage_isolated(states: Sequence[SpatialSlotRuntimeState]) -> None:
    tensors = tuple(
        tensor
        for state in states
        for tensor in (state.slots, state.slot_valid_mask, state.slot_confidence)
    )
    for left_index, left in enumerate(tensors):
        for right in tensors[left_index + 1 :]:
            if _shares_storage(left, right):
                raise ValueError("spatial runtime batch rows must not share mutable storage")


def _assert_optional_runtime_state_storage_isolated(
    states: Sequence[SpatialSlotRuntimeState | None],
) -> None:
    present = tuple(state for state in states if state is not None)
    _assert_runtime_state_storage_isolated(present)
