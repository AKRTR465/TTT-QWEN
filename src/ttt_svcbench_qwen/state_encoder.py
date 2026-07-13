"""Implement spatial slots while retaining the P7 temporal encoder boundary.

Inputs: adapted visual tokens, q_target, masks, grid metadata, and prior per-video state.
Outputs: query-conditioned recurrent spatial slots plus explicit runtime/audit state.
Forbidden: hard counting, semantic overflow inference, Bank mutation, or optimizer steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NoReturn

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import ProjectConfig, SpatialEncoderConfig
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
            expected_spatial = self.geometry_valid_mask & self.tubelet_valid_mask[
                :, :, None, None
            ]
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
        if (
            self.slots.ndim != 2
            or not 1 <= self.slots.shape[0] <= 64
            or self.slots.shape[1] != 768
        ):
            raise ValueError("runtime slots must be [K_a, 768] with 1 <= K_a <= 64")
        if not torch.is_floating_point(self.slots):
            raise TypeError("runtime slots must use a floating dtype")
        if (
            self.slot_valid_mask.shape != self.slots.shape[:1]
            or self.slot_valid_mask.dtype != torch.bool
        ):
            raise ValueError("runtime slot_valid_mask must be bool [K_a]")
        if (
            self.slot_confidence.shape != self.slots.shape[:1]
            or not torch.is_floating_point(self.slot_confidence)
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
        if (
            self.slots.ndim != 3
            or not 1 <= self.slots.shape[1] <= 64
            or self.slots.shape[2] != 768
        ):
            raise ValueError("slots must be [B, K_a, 768] with 1 <= K_a <= 64")
        if not torch.is_floating_point(self.slots):
            raise TypeError("slots must use a floating dtype")
        expected_mask = self.slots.shape[:2]
        if (
            self.slot_valid_mask.shape != expected_mask
            or self.slot_valid_mask.dtype != torch.bool
        ):
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
    hidden: Tensor
    timestamps: Tensor
    valid_mask: Tensor
    video_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.hidden.ndim != 3 or self.hidden.shape[-1] != 768:
            raise ValueError("temporal cache hidden must be [B, T_cache, 768]")
        if self.hidden.shape[1] > 64:
            raise ValueError("temporal cache cannot exceed 64 tubelets")
        shape = self.hidden.shape[:2]
        if self.timestamps.shape != shape or not torch.is_floating_point(self.timestamps):
            raise ValueError("temporal cache timestamps must be floating [B, T_cache]")
        if self.valid_mask.shape != shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("temporal cache valid_mask must be bool [B, T_cache]")
        if len(self.video_ids) != self.hidden.shape[0] or not all(self.video_ids):
            raise ValueError("temporal cache requires one non-empty video_id per batch item")


@dataclass(frozen=True, slots=True)
class TemporalEncoderOutput:
    hidden: Tensor
    timestamps: Tensor
    valid_mask: Tensor
    cache: TemporalCache

    def __post_init__(self) -> None:
        if self.hidden.ndim != 3 or self.hidden.shape[-1] != 768:
            raise ValueError("temporal hidden must be [B, T, 768]")
        shape = self.hidden.shape[:2]
        if self.timestamps.shape != shape or not torch.is_floating_point(self.timestamps):
            raise ValueError("temporal timestamps must be floating [B, T]")
        if self.valid_mask.shape != shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("temporal valid_mask must be bool [B, T]")


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
                logits.float()
                if logits.dtype in (torch.float16, torch.bfloat16)
                else logits
            )
            assignments = torch.softmax(normalization_logits, dim=2)
            valid_pairs = (
                slot_valid_mask[:, None, :, None]
                & token_valid_mask[:, None, None, :]
            )
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
            self.shared_slot_seed[None, None, :]
            + query_condition[:, None, :]
            + codes[None, :, :]
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
            raise ValueError(
                f"q_target must be floating [B, {self.config.query_dim}]"
            )
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
        next_confidence = (
            slot_confidence.detach().clone() if detach else slot_confidence.clone()
        )
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


def build_state_encoders(_config: ProjectConfig | None = None) -> NoReturn:
    """P7 owns the joint spatial/temporal builder after the P6 spatial implementation."""

    raise NotImplementedError("Joint state encoder implementation is deferred to P7")


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
        torch.arange(0, hidden_dim, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / hidden_dim)
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
        state is not None and not isinstance(state, SpatialSlotRuntimeState)
        for state in normalized
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
