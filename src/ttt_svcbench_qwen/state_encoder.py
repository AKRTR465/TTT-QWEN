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


@dataclass(frozen=True, slots=True)
class SpatialEncoderOutput:
    slots: Tensor
    slot_valid_mask: Tensor
    active_slot_overflow_count: Tensor
    slot_confidence: Tensor | None = None
    next_states: tuple[SpatialSlotRuntimeState, ...] | None = None

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

    @property
    def batch_size(self) -> int:
        return int(self.hidden.shape[0])

    @property
    def cache_length(self) -> int:
        return int(self.hidden.shape[1])

    @property
    def replay_length(self) -> int:
        return int(self.replay_position_ids.shape[1])

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
        reference = normalized[0]
        video_ids = tuple(state.video_ids[0] for state in normalized)
        trajectory_ids = tuple(state.trajectory_ids[0] for state in normalized)
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
class TemporalEncoderOutput:
    hidden: Tensor
    timestamps: Tensor
    position_ids: Tensor
    valid_mask: Tensor
    cache: TemporalCache


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
            # `query_condition` is mathematically inert here and is kept only
            # because removing it is not bitwise-neutral.  `conditioned` feeds
            # nothing but `queries`, `queries` feed nothing but `logits`, and
            # `q_projection` is linear, so this term lands on every slot equally;
            # the softmax below is taken over the slot axis (`dim=2`), which is
            # invariant to a slot-constant shift.  Query conditioning does its
            # real work once, in `_initial_slots`.  Dropping the term here shifts
            # the pre-softmax magnitudes and perturbs the result at ~1.7e-6 in
            # FP32, so it is retained deliberately rather than cleaned up.
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
        query_condition = self.query_projection(valid_query)
        safe_tokens = torch.where(
            restored.spatial_valid_mask.unsqueeze(-1),
            restored.tokens,
            0.0,
        )
        states = _normalize_prior_states(prior_states, batch_size)

        fresh_slots = self._initial_slots(query_condition)
        current_rows: list[Tensor] = []
        mask_rows: list[Tensor] = []
        confidence_rows: list[Tensor] = []
        prior_overflow: list[int] = []
        prior_events: list[int] = []
        prior_processed: list[int] = []
        for row, state in enumerate(states):
            if state is None:
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
        return SpatialEncoderOutput(
            slots=current,
            slot_valid_mask=current_mask,
            active_slot_overflow_count=overflow_counts,
            slot_confidence=current_confidence,
            next_states=next_states,
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

        safe_query = q_target if query_valid else torch.zeros_like(q_target)
        condition = self.query_projection(safe_query.unsqueeze(0))
        slots = self._initial_slots(condition)[0]
        if slot_valid_mask is None:
            valid_mask = torch.ones(
                self.config.active_slots,
                dtype=torch.bool,
                device=slots.device,
            )
        else:
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


class TemporalEventEncoder(nn.Module):  # type: ignore[misc]
    """Query-conditioned tubelet pooling followed by a six-layer causal Transformer."""

    def __init__(self, config: TemporalEncoderConfig) -> None:
        super().__init__()
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

        restored = restore_merged_grid(
            adapted_embeddings,
            visual_valid_mask,
            metadata,
            tubelet_valid_mask,
        )
        normalized_video_ids = tuple(video_ids)
        normalized_trajectory_ids = tuple(trajectory_ids)
        pooled, _ = self.spatial_pool(restored, q_target)
        batch_size, max_time, _ = pooled.shape
        prior_rows = cache.split() if cache is not None else (None,) * batch_size
        hidden_rows: list[Tensor] = []
        next_rows: list[TemporalCache] = []

        for row in range(batch_size):
            valid_count = int(restored.tubelet_valid_mask[row].sum().item())
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
            combined_context_timestamps = torch.cat(
                (history.timestamps, current_timestamps.to(dtype=torch.float64)), dim=0
            )
            combined_context_positions = torch.cat((history.position_ids, current_positions), dim=0)
            main_count = min(combined_context_positions.shape[0], self.config.cache_tubelets)
            context_main_start = combined_context_positions.shape[0] - main_count
            context_replay_start = max(0, context_main_start - self.config.replay_context_tubelets)
            cache_hidden = combined_hidden[:, -main_count:]
            cache_timestamps = combined_context_timestamps[-main_count:].unsqueeze(0)
            cache_positions = combined_context_positions[-main_count:].unsqueeze(0)
            replay_timestamps = combined_context_timestamps[
                context_replay_start:context_main_start
            ].unsqueeze(0)
            replay_positions = combined_context_positions[
                context_replay_start:context_main_start
            ].unsqueeze(0)
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
        return TemporalEncoderOutput(
            hidden=output_hidden,
            timestamps=output_timestamps,
            position_ids=output_positions,
            valid_mask=restored.tubelet_valid_mask,
            cache=next_cache,
        )

    def reset_cache(
        self,
        video_id: str,
        trajectory_id: str,
        q_target: Tensor,
    ) -> TemporalCache:
        """Create an empty singleton cache with explicit ownership and query signature."""

        return _empty_temporal_cache(
            video_id,
            trajectory_id,
            q_target,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def _prepare_prior_for_positions(
        self,
        prior: TemporalCache,
        current_positions: Tensor,
        current_timestamps: Tensor,
    ) -> _PreparedTemporalHistory:
        if prior.cache_length == 0:
            return _PreparedTemporalHistory(
                layer_keys=prior.layer_keys,
                layer_values=prior.layer_values,
                timestamps=prior.timestamps[0],
                position_ids=prior.position_ids[0],
                retained_hidden=prior.hidden,
                retained_timestamps=prior.timestamps[0],
                retained_position_ids=prior.position_ids[0],
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
        cached_first = int(cached_positions[0].item())
        cached_last = int(cached_positions[-1].item())
        context_first = int(context_positions[0].item())
        retain_context_count = context_positions.shape[0]
        retain_main_count = prior.cache_length
        if first_position <= cached_last:
            retain_context_count = first_position - context_first
            retain_main_count = first_position - cached_first
        return _PreparedTemporalHistory(
            layer_keys=tuple(value[:, :, :retain_context_count] for value in context_keys),
            layer_values=tuple(value[:, :, :retain_context_count] for value in context_values),
            timestamps=context_timestamps[:retain_context_count],
            position_ids=context_positions[:retain_context_count],
            retained_hidden=prior.hidden[:, :retain_main_count],
            retained_timestamps=cached_timestamps[:retain_main_count],
            retained_position_ids=cached_positions[:retain_main_count],
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
    batch_size, width, hidden_dim = adapted_embeddings.shape
    grid_shapes = tuple(
        (int(row[0]), int(row[1]), int(row[2]))
        for row in metadata.merged_grid_thw.detach().cpu().tolist()
    )
    max_t = max(shape[0] for shape in grid_shapes)
    max_h = max(shape[1] for shape in grid_shapes)
    max_w = max(shape[2] for shape in grid_shapes)
    time_positions = torch.arange(max_t, device=adapted_embeddings.device).unsqueeze(0)
    geometric_tubelet_mask = time_positions < torch.tensor(
        [shape[0] for shape in grid_shapes],
        device=adapted_embeddings.device,
    ).unsqueeze(1)
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


def build_spatial_encoder(config: ProjectConfig) -> SpatialObjectEncoder:
    return SpatialObjectEncoder(config.spatial_encoder)


def build_temporal_encoder(config: ProjectConfig) -> TemporalEventEncoder:
    return TemporalEventEncoder(config.temporal_encoder)


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
    return tuple(video_ids)


def _normalize_query_valid_mask(q_target: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return torch.ones(q_target.shape[0], dtype=torch.bool, device=q_target.device)
    return mask


def _normalize_prior_states(
    states: Sequence[SpatialSlotRuntimeState | None] | None,
    batch_size: int,
) -> tuple[SpatialSlotRuntimeState | None, ...]:
    if states is None:
        return (None,) * batch_size
    return tuple(states)


def _normalize_required_slot_counts(
    counts: Tensor | None,
    batch_size: int,
    device: torch.device,
    default_count: int,
) -> Tensor:
    if counts is None:
        return torch.full((batch_size,), default_count, dtype=torch.int64, device=device)
    return counts.to(dtype=torch.int64)

