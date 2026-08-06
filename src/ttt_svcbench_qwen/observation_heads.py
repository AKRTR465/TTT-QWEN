"""Implement four differentiable soft Observation Decoders and stream state.

Inputs: spatial slots, causal temporal states, masks, timestamps, positions, and owners.
Outputs: O1/O2/E1/E2 logits, diagnostic probabilities, and functional stream states.
Forbidden: hard thresholds, integer accumulation, Bank/FSM mutation, or input detachment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import (
    E1Config,
    E2Config,
    O1Config,
    O2Config,
    ProjectConfig,
)
from ttt_svcbench_qwen.state_encoder import SpatialEncoderOutput, TemporalEncoderOutput


@dataclass(frozen=True, slots=True)
class O1SoftOutput:
    """Per-slot O1 evidence; no field is a hard count or Bank mutation."""

    LOGIT_NAMES: ClassVar[tuple[str, ...]] = (
        "object",
        "target",
        "visible",
        "enter",
        "exit",
        "confidence",
    )

    logits: Tensor
    probabilities: Tensor
    soft_count: Tensor
    valid_mask: Tensor
    timestamps: Tensor
    position_ids: Tensor

    @property
    def count_prediction(self) -> Tensor:
        return self.soft_count


@dataclass(frozen=True, slots=True)
class O2SoftOutput:
    SCORE_NAMES: ClassVar[tuple[str, ...]] = ("novelty", "match_confidence")

    identity: Tensor
    score_logits: Tensor
    score_probabilities: Tensor
    valid_mask: Tensor
    timestamps: Tensor
    position_ids: Tensor
    count_prediction: Tensor = field(default_factory=lambda: torch.empty(0))
    relevance: Tensor = field(default_factory=lambda: torch.empty(0))

    def __post_init__(self) -> None:
        if self.count_prediction.numel() == 0:
            object.__setattr__(
                self,
                "count_prediction",
                (self.score_probabilities[..., 0] * self.valid_mask).sum(dim=1),
            )
        if self.relevance.numel() == 0 and self.valid_mask.numel():
            object.__setattr__(
                self,
                "relevance",
                0.5 * self.valid_mask.to(dtype=self.score_logits.dtype),
            )


@dataclass(frozen=True, slots=True)
class E1RuntimeState:
    """Per-trajectory projected-input history for RF=63 overlap-safe TCN replay."""

    video_id: str
    trajectory_id: str
    query_signature: Tensor
    projected_history: Tensor
    timestamps: Tensor
    position_ids: Tensor
    total_seen: int
    differentiable: bool = False


@dataclass(frozen=True, slots=True)
class E1SoftOutput:
    LOGIT_NAMES: ClassVar[tuple[str, ...]] = (
        "eventness",
        "completion",
        "transition",
    )

    logits: Tensor
    probabilities: Tensor
    valid_mask: Tensor
    timestamps: Tensor
    position_ids: Tensor
    next_states: tuple[E1RuntimeState, ...]
    count_prediction: Tensor = field(default_factory=lambda: torch.empty(0))

    def __post_init__(self) -> None:
        if self.count_prediction.numel() == 0:
            object.__setattr__(
                self,
                "count_prediction",
                (self.probabilities[..., 1] * self.valid_mask).sum(dim=1),
            )


@dataclass(frozen=True, slots=True)
class E2RuntimeState:
    """Per-trajectory GRU state plus five checkpoints for four-position overlap replay."""

    video_id: str
    trajectory_id: str
    query_signature: Tensor
    hidden: Tensor
    checkpoint_hidden: Tensor
    timestamps: Tensor
    position_ids: Tensor
    total_seen: int
    differentiable: bool = False


@dataclass(frozen=True, slots=True)
class E2SoftOutput:
    EVENT_NAMES: ClassVar[tuple[str, ...]] = (
        "start",
        "active",
        "end",
        "complete",
    )
    PHASE_NAMES: ClassVar[tuple[str, ...]] = (
        "inactive",
        "active",
        "end_candidate",
        "completed",
    )

    event_logits: Tensor
    phase_logits: Tensor
    event_probabilities: Tensor
    phase_probabilities: Tensor
    valid_mask: Tensor
    timestamps: Tensor
    position_ids: Tensor
    next_states: tuple[E2RuntimeState, ...]
    count_prediction: Tensor = field(default_factory=lambda: torch.empty(0))

    def __post_init__(self) -> None:
        if self.count_prediction.numel() == 0:
            object.__setattr__(
                self,
                "count_prediction",
                (self.event_probabilities[..., 3] * self.valid_mask).sum(dim=1),
            )


@dataclass(frozen=True, slots=True)
class ObservationOutputs:
    o1: O1SoftOutput
    o2: O2SoftOutput
    e1: E1SoftOutput
    e2: E2SoftOutput


class CumulativeCountHead(nn.Module):  # type: ignore[misc]
    """Unbounded positive cumulative count independent of the local sequence length."""

    def __init__(self, input_dim: int, *, layer_norm_eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.hidden = nn.Linear(input_dim, 256, bias=True)
        self.output = nn.Linear(256, 1, bias=True)

    def forward(self, features: Tensor) -> Tensor:
        logits = self.output(F.silu(self.hidden(self.norm(features)))).squeeze(-1)
        return F.softplus(logits.float()).to(dtype=logits.dtype)


class O1CurrentCountDecoder(nn.Module):  # type: ignore[misc]
    def __init__(self, config: O1Config) -> None:
        super().__init__()
        self.config = config
        self.slot_norm = nn.LayerNorm(config.input_dim, eps=config.layer_norm_eps)
        self.film_projection = nn.Linear(config.query_dim, config.film_dim, bias=True)
        self.mlp_1 = nn.Linear(config.input_dim, config.hidden_dims[0], bias=True)
        self.mlp_2 = nn.Linear(config.hidden_dims[0], config.hidden_dims[1], bias=True)
        self.output_projection = nn.Linear(config.hidden_dims[1], config.output_dim, bias=True)

    def forward(
        self,
        slots: Tensor,
        slot_valid_mask: Tensor,
        q_target: Tensor,
        observation_timestamps: Tensor,
        observation_position_ids: Tensor,
    ) -> O1SoftOutput:
        safe_slots, expanded_timestamps, expanded_positions = _prepare_spatial_head_inputs(
            slots,
            slot_valid_mask,
            observation_timestamps,
            observation_position_ids,
        )
        scale, shift = self.film_projection(q_target).chunk(2, dim=-1)
        conditioned = self.slot_norm(safe_slots) * (1.0 + scale.unsqueeze(1))
        conditioned = conditioned + shift.unsqueeze(1)
        hidden = F.silu(self.mlp_1(conditioned))
        hidden = F.silu(self.mlp_2(hidden))
        logits = self.output_projection(hidden)
        logits = torch.where(slot_valid_mask.unsqueeze(-1), logits, 0.0)
        probabilities = torch.sigmoid(logits.float()).to(dtype=logits.dtype)
        probabilities = torch.where(slot_valid_mask.unsqueeze(-1), probabilities, 0.0)
        soft_count = (
            probabilities[..., 0]
            * probabilities[..., 1]
            * probabilities[..., 2]
            * slot_valid_mask.to(dtype=probabilities.dtype)
        ).sum(dim=1)
        return O1SoftOutput(
            logits=logits,
            probabilities=probabilities,
            soft_count=soft_count,
            valid_mask=slot_valid_mask.clone(),
            timestamps=expanded_timestamps,
            position_ids=expanded_positions,
        )


class O2IdentityDecoder(nn.Module):  # type: ignore[misc]
    def __init__(self, config: O2Config) -> None:
        super().__init__()
        self.config = config
        self.slot_norm = nn.LayerNorm(config.input_dim, eps=config.layer_norm_eps)
        self.trunk_1 = nn.Linear(config.input_dim, config.hidden_dims[0], bias=True)
        self.trunk_2 = nn.Linear(config.hidden_dims[0], config.hidden_dims[1], bias=True)
        self.identity_projection = nn.Linear(config.hidden_dims[1], config.identity_dim, bias=True)
        self.score_projection = nn.Linear(config.hidden_dims[1], config.score_dim, bias=True)
        self.count_head = CumulativeCountHead(
            config.hidden_dims[1] + 512,
            layer_norm_eps=config.layer_norm_eps,
        )
        # Multiplicative query interaction: sigma(<identity_i, W q_target>). A concat-Linear
        # would rank slots independently of the query and cannot express "is this the asked
        # class"; the bilinear form is the smallest head that can.
        self.relevance_projection = nn.Linear(512, config.identity_dim, bias=True)

    def forward(
        self,
        slots: Tensor,
        slot_valid_mask: Tensor,
        observation_timestamps: Tensor,
        observation_position_ids: Tensor,
        *,
        q_target: Tensor | None = None,
    ) -> O2SoftOutput:
        safe_slots, expanded_timestamps, expanded_positions = _prepare_spatial_head_inputs(
            slots,
            slot_valid_mask,
            observation_timestamps,
            observation_position_ids,
        )
        hidden = F.silu(self.trunk_1(self.slot_norm(safe_slots)))
        hidden = F.silu(self.trunk_2(hidden))
        if q_target is None:
            q_target = hidden.new_zeros((hidden.shape[0], 512))
        valid_weights = slot_valid_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * valid_weights).sum(dim=1) / valid_weights.sum(dim=1).clamp_min(1.0)
        count_prediction = self.count_head(torch.cat((pooled, q_target), dim=-1))
        raw_identity = self.identity_projection(hidden)
        raw_fp32 = raw_identity.float()
        norms = torch.linalg.vector_norm(raw_fp32, dim=-1, keepdim=True)
        fallback = torch.zeros_like(raw_fp32)
        fallback[..., 0] = 1.0
        safe_identity = torch.where(
            norms > self.config.normalization_eps,
            raw_fp32,
            fallback,
        )
        identity = F.normalize(
            safe_identity,
            dim=-1,
            eps=self.config.normalization_eps,
        ).to(dtype=raw_identity.dtype)
        identity = torch.where(slot_valid_mask.unsqueeze(-1), identity, 0.0)
        score_logits = self.score_projection(hidden)
        score_logits = torch.where(slot_valid_mask.unsqueeze(-1), score_logits, 0.0)
        score_probabilities = torch.sigmoid(score_logits.float()).to(dtype=score_logits.dtype)
        score_probabilities = torch.where(slot_valid_mask.unsqueeze(-1), score_probabilities, 0.0)
        relevance_query = self.relevance_projection(q_target)
        relevance = torch.sigmoid(
            torch.einsum("bnd,bd->bn", identity.float(), relevance_query.float())
        ).to(dtype=identity.dtype)
        relevance = torch.where(slot_valid_mask, relevance, 0.0)
        return O2SoftOutput(
            identity=identity,
            score_logits=score_logits,
            score_probabilities=score_probabilities,
            valid_mask=slot_valid_mask.clone(),
            timestamps=expanded_timestamps,
            position_ids=expanded_positions,
            count_prediction=count_prediction,
            relevance=relevance,
        )


class GatedCausalTCNBlock(nn.Module):  # type: ignore[misc]
    def __init__(self, channels: int, kernel_size: int, dilation: int, layer_norm_eps: float):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_padding = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=0,
            bias=True,
        )
        self.gate_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=0,
            bias=True,
        )
        self.residual_projection = nn.Conv1d(channels, channels, 1, bias=True)
        self.output_norm = nn.LayerNorm(channels, eps=layer_norm_eps)

    def forward(self, states: Tensor) -> Tensor:
        channels_first = states.transpose(1, 2)
        padded = F.pad(channels_first, (self.left_padding, 0))
        filtered = F.silu(self.filter_conv(padded))
        gated = torch.sigmoid(self.gate_conv(padded))
        residual = channels_first + self.residual_projection(filtered * gated)
        return self.output_norm(residual.transpose(1, 2))


class E1PointEventDecoder(nn.Module):  # type: ignore[misc]
    def __init__(self, config: E1Config) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim, eps=config.layer_norm_eps)
        self.input_projection = nn.Linear(config.input_dim, config.channels, bias=True)
        self.blocks = nn.ModuleList(
            GatedCausalTCNBlock(
                config.channels,
                config.kernel_size,
                dilation,
                config.layer_norm_eps,
            )
            for dilation in config.dilations
        )
        self.output_projection = nn.Linear(config.channels, config.output_dim, bias=True)
        self.count_head = CumulativeCountHead(
            config.channels,
            layer_norm_eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden: Tensor,
        valid_mask: Tensor,
        timestamps: Tensor,
        position_ids: Tensor,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
        query_signatures: Tensor,
        *,
        prior_states: Sequence[E1RuntimeState | None] | None = None,
        detach_runtime_state: bool = True,
    ) -> E1SoftOutput:
        safe_hidden, normalized_timestamps = _prepare_temporal_head_inputs(
            hidden,
            valid_mask,
            timestamps,
        )
        owners = _normalize_stream_owners(video_ids, trajectory_ids, hidden.shape[0], "E1")
        states = _normalize_stream_states(prior_states, hidden.shape[0])
        projected = self.input_projection(self.input_norm(safe_hidden))
        projected = torch.where(valid_mask.unsqueeze(-1), projected, 0.0)
        output_rows: list[Tensor] = []
        next_states: list[E1RuntimeState] = []
        count_features: list[Tensor] = []
        for row in range(hidden.shape[0]):
            state = states[row] or self._empty_state(
                owners[0][row],
                owners[1][row],
                query_signatures[row],
            )
            count = int(valid_mask[row].sum().item())
            if count == 0:
                next_state = _clone_e1_state(state, detach=detach_runtime_state)
                output_rows.append(hidden.new_zeros((1, hidden.shape[1], 3)))
                next_states.append(next_state)
                count_features.append(projected[row].new_zeros((self.config.channels,)))
                continue
            current_positions = position_ids[row, :count]
            current_timestamps = normalized_timestamps[row, :count]
            history, history_timestamps, history_positions, overlap_count = self._prepare_history(
                state, current_positions, current_timestamps
            )
            current_projected = projected[row, :count]
            combined = torch.cat((history, current_projected), dim=0)
            encoded = combined.unsqueeze(0)
            for block in self.blocks:
                encoded = block(encoded)
            current_logits = self.output_projection(encoded[:, -count:])
            combined_timestamps = torch.cat((history_timestamps, current_timestamps), dim=0)
            combined_positions = torch.cat((history_positions, current_positions), dim=0)
            next_state = self._make_state(
                owners[0][row],
                owners[1][row],
                query_signatures[row],
                combined,
                combined_timestamps,
                combined_positions,
                detach=detach_runtime_state,
            )
            output_rows.append(F.pad(current_logits, (0, 0, 0, hidden.shape[1] - count), value=0.0))
            next_states.append(next_state)
            count_features.append(encoded[0, -1])
        logits = torch.cat(output_rows, dim=0)
        logits = torch.where(valid_mask.unsqueeze(-1), logits, 0.0)
        probabilities = torch.sigmoid(logits.float()).to(dtype=logits.dtype)
        probabilities = torch.where(valid_mask.unsqueeze(-1), probabilities, 0.0)
        return E1SoftOutput(
            logits=logits,
            probabilities=probabilities,
            valid_mask=valid_mask.clone(),
            timestamps=normalized_timestamps,
            position_ids=position_ids.clone(),
            next_states=tuple(next_states),
            count_prediction=self.count_head(torch.stack(count_features, dim=0)),
        )

    def _empty_state(
        self,
        video_id: str,
        trajectory_id: str,
        query_signature: Tensor,
    ) -> E1RuntimeState:
        parameter = next(self.parameters())
        return E1RuntimeState(
            video_id=video_id,
            trajectory_id=trajectory_id,
            query_signature=query_signature.detach().clone(),
            projected_history=parameter.new_zeros((0, self.config.channels)),
            timestamps=torch.empty(0, dtype=torch.float64, device=parameter.device),
            position_ids=torch.empty(0, dtype=torch.int64, device=parameter.device),
            total_seen=0,
            differentiable=False,
        )

    def _prepare_history(
        self,
        state: E1RuntimeState,
        current_positions: Tensor,
        current_timestamps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        if state.total_seen == 0:
            return (
                state.projected_history,
                state.timestamps,
                state.position_ids,
                0,
            )
        first = int(current_positions[0].item())
        cached_first = int(state.position_ids[0].item())
        cached_last = int(state.position_ids[-1].item())
        retain_count = int(state.position_ids.shape[0])
        overlap_count = 0
        if first <= cached_last:
            retain_count = first - cached_first
            overlap_count = cached_last - first + 1
        return (
            state.projected_history[:retain_count],
            state.timestamps[:retain_count],
            state.position_ids[:retain_count],
            overlap_count,
        )

    def _make_state(
        self,
        video_id: str,
        trajectory_id: str,
        query_signature: Tensor,
        history: Tensor,
        timestamps: Tensor,
        position_ids: Tensor,
        *,
        detach: bool,
    ) -> E1RuntimeState:
        history = history[-self.config.history_tubelets :]
        timestamps = timestamps[-self.config.history_tubelets :]
        position_ids = position_ids[-self.config.history_tubelets :]
        return E1RuntimeState(
            video_id=video_id,
            trajectory_id=trajectory_id,
            query_signature=query_signature.detach().clone(),
            projected_history=_runtime_tensor(history, detach),
            timestamps=timestamps.detach().to(dtype=torch.float64).clone(),
            position_ids=position_ids.clone(),
            total_seen=int(position_ids[-1].item()) + 1,
            differentiable=not detach,
        )


class E2IntervalEventDecoder(nn.Module):  # type: ignore[misc]
    def __init__(self, config: E2Config) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim, eps=config.layer_norm_eps)
        self.gru = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            bias=True,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=False,
        )
        self.event_projection = nn.Linear(config.hidden_dim, config.event_output_dim, bias=True)
        self.phase_projection = nn.Linear(config.hidden_dim, config.phase_output_dim, bias=True)
        self.count_head = CumulativeCountHead(
            config.hidden_dim,
            layer_norm_eps=config.layer_norm_eps,
        )

    def forward(
        self,
        hidden: Tensor,
        valid_mask: Tensor,
        timestamps: Tensor,
        position_ids: Tensor,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
        query_signatures: Tensor,
        *,
        prior_states: Sequence[E2RuntimeState | None] | None = None,
        detach_runtime_state: bool = True,
    ) -> E2SoftOutput:
        safe_hidden, normalized_timestamps = _prepare_temporal_head_inputs(
            hidden,
            valid_mask,
            timestamps,
        )
        owners = _normalize_stream_owners(video_ids, trajectory_ids, hidden.shape[0], "E2")
        states = _normalize_stream_states(prior_states, hidden.shape[0])
        normalized = self.input_norm(safe_hidden)
        event_rows: list[Tensor] = []
        phase_rows: list[Tensor] = []
        next_states: list[E2RuntimeState] = []
        count_features: list[Tensor] = []
        for row in range(hidden.shape[0]):
            state = states[row] or self._empty_state(
                owners[0][row],
                owners[1][row],
                query_signatures[row],
            )
            count = int(valid_mask[row].sum().item())
            if count == 0:
                next_state = _clone_e2_state(state, detach=detach_runtime_state)
                event_rows.append(hidden.new_zeros((1, hidden.shape[1], 4)))
                phase_rows.append(hidden.new_zeros((1, hidden.shape[1], 4)))
                next_states.append(next_state)
                count_features.append(state.hidden[-1])
                continue
            current_positions = position_ids[row, :count]
            current_timestamps = normalized_timestamps[row, :count]
            initial_hidden, retained_hidden, retained_times, retained_positions, overlap = (
                self._prepare_state(state, current_positions, current_timestamps)
            )
            recurrent_hidden = initial_hidden.unsqueeze(1)
            outputs: list[Tensor] = []
            checkpoints: list[Tensor] = []
            for index in range(count):
                step_output, recurrent_hidden = self.gru(
                    normalized[row : row + 1, index : index + 1],
                    recurrent_hidden,
                )
                outputs.append(step_output)
                checkpoints.append(recurrent_hidden.squeeze(1))
            current_output = torch.cat(outputs, dim=1)
            current_checkpoints = torch.stack(checkpoints, dim=0)
            combined_checkpoints = torch.cat((retained_hidden, current_checkpoints), dim=0)
            combined_timestamps = torch.cat((retained_times, current_timestamps), dim=0)
            combined_positions = torch.cat((retained_positions, current_positions), dim=0)
            next_state = self._make_state(
                owners[0][row],
                owners[1][row],
                query_signatures[row],
                recurrent_hidden.squeeze(1),
                combined_checkpoints,
                combined_timestamps,
                combined_positions,
                detach=detach_runtime_state,
            )
            event_logits = self.event_projection(current_output)
            phase_logits = self.phase_projection(current_output)
            event_rows.append(F.pad(event_logits, (0, 0, 0, hidden.shape[1] - count), value=0.0))
            phase_rows.append(F.pad(phase_logits, (0, 0, 0, hidden.shape[1] - count), value=0.0))
            next_states.append(next_state)
            count_features.append(recurrent_hidden[-1, 0])
        event_logits = torch.cat(event_rows, dim=0)
        phase_logits = torch.cat(phase_rows, dim=0)
        event_logits = torch.where(valid_mask.unsqueeze(-1), event_logits, 0.0)
        phase_logits = torch.where(valid_mask.unsqueeze(-1), phase_logits, 0.0)
        event_probabilities = torch.sigmoid(event_logits.float()).to(dtype=event_logits.dtype)
        phase_probabilities = torch.softmax(phase_logits.float(), dim=-1).to(
            dtype=phase_logits.dtype
        )
        event_probabilities = torch.where(valid_mask.unsqueeze(-1), event_probabilities, 0.0)
        phase_probabilities = torch.where(valid_mask.unsqueeze(-1), phase_probabilities, 0.0)
        return E2SoftOutput(
            event_logits=event_logits,
            phase_logits=phase_logits,
            event_probabilities=event_probabilities,
            phase_probabilities=phase_probabilities,
            valid_mask=valid_mask.clone(),
            timestamps=normalized_timestamps,
            position_ids=position_ids.clone(),
            next_states=tuple(next_states),
            count_prediction=self.count_head(torch.stack(count_features, dim=0)),
        )

    def _empty_state(
        self,
        video_id: str,
        trajectory_id: str,
        query_signature: Tensor,
    ) -> E2RuntimeState:
        parameter = next(self.parameters())
        hidden = parameter.new_zeros((self.config.num_layers, self.config.hidden_dim))
        return E2RuntimeState(
            video_id=video_id,
            trajectory_id=trajectory_id,
            query_signature=query_signature.detach().clone(),
            hidden=hidden,
            checkpoint_hidden=parameter.new_zeros(
                (0, self.config.num_layers, self.config.hidden_dim)
            ),
            timestamps=torch.empty(0, dtype=torch.float64, device=parameter.device),
            position_ids=torch.empty(0, dtype=torch.int64, device=parameter.device),
            total_seen=0,
            differentiable=False,
        )

    def _prepare_state(
        self,
        state: E2RuntimeState,
        current_positions: Tensor,
        current_timestamps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        if state.total_seen == 0:
            return (
                state.hidden,
                state.checkpoint_hidden,
                state.timestamps,
                state.position_ids,
                0,
            )
        first = int(current_positions[0].item())
        cached_first = int(state.position_ids[0].item())
        cached_last = int(state.position_ids[-1].item())
        retain_count = int(state.position_ids.shape[0])
        overlap_count = 0
        initial_hidden = state.hidden
        if first <= cached_last:
            predecessor = first - 1
            predecessor_index = predecessor - cached_first
            retain_count = predecessor_index + 1
            initial_hidden = state.checkpoint_hidden[predecessor_index]
            overlap_count = cached_last - first + 1
        return (
            initial_hidden,
            state.checkpoint_hidden[:retain_count],
            state.timestamps[:retain_count],
            state.position_ids[:retain_count],
            overlap_count,
        )

    def _make_state(
        self,
        video_id: str,
        trajectory_id: str,
        query_signature: Tensor,
        hidden: Tensor,
        checkpoints: Tensor,
        timestamps: Tensor,
        position_ids: Tensor,
        *,
        detach: bool,
    ) -> E2RuntimeState:
        checkpoints = checkpoints[-self.config.checkpoint_tubelets :]
        timestamps = timestamps[-self.config.checkpoint_tubelets :]
        position_ids = position_ids[-self.config.checkpoint_tubelets :]
        return E2RuntimeState(
            video_id=video_id,
            trajectory_id=trajectory_id,
            query_signature=query_signature.detach().clone(),
            hidden=_runtime_tensor(hidden, detach),
            checkpoint_hidden=_runtime_tensor(checkpoints, detach),
            timestamps=timestamps.detach().to(dtype=torch.float64).clone(),
            position_ids=position_ids.clone(),
            total_seen=int(position_ids[-1].item()) + 1,
            differentiable=not detach,
        )


class ObservationHeads(nn.Module):  # type: ignore[misc]
    """Registered four-head bundle; this is not top-level P13 orchestration."""

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.config = config.observation_heads
        self.o1 = O1CurrentCountDecoder(self.config.o1)
        self.o2 = O2IdentityDecoder(self.config.o2)
        self.e1 = E1PointEventDecoder(self.config.e1)
        self.e2 = E2IntervalEventDecoder(self.config.e2)

    def forward(
        self,
        spatial: SpatialEncoderOutput,
        temporal: TemporalEncoderOutput,
        q_target: Tensor,
        video_ids: Sequence[str],
        trajectory_ids: Sequence[str],
        *,
        e1_prior_states: Sequence[E1RuntimeState | None] | None = None,
        e2_prior_states: Sequence[E2RuntimeState | None] | None = None,
        detach_runtime_state: bool = True,
    ) -> ObservationOutputs:
        batch_size = spatial.slots.shape[0]
        owners = _normalize_stream_owners(
            video_ids,
            trajectory_ids,
            batch_size,
            "ObservationHeads",
        )
        row_has_time = temporal.valid_mask.any(dim=1)
        effective_slot_mask = spatial.slot_valid_mask & row_has_time.unsqueeze(1)
        observation_timestamps = torch.full(
            (batch_size,),
            -1.0,
            dtype=temporal.timestamps.dtype,
            device=temporal.timestamps.device,
        )
        observation_position_ids = torch.full(
            (batch_size,),
            -1,
            dtype=torch.int64,
            device=temporal.position_ids.device,
        )
        for row in range(batch_size):
            count = int(temporal.valid_mask[row].sum().item())
            if count:
                observation_timestamps[row] = temporal.timestamps[row, count - 1]
                observation_position_ids[row] = temporal.position_ids[row, count - 1]
        o1 = self.o1(
            spatial.slots,
            effective_slot_mask,
            q_target,
            observation_timestamps,
            observation_position_ids,
        )
        o2 = self.o2(
            spatial.slots,
            effective_slot_mask,
            observation_timestamps,
            observation_position_ids,
            q_target=q_target,
        )
        e1 = self.e1(
            temporal.hidden,
            temporal.valid_mask,
            temporal.timestamps,
            temporal.position_ids,
            owners[0],
            owners[1],
            temporal.cache.query_signatures,
            prior_states=e1_prior_states,
            detach_runtime_state=detach_runtime_state,
        )
        e2 = self.e2(
            temporal.hidden,
            temporal.valid_mask,
            temporal.timestamps,
            temporal.position_ids,
            owners[0],
            owners[1],
            temporal.cache.query_signatures,
            prior_states=e2_prior_states,
            detach_runtime_state=detach_runtime_state,
        )
        return ObservationOutputs(o1=o1, o2=o2, e1=e1, e2=e2)

    def set_online_frozen(self, frozen: bool = True) -> ObservationHeads:
        """Freeze decoder parameters without disabling gradients to decoder inputs."""

        for parameter in self.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.eval()
        return self

    @property
    def online_frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.parameters())


def build_observation_heads(config: ProjectConfig | None = None) -> ObservationHeads:
    return ObservationHeads(config)


def _prepare_spatial_head_inputs(
    slots: Tensor,
    valid_mask: Tensor,
    observation_timestamps: Tensor,
    observation_position_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    slot_count = slots.shape[1]
    expanded = observation_timestamps.unsqueeze(1).expand(-1, slot_count)
    expanded = torch.where(valid_mask, expanded, torch.full_like(expanded, -1.0))
    expanded_positions = observation_position_ids.unsqueeze(1).expand(-1, slot_count)
    expanded_positions = torch.where(
        valid_mask,
        expanded_positions,
        torch.full_like(expanded_positions, -1),
    )
    safe_slots = torch.where(valid_mask.unsqueeze(-1), slots, 0.0)
    return safe_slots, expanded.clone(), expanded_positions.clone()


def _prepare_temporal_head_inputs(
    hidden: Tensor,
    valid_mask: Tensor,
    timestamps: Tensor,
) -> tuple[Tensor, Tensor]:
    safe_hidden = torch.where(valid_mask.unsqueeze(-1), hidden, 0.0)
    normalized_timestamps = torch.where(
        valid_mask,
        timestamps.to(dtype=torch.float64),
        torch.full_like(timestamps, -1.0, dtype=torch.float64),
    )
    return safe_hidden, normalized_timestamps


def _normalize_stream_owners(
    video_ids: Sequence[str],
    trajectory_ids: Sequence[str],
    batch_size: int,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    videos = tuple(video_ids)
    trajectories = tuple(trajectory_ids)
    return videos, trajectories


def _normalize_stream_states[StateT: (E1RuntimeState, E2RuntimeState)](
    states: Sequence[StateT | None] | None,
    batch_size: int,
) -> tuple[StateT | None, ...]:
    if states is None:
        return (None,) * batch_size
    return tuple(states)


def _runtime_tensor(tensor: Tensor, detach: bool) -> Tensor:
    return tensor.detach().clone() if detach else tensor.clone()


def _clone_e1_state(state: E1RuntimeState, *, detach: bool) -> E1RuntimeState:
    return E1RuntimeState(
        video_id=state.video_id,
        trajectory_id=state.trajectory_id,
        query_signature=state.query_signature.detach().clone(),
        projected_history=_runtime_tensor(state.projected_history, detach),
        timestamps=state.timestamps.detach().clone(),
        position_ids=state.position_ids.clone(),
        total_seen=state.total_seen,
        differentiable=not detach,
    )


def _clone_e2_state(state: E2RuntimeState, *, detach: bool) -> E2RuntimeState:
    return E2RuntimeState(
        video_id=state.video_id,
        trajectory_id=state.trajectory_id,
        query_signature=state.query_signature.detach().clone(),
        hidden=_runtime_tensor(state.hidden, detach),
        checkpoint_hidden=_runtime_tensor(state.checkpoint_hidden, detach),
        timestamps=state.timestamps.detach().clone(),
        position_ids=state.position_ids.clone(),
        total_seen=state.total_seen,
        differentiable=not detach,
    )

