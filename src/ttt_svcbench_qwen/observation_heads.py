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
    ObservationHeadsConfig,
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
    HARD_STATE_FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "current_visible_count",
        "baseline_count",
        "enter",
        "exit",
        "visible",
        "timestamp",
        "confidence",
    )

    logits: Tensor
    probabilities: Tensor
    soft_count: Tensor
    valid_mask: Tensor
    timestamps: Tensor
    position_ids: Tensor

    def __post_init__(self) -> None:
        _validate_soft_axis_output(
            self.logits,
            self.probabilities,
            self.valid_mask,
            self.timestamps,
            self.position_ids,
            last_dim=6,
            name="O1",
        )
        if (
            self.soft_count.shape != (self.logits.shape[0],)
            or not torch.is_floating_point(self.soft_count)
            or self.soft_count.dtype != self.logits.dtype
            or self.soft_count.device != self.logits.device
        ):
            raise ValueError("O1 soft_count must match logits as floating [B]")
        if self.logits.device.type != "meta":
            if not bool(torch.isfinite(self.soft_count).all()) or bool(
                torch.any(self.soft_count < 0.0)
            ):
                raise ValueError("O1 soft_count must be finite and non-negative")
            max_count = self.valid_mask.sum(dim=1).to(dtype=self.soft_count.dtype)
            if bool(torch.any(self.soft_count > max_count + 1.0e-5)):
                raise ValueError("O1 soft_count cannot exceed the valid slot count")

    @property
    def object_probability(self) -> Tensor:
        return self.probabilities[..., 0]

    @property
    def target_probability(self) -> Tensor:
        return self.probabilities[..., 1]

    @property
    def visible_probability(self) -> Tensor:
        return self.probabilities[..., 2]

    @property
    def enter_probability(self) -> Tensor:
        return self.probabilities[..., 3]

    @property
    def exit_probability(self) -> Tensor:
        return self.probabilities[..., 4]

    @property
    def confidence_probability(self) -> Tensor:
        return self.probabilities[..., 5]

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

    def __post_init__(self) -> None:
        _require_float_shape(self.identity, 256, "O2 identity")
        _validate_soft_axis_output(
            self.score_logits,
            self.score_probabilities,
            self.valid_mask,
            self.timestamps,
            self.position_ids,
            last_dim=2,
            name="O2 score",
        )
        if self.identity.shape[:2] != self.score_logits.shape[:2]:
            raise ValueError("O2 identity and score must share batch and slot dimensions")
        if self.count_prediction.numel() == 0:
            object.__setattr__(
                self,
                "count_prediction",
                (self.score_probabilities[..., 0] * self.valid_mask).sum(dim=1),
            )
        _validate_count_prediction(self.count_prediction, self.score_logits, "O2")
        if (
            self.identity.dtype != self.score_logits.dtype
            or self.identity.device != self.score_logits.device
        ):
            raise ValueError("O2 identity and score must share dtype/device")
        if self.identity.device.type != "meta":
            if not bool(torch.isfinite(self.identity).all()):
                raise ValueError("O2 identity must be finite")
            if bool(torch.any(self.identity[~self.valid_mask] != 0.0)):
                raise ValueError("invalid O2 identities must be zero")
            valid_identity = self.identity[self.valid_mask]
            if valid_identity.numel():
                norms = torch.linalg.vector_norm(valid_identity.float(), dim=-1)
                norm_tolerance = max(
                    5.0e-4,
                    2.0 * float(torch.finfo(valid_identity.dtype).eps),
                )
                if not torch.allclose(
                    norms,
                    torch.ones_like(norms),
                    atol=norm_tolerance,
                    rtol=0.0,
                ):
                    raise ValueError("valid O2 identities must have unit L2 norm")

    @property
    def score(self) -> Tensor:
        """Compatibility alias for raw novelty/match-confidence logits."""

        return self.score_logits

    @property
    def diagnostic_local_novelty_sum(self) -> Tensor:
        return (self.score_probabilities[..., 0] * self.valid_mask).sum(dim=1)


@dataclass(frozen=True, slots=True)
class StreamReplayAudit:
    head: str
    valid_counts: tuple[int, ...]
    overlap_replay_counts: tuple[int, ...]
    state_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.valid_counts)
        if (
            self.head not in {"e1", "e2"}
            or batch_size <= 0
            or len(self.overlap_replay_counts) != batch_size
            or len(self.state_lengths) != batch_size
        ):
            raise ValueError("stream audit fields must align to one E1/E2 batch")
        values = (*self.valid_counts, *self.overlap_replay_counts, *self.state_lengths)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("stream audit counters must be non-negative integers")


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

    def __post_init__(self) -> None:
        if not self.video_id or not self.trajectory_id:
            raise ValueError("E1 runtime owner identifiers must be non-empty")
        if (
            self.projected_history.ndim != 2
            or self.projected_history.shape[1] != 512
            or self.projected_history.shape[0] > 66
            or not torch.is_floating_point(self.projected_history)
        ):
            raise ValueError("E1 projected_history must be floating [L<=66, 512]")
        length = int(self.projected_history.shape[0])
        if (
            self.query_signature.shape != (512,)
            or not torch.is_floating_point(self.query_signature)
            or self.query_signature.dtype != self.projected_history.dtype
            or self.query_signature.device != self.projected_history.device
        ):
            raise ValueError("E1 query_signature must match history as floating [512]")
        if (
            self.timestamps.shape != (length,)
            or self.timestamps.dtype != torch.float64
            or self.timestamps.device != self.projected_history.device
        ):
            raise ValueError("E1 runtime timestamps must be float64 [L]")
        if (
            self.position_ids.shape != (length,)
            or self.position_ids.dtype != torch.int64
            or self.position_ids.device != self.projected_history.device
        ):
            raise ValueError("E1 runtime position_ids must be int64 [L]")
        if type(self.total_seen) is not int or self.total_seen < 0:
            raise ValueError("E1 runtime total_seen must be a non-negative integer")
        if type(self.differentiable) is not bool:
            raise TypeError("E1 runtime differentiable must be a bool")
        if not self.differentiable and (
            self.projected_history.requires_grad
            or self.query_signature.requires_grad
            or self.timestamps.requires_grad
        ):
            raise ValueError("non-differentiable E1 runtime tensors must be detached")
        if self.projected_history.device.type == "meta":
            return
        if not bool(torch.isfinite(self.query_signature).all()) or not bool(
            torch.isfinite(self.projected_history).all()
        ):
            raise ValueError("E1 runtime floating tensors must be finite")
        expected_length = min(self.total_seen, 66)
        if length != expected_length:
            raise ValueError("E1 runtime history length must match total_seen and capacity")
        if length:
            if (
                not bool(torch.isfinite(self.timestamps).all())
                or bool(torch.any(self.timestamps < 0.0))
                or int(self.position_ids[-1].item()) + 1 != self.total_seen
                or int(self.position_ids[0].item()) != self.total_seen - length
            ):
                raise ValueError("E1 runtime metadata must be finite contiguous history")
            if length > 1 and (
                bool(torch.any(self.timestamps[1:] <= self.timestamps[:-1]))
                or bool(torch.any(self.position_ids[1:] != self.position_ids[:-1] + 1))
            ):
                raise ValueError("E1 runtime metadata must increase strictly")
        if _shares_any_storage(
            (
                self.query_signature,
                self.projected_history,
                self.timestamps,
                self.position_ids,
            )
        ):
            raise ValueError("E1 runtime fields must use independent storage")


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
    audit: StreamReplayAudit
    count_prediction: Tensor = field(default_factory=lambda: torch.empty(0))

    def __post_init__(self) -> None:
        _validate_soft_axis_output(
            self.logits,
            self.probabilities,
            self.valid_mask,
            self.timestamps,
            self.position_ids,
            last_dim=3,
            name="E1",
        )
        if len(self.next_states) != self.logits.shape[0] or self.audit.head != "e1":
            raise ValueError("E1 output requires one next state and an E1 audit per batch row")
        if self.count_prediction.numel() == 0:
            object.__setattr__(
                self,
                "count_prediction",
                (self.probabilities[..., 1] * self.valid_mask).sum(dim=1),
            )
        _validate_count_prediction(self.count_prediction, self.logits, "E1")
        _assert_e1_state_storage_isolated(self.next_states)

    @property
    def diagnostic_local_completion_sum(self) -> Tensor:
        return (self.probabilities[..., 1] * self.valid_mask).sum(dim=1)


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

    def __post_init__(self) -> None:
        if not self.video_id or not self.trajectory_id:
            raise ValueError("E2 runtime owner identifiers must be non-empty")
        if self.hidden.shape != (2, 768) or not torch.is_floating_point(self.hidden):
            raise ValueError("E2 hidden must be floating [2, 768]")
        if (
            self.checkpoint_hidden.ndim != 3
            or self.checkpoint_hidden.shape[1:] != (2, 768)
            or self.checkpoint_hidden.shape[0] > 5
            or not torch.is_floating_point(self.checkpoint_hidden)
            or self.checkpoint_hidden.dtype != self.hidden.dtype
            or self.checkpoint_hidden.device != self.hidden.device
        ):
            raise ValueError("E2 checkpoint_hidden must match hidden as [L<=5, 2, 768]")
        length = int(self.checkpoint_hidden.shape[0])
        if (
            self.query_signature.shape != (512,)
            or not torch.is_floating_point(self.query_signature)
            or self.query_signature.dtype != self.hidden.dtype
            or self.query_signature.device != self.hidden.device
        ):
            raise ValueError("E2 query_signature must match hidden as floating [512]")
        if (
            self.timestamps.shape != (length,)
            or self.timestamps.dtype != torch.float64
            or self.timestamps.device != self.hidden.device
        ):
            raise ValueError("E2 runtime timestamps must be float64 [L]")
        if (
            self.position_ids.shape != (length,)
            or self.position_ids.dtype != torch.int64
            or self.position_ids.device != self.hidden.device
        ):
            raise ValueError("E2 runtime position_ids must be int64 [L]")
        if type(self.total_seen) is not int or self.total_seen < 0:
            raise ValueError("E2 runtime total_seen must be a non-negative integer")
        if type(self.differentiable) is not bool:
            raise TypeError("E2 runtime differentiable must be a bool")
        if not self.differentiable and (
            self.hidden.requires_grad
            or self.checkpoint_hidden.requires_grad
            or self.query_signature.requires_grad
            or self.timestamps.requires_grad
        ):
            raise ValueError("non-differentiable E2 runtime tensors must be detached")
        if self.hidden.device.type == "meta":
            return
        if (
            not bool(torch.isfinite(self.hidden).all())
            or not bool(torch.isfinite(self.checkpoint_hidden).all())
            or not bool(torch.isfinite(self.query_signature).all())
        ):
            raise ValueError("E2 runtime floating tensors must be finite")
        expected_length = min(self.total_seen, 5)
        if length != expected_length:
            raise ValueError("E2 checkpoint length must match total_seen and capacity")
        if length:
            if (
                not bool(torch.isfinite(self.timestamps).all())
                or bool(torch.any(self.timestamps < 0.0))
                or int(self.position_ids[-1].item()) + 1 != self.total_seen
                or int(self.position_ids[0].item()) != self.total_seen - length
                or not torch.equal(self.hidden, self.checkpoint_hidden[-1])
            ):
                raise ValueError("E2 runtime checkpoint metadata/hidden is inconsistent")
            if length > 1 and (
                bool(torch.any(self.timestamps[1:] <= self.timestamps[:-1]))
                or bool(torch.any(self.position_ids[1:] != self.position_ids[:-1] + 1))
            ):
                raise ValueError("E2 runtime metadata must increase strictly")
        elif bool(torch.any(self.hidden != 0.0)):
            raise ValueError("a fresh E2 runtime must have zero hidden state")
        if _shares_any_storage(
            (
                self.query_signature,
                self.hidden,
                self.checkpoint_hidden,
                self.timestamps,
                self.position_ids,
            )
        ):
            raise ValueError("E2 runtime fields must use independent storage")


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
    audit: StreamReplayAudit
    count_prediction: Tensor = field(default_factory=lambda: torch.empty(0))

    def __post_init__(self) -> None:
        _validate_soft_axis_output(
            self.event_logits,
            self.event_probabilities,
            self.valid_mask,
            self.timestamps,
            self.position_ids,
            last_dim=4,
            name="E2 event",
        )
        _validate_soft_axis_output(
            self.phase_logits,
            self.phase_probabilities,
            self.valid_mask,
            self.timestamps,
            self.position_ids,
            last_dim=4,
            name="E2 phase",
        )
        if self.event_logits.shape != self.phase_logits.shape:
            raise ValueError("E2 event and phase logits must have identical shapes")
        if len(self.next_states) != self.event_logits.shape[0] or self.audit.head != "e2":
            raise ValueError("E2 output requires one next state and an E2 audit per batch row")
        if self.count_prediction.numel() == 0:
            object.__setattr__(
                self,
                "count_prediction",
                (self.event_probabilities[..., 3] * self.valid_mask).sum(dim=1),
            )
        _validate_count_prediction(self.count_prediction, self.event_logits, "E2")
        if self.event_logits.device.type != "meta":
            valid_phase = self.phase_probabilities[self.valid_mask]
            sum_tolerance = max(
                5.0e-4,
                2.0 * float(torch.finfo(self.phase_probabilities.dtype).eps),
            )
            if valid_phase.numel() and not torch.allclose(
                valid_phase.float().sum(dim=-1),
                torch.ones(valid_phase.shape[0], device=valid_phase.device),
                atol=sum_tolerance,
                rtol=0.0,
            ):
                raise ValueError("valid E2 phase probabilities must sum to one")
        _assert_e2_state_storage_isolated(self.next_states)

    @property
    def diagnostic_local_completion_sum(self) -> Tensor:
        return (self.event_probabilities[..., 3] * self.valid_mask).sum(dim=1)


@dataclass(frozen=True, slots=True)
class ObservationOutputs:
    o1: O1SoftOutput
    o2: O2SoftOutput
    e1: E1SoftOutput
    e2: E2SoftOutput

    def __post_init__(self) -> None:
        if (
            self.o1.valid_mask.shape != self.o2.valid_mask.shape
            or not torch.equal(self.o1.valid_mask, self.o2.valid_mask)
            or not torch.equal(self.o1.timestamps, self.o2.timestamps)
            or not torch.equal(self.o1.position_ids, self.o2.position_ids)
        ):
            raise ValueError("O1/O2 outputs must share slot mask and metadata")
        if (
            self.e1.valid_mask.shape != self.e2.valid_mask.shape
            or not torch.equal(self.e1.valid_mask, self.e2.valid_mask)
            or not torch.equal(self.e1.timestamps, self.e2.timestamps)
            or not torch.equal(self.e1.position_ids, self.e2.position_ids)
        ):
            raise ValueError("E1/E2 outputs must share tubelet mask and metadata")
        if self.o1.logits.shape[0] != self.e1.logits.shape[0]:
            raise ValueError("all observation heads must share one batch size")


class CumulativeCountHead(nn.Module):  # type: ignore[misc]
    """Unbounded positive cumulative count independent of the local sequence length."""

    def __init__(self, input_dim: int, *, layer_norm_eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.hidden = nn.Linear(input_dim, 256, bias=True)
        self.output = nn.Linear(256, 1, bias=True)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("cumulative count features must be [B, D]")
        logits = self.output(F.silu(self.hidden(self.norm(features)))).squeeze(-1)
        return F.softplus(logits.float()).to(dtype=logits.dtype)


class O1CurrentCountDecoder(nn.Module):  # type: ignore[misc]
    def __init__(self, config: O1Config) -> None:
        super().__init__()
        _validate_o1_config(config)
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
        safe_slots, expanded_timestamps, expanded_positions = _validate_spatial_head_inputs(
            self,
            slots,
            slot_valid_mask,
            observation_timestamps,
            observation_position_ids,
            name="O1",
        )
        _validate_query(q_target, slots, self.config.query_dim, "O1")
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
        _validate_o2_config(config)
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

    def forward(
        self,
        slots: Tensor,
        slot_valid_mask: Tensor,
        observation_timestamps: Tensor,
        observation_position_ids: Tensor,
        *,
        q_target: Tensor | None = None,
    ) -> O2SoftOutput:
        safe_slots, expanded_timestamps, expanded_positions = _validate_spatial_head_inputs(
            self,
            slots,
            slot_valid_mask,
            observation_timestamps,
            observation_position_ids,
            name="O2",
        )
        hidden = F.silu(self.trunk_1(self.slot_norm(safe_slots)))
        hidden = F.silu(self.trunk_2(hidden))
        if q_target is None:
            q_target = hidden.new_zeros((hidden.shape[0], 512))
        _validate_query(q_target, hidden, 512, "O2")
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
        return O2SoftOutput(
            identity=identity,
            score_logits=score_logits,
            score_probabilities=score_probabilities,
            valid_mask=slot_valid_mask.clone(),
            timestamps=expanded_timestamps,
            position_ids=expanded_positions,
            count_prediction=count_prediction,
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
        _validate_e1_config(config)
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
        if type(detach_runtime_state) is not bool:
            raise TypeError("detach_runtime_state must be a bool")
        safe_hidden, normalized_timestamps = _validate_temporal_head_inputs(
            self,
            hidden,
            valid_mask,
            timestamps,
            position_ids,
            name="E1",
        )
        owners = _normalize_stream_owners(video_ids, trajectory_ids, hidden.shape[0], "E1")
        _validate_query_signatures(query_signatures, hidden, owners[0], "E1")
        states = _normalize_e1_states(prior_states, hidden.shape[0])
        _validate_e1_prior_states(states, owners, query_signatures, hidden)
        projected = self.input_projection(self.input_norm(safe_hidden))
        projected = torch.where(valid_mask.unsqueeze(-1), projected, 0.0)
        output_rows: list[Tensor] = []
        next_states: list[E1RuntimeState] = []
        valid_counts: list[int] = []
        overlap_counts: list[int] = []
        state_lengths: list[int] = []
        count_features: list[Tensor] = []
        for row in range(hidden.shape[0]):
            state = states[row] or self._empty_state(
                owners[0][row],
                owners[1][row],
                query_signatures[row],
            )
            count = int(valid_mask[row].sum().item())
            valid_counts.append(count)
            if count == 0:
                next_state = _clone_e1_state(state, detach=detach_runtime_state)
                output_rows.append(hidden.new_zeros((1, hidden.shape[1], 3)))
                next_states.append(next_state)
                overlap_counts.append(0)
                state_lengths.append(int(next_state.projected_history.shape[0]))
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
            overlap_counts.append(overlap_count)
            state_lengths.append(int(next_state.projected_history.shape[0]))
            count_features.append(encoded[0, -1])
        logits = torch.cat(output_rows, dim=0)
        logits = torch.where(valid_mask.unsqueeze(-1), logits, 0.0)
        probabilities = torch.sigmoid(logits.float()).to(dtype=logits.dtype)
        probabilities = torch.where(valid_mask.unsqueeze(-1), probabilities, 0.0)
        audit = StreamReplayAudit(
            head="e1",
            valid_counts=tuple(valid_counts),
            overlap_replay_counts=tuple(overlap_counts),
            state_lengths=tuple(state_lengths),
        )
        return E1SoftOutput(
            logits=logits,
            probabilities=probabilities,
            valid_mask=valid_mask.clone(),
            timestamps=normalized_timestamps,
            position_ids=position_ids.clone(),
            next_states=tuple(next_states),
            audit=audit,
            count_prediction=self.count_head(torch.stack(count_features, dim=0)),
        )

    def reset_state(
        self,
        video_id: str,
        trajectory_id: str,
        query_signature: Tensor,
    ) -> E1RuntimeState:
        _validate_reset_signature(self, video_id, trajectory_id, query_signature, "E1")
        return self._empty_state(video_id, trajectory_id, query_signature)

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
            if int(current_positions[0].item()) != 0:
                raise ValueError("a fresh E1 trajectory must start at position zero")
            return (
                state.projected_history,
                state.timestamps,
                state.position_ids,
                0,
            )
        first = int(current_positions[0].item())
        last = int(current_positions[-1].item())
        cached_first = int(state.position_ids[0].item())
        cached_last = int(state.position_ids[-1].item())
        retain_count = int(state.position_ids.shape[0])
        overlap_count = 0
        if first <= cached_last:
            required_first = max(0, first - (self.config.receptive_field - 1))
            if first < cached_first or cached_first > required_first:
                raise ValueError("E1 overlap rewind lacks its complete causal history")
            if last < cached_last:
                raise ValueError("E1 overlap cannot discard an unobserved cached suffix")
            retain_count = first - cached_first
            overlap_count = cached_last - first + 1
            if overlap_count > self.config.overlap_tubelets:
                raise ValueError("E1 overlap exceeds the configured replay window")
            cached_overlap = state.timestamps[retain_count : retain_count + overlap_count]
            if not _timestamps_match(current_timestamps[:overlap_count], cached_overlap):
                raise ValueError("E1 overlap timestamps must match cached positions")
        elif first != cached_last + 1:
            raise ValueError("E1 position IDs cannot contain gaps")
        if retain_count and not bool(current_timestamps[0] > state.timestamps[retain_count - 1]):
            raise ValueError("E1 current timestamps must follow retained history")
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
        _validate_e2_config(config)
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
        if type(detach_runtime_state) is not bool:
            raise TypeError("detach_runtime_state must be a bool")
        safe_hidden, normalized_timestamps = _validate_temporal_head_inputs(
            self,
            hidden,
            valid_mask,
            timestamps,
            position_ids,
            name="E2",
        )
        owners = _normalize_stream_owners(video_ids, trajectory_ids, hidden.shape[0], "E2")
        _validate_query_signatures(query_signatures, hidden, owners[0], "E2")
        states = _normalize_e2_states(prior_states, hidden.shape[0])
        _validate_e2_prior_states(states, owners, query_signatures, hidden)
        normalized = self.input_norm(safe_hidden)
        event_rows: list[Tensor] = []
        phase_rows: list[Tensor] = []
        next_states: list[E2RuntimeState] = []
        valid_counts: list[int] = []
        overlap_counts: list[int] = []
        state_lengths: list[int] = []
        count_features: list[Tensor] = []
        for row in range(hidden.shape[0]):
            state = states[row] or self._empty_state(
                owners[0][row],
                owners[1][row],
                query_signatures[row],
            )
            count = int(valid_mask[row].sum().item())
            valid_counts.append(count)
            if count == 0:
                next_state = _clone_e2_state(state, detach=detach_runtime_state)
                event_rows.append(hidden.new_zeros((1, hidden.shape[1], 4)))
                phase_rows.append(hidden.new_zeros((1, hidden.shape[1], 4)))
                next_states.append(next_state)
                overlap_counts.append(0)
                state_lengths.append(int(next_state.checkpoint_hidden.shape[0]))
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
            overlap_counts.append(overlap)
            state_lengths.append(int(next_state.checkpoint_hidden.shape[0]))
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
        audit = StreamReplayAudit(
            head="e2",
            valid_counts=tuple(valid_counts),
            overlap_replay_counts=tuple(overlap_counts),
            state_lengths=tuple(state_lengths),
        )
        return E2SoftOutput(
            event_logits=event_logits,
            phase_logits=phase_logits,
            event_probabilities=event_probabilities,
            phase_probabilities=phase_probabilities,
            valid_mask=valid_mask.clone(),
            timestamps=normalized_timestamps,
            position_ids=position_ids.clone(),
            next_states=tuple(next_states),
            audit=audit,
            count_prediction=self.count_head(torch.stack(count_features, dim=0)),
        )

    def reset_state(
        self,
        video_id: str,
        trajectory_id: str,
        query_signature: Tensor,
    ) -> E2RuntimeState:
        _validate_reset_signature(self, video_id, trajectory_id, query_signature, "E2")
        return self._empty_state(video_id, trajectory_id, query_signature)

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
            if int(current_positions[0].item()) != 0:
                raise ValueError("a fresh E2 trajectory must start at position zero")
            return (
                state.hidden,
                state.checkpoint_hidden,
                state.timestamps,
                state.position_ids,
                0,
            )
        first = int(current_positions[0].item())
        last = int(current_positions[-1].item())
        cached_first = int(state.position_ids[0].item())
        cached_last = int(state.position_ids[-1].item())
        retain_count = int(state.position_ids.shape[0])
        overlap_count = 0
        initial_hidden = state.hidden
        if first <= cached_last:
            predecessor = first - 1
            if first < cached_first + 1 or last < cached_last:
                raise ValueError("E2 overlap rewind exceeds its five-checkpoint window")
            predecessor_index = predecessor - cached_first
            retain_count = predecessor_index + 1
            initial_hidden = state.checkpoint_hidden[predecessor_index]
            overlap_count = cached_last - first + 1
            cached_overlap = state.timestamps[retain_count : retain_count + overlap_count]
            if not _timestamps_match(current_timestamps[:overlap_count], cached_overlap):
                raise ValueError("E2 overlap timestamps must match cached positions")
        elif first != cached_last + 1:
            raise ValueError("E2 position IDs cannot contain gaps")
        if retain_count and not bool(current_timestamps[0] > state.timestamps[retain_count - 1]):
            raise ValueError("E2 current timestamps must follow retained checkpoints")
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
        _validate_observation_heads_config(self.config)
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
        if not isinstance(spatial, SpatialEncoderOutput) or not isinstance(
            temporal, TemporalEncoderOutput
        ):
            raise TypeError("ObservationHeads requires typed spatial and temporal outputs")
        batch_size = spatial.slots.shape[0]
        if temporal.hidden.shape[0] != batch_size:
            raise ValueError("spatial and temporal outputs must share batch size")
        owners = _normalize_stream_owners(
            video_ids,
            trajectory_ids,
            batch_size,
            "ObservationHeads",
        )
        if owners[0] != temporal.cache.video_ids or owners[1] != temporal.cache.trajectory_ids:
            raise ValueError("observation owners must exactly match temporal cache owners")
        _validate_query(q_target, spatial.slots, 512, "ObservationHeads")
        if temporal.cache.query_signatures.shape != q_target.shape or not torch.equal(
            temporal.cache.query_signatures, q_target.detach()
        ):
            raise ValueError("observation query must match temporal cache query signature")
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

        if type(frozen) is not bool:
            raise TypeError("online frozen flag must be a bool")
        for parameter in self.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.eval()
        return self

    @property
    def online_frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.parameters())


def build_observation_heads(config: ProjectConfig | None = None) -> ObservationHeads:
    if config is None:
        raise ValueError("build_observation_heads requires a validated ProjectConfig")
    return ObservationHeads(config)


def observation_head_parameter_counts(module: ObservationHeads) -> dict[str, int]:
    return {
        name: sum(parameter.numel() for parameter in getattr(module, name).parameters())
        for name in ("o1", "o2", "e1", "e2")
    }


def observation_heads_parameter_count(module: ObservationHeads) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _require_float_shape(tensor: Tensor, last_dim: int, name: str) -> None:
    if tensor.ndim != 3 or tensor.shape[-1] != last_dim or not torch.is_floating_point(tensor):
        raise ValueError(f"{name} must be floating [B, N, {last_dim}]")


def _validate_soft_axis_output(
    logits: Tensor,
    probabilities: Tensor,
    valid_mask: Tensor,
    timestamps: Tensor,
    position_ids: Tensor,
    *,
    last_dim: int,
    name: str,
) -> None:
    _require_float_shape(logits, last_dim, f"{name} logits")
    if (
        probabilities.shape != logits.shape
        or not torch.is_floating_point(probabilities)
        or probabilities.dtype != logits.dtype
        or probabilities.device != logits.device
    ):
        raise ValueError(f"{name} probabilities must match logits")
    shape = logits.shape[:2]
    if (
        valid_mask.shape != shape
        or valid_mask.dtype != torch.bool
        or valid_mask.device != logits.device
    ):
        raise ValueError(f"{name} valid_mask must be bool [B, N]")
    if (
        timestamps.shape != shape
        or timestamps.dtype not in (torch.float32, torch.float64)
        or timestamps.device != logits.device
    ):
        raise ValueError(f"{name} timestamps must be FP32/FP64 [B, N]")
    if (
        position_ids.shape != shape
        or position_ids.dtype != torch.int64
        or position_ids.device != logits.device
    ):
        raise ValueError(f"{name} position_ids must be int64 [B, N]")
    if logits.device.type == "meta":
        return
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(probabilities).all()):
        raise ValueError(f"{name} outputs must be finite")
    if bool(torch.any((probabilities < 0.0) | (probabilities > 1.0))):
        raise ValueError(f"{name} probabilities must stay within [0, 1]")
    if bool(torch.any(logits[~valid_mask] != 0.0)) or bool(
        torch.any(probabilities[~valid_mask] != 0.0)
    ):
        raise ValueError(f"invalid {name} outputs must be zero")
    if bool(torch.any(timestamps[~valid_mask] != -1.0)):
        raise ValueError(f"invalid {name} timestamps must use -1")
    if bool(torch.any(position_ids[~valid_mask] != -1)):
        raise ValueError(f"invalid {name} position IDs must use -1")
    valid_times = timestamps[valid_mask]
    if valid_times.numel() and (
        not bool(torch.isfinite(valid_times).all()) or bool(torch.any(valid_times < 0.0))
    ):
        raise ValueError(f"valid {name} timestamps must be finite and non-negative")
    if bool(torch.any(position_ids[valid_mask] < 0)):
        raise ValueError(f"valid {name} position IDs must be non-negative")


def _validate_spatial_head_inputs(
    module: nn.Module,
    slots: Tensor,
    valid_mask: Tensor,
    observation_timestamps: Tensor,
    observation_position_ids: Tensor,
    *,
    name: str,
) -> tuple[Tensor, Tensor, Tensor]:
    _require_float_shape(slots, 768, f"{name} slots")
    batch_size, slot_count = slots.shape[:2]
    if (
        valid_mask.shape != (batch_size, slot_count)
        or valid_mask.dtype != torch.bool
        or valid_mask.device != slots.device
    ):
        raise ValueError(f"{name} slot_valid_mask must be bool [B, K]")
    if (
        observation_timestamps.shape != (batch_size,)
        or observation_timestamps.dtype not in (torch.float32, torch.float64)
        or observation_timestamps.device != slots.device
    ):
        raise ValueError(f"{name} observation_timestamps must be FP32/FP64 [B]")
    if (
        observation_position_ids.shape != (batch_size,)
        or observation_position_ids.dtype != torch.int64
        or observation_position_ids.device != slots.device
    ):
        raise ValueError(f"{name} observation_position_ids must be int64 [B]")
    _validate_module_dtype_device(module, slots, name)
    expanded = observation_timestamps.unsqueeze(1).expand(-1, slot_count)
    expanded = torch.where(valid_mask, expanded, torch.full_like(expanded, -1.0))
    expanded_positions = observation_position_ids.unsqueeze(1).expand(-1, slot_count)
    expanded_positions = torch.where(
        valid_mask,
        expanded_positions,
        torch.full_like(expanded_positions, -1),
    )
    safe_slots = torch.where(valid_mask.unsqueeze(-1), slots, 0.0)
    if slots.device.type != "meta":
        if not bool(torch.isfinite(safe_slots).all()):
            raise ValueError(f"valid {name} slots must be finite")
        valid_times = expanded[valid_mask]
        if valid_times.numel() and (
            not bool(torch.isfinite(valid_times).all()) or bool(torch.any(valid_times < 0.0))
        ):
            raise ValueError(f"valid {name} observation timestamps must be legal")
        if bool(torch.any(expanded_positions[valid_mask] < 0)):
            raise ValueError(f"valid {name} observation position IDs must be legal")
    return safe_slots, expanded.clone(), expanded_positions.clone()


def _validate_temporal_head_inputs(
    module: nn.Module,
    hidden: Tensor,
    valid_mask: Tensor,
    timestamps: Tensor,
    position_ids: Tensor,
    *,
    name: str,
) -> tuple[Tensor, Tensor]:
    _require_float_shape(hidden, 768, f"{name} hidden")
    batch_size, time_count = hidden.shape[:2]
    if (
        valid_mask.shape != (batch_size, time_count)
        or valid_mask.dtype != torch.bool
        or valid_mask.device != hidden.device
    ):
        raise ValueError(f"{name} valid_mask must be bool [B, T]")
    if (
        time_count > 1
        and hidden.device.type != "meta"
        and bool(torch.any(valid_mask[:, 1:] & ~valid_mask[:, :-1]))
    ):
        raise ValueError(f"{name} valid_mask must be a valid prefix")
    if (
        timestamps.shape != (batch_size, time_count)
        or timestamps.dtype not in (torch.float32, torch.float64)
        or timestamps.device != hidden.device
    ):
        raise ValueError(f"{name} timestamps must be FP32/FP64 [B, T]")
    if (
        position_ids.shape != (batch_size, time_count)
        or position_ids.dtype != torch.int64
        or position_ids.device != hidden.device
    ):
        raise ValueError(f"{name} position_ids must be int64 [B, T]")
    _validate_module_dtype_device(module, hidden, name)
    safe_hidden = torch.where(valid_mask.unsqueeze(-1), hidden, 0.0)
    normalized_timestamps = torch.where(
        valid_mask,
        timestamps.to(dtype=torch.float64),
        torch.full_like(timestamps, -1.0, dtype=torch.float64),
    )
    if hidden.device.type != "meta":
        if not bool(torch.isfinite(safe_hidden).all()):
            raise ValueError(f"valid {name} hidden states must be finite")
        if bool(torch.any(timestamps[~valid_mask] != -1.0)) or bool(
            torch.any(position_ids[~valid_mask] != -1)
        ):
            raise ValueError(f"invalid {name} temporal metadata must use -1")
        for row in range(batch_size):
            count = int(valid_mask[row].sum().item())
            if not count:
                continue
            valid_times = normalized_timestamps[row, :count]
            valid_positions = position_ids[row, :count]
            if (
                not bool(torch.isfinite(valid_times).all())
                or bool(torch.any(valid_times < 0.0))
                or bool(torch.any(valid_positions < 0))
            ):
                raise ValueError(f"valid {name} temporal metadata must be legal")
            if count > 1 and (
                bool(torch.any(valid_times[1:] <= valid_times[:-1]))
                or bool(torch.any(valid_positions[1:] != valid_positions[:-1] + 1))
            ):
                raise ValueError(f"valid {name} temporal metadata must increase strictly")
    return safe_hidden, normalized_timestamps


def _validate_module_dtype_device(module: nn.Module, inputs: Tensor, name: str) -> None:
    for parameter in module.parameters():
        if parameter.dtype != inputs.dtype or parameter.device != inputs.device:
            raise ValueError(f"{name} module and inputs must share dtype/device")


def _validate_count_prediction(prediction: Tensor, reference: Tensor, name: str) -> None:
    if (
        prediction.shape != (reference.shape[0],)
        or not torch.is_floating_point(prediction)
        or prediction.dtype != reference.dtype
        or prediction.device != reference.device
    ):
        raise ValueError(f"{name} count_prediction must match logits as floating [B]")
    if prediction.device.type != "meta" and (
        not bool(torch.isfinite(prediction).all()) or bool(torch.any(prediction < 0.0))
    ):
        raise ValueError(f"{name} count_prediction must be finite and non-negative")


def _validate_query(query: Tensor, reference: Tensor, query_dim: int, name: str) -> None:
    if (
        query.shape != (reference.shape[0], query_dim)
        or not torch.is_floating_point(query)
        or query.dtype != reference.dtype
        or query.device != reference.device
    ):
        raise ValueError(f"{name} q_target must match inputs as floating [B, {query_dim}]")
    if query.device.type != "meta" and not bool(torch.isfinite(query).all()):
        raise ValueError(f"{name} q_target must be finite")


def _normalize_stream_owners(
    video_ids: Sequence[str],
    trajectory_ids: Sequence[str],
    batch_size: int,
    name: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    videos = tuple(video_ids)
    trajectories = tuple(trajectory_ids)
    if (
        len(videos) != batch_size
        or any(not value for value in videos)
        or len(set(videos)) != batch_size
    ):
        raise ValueError(f"{name} requires unique non-empty video_ids")
    if len(trajectories) != batch_size or any(not value for value in trajectories):
        raise ValueError(f"{name} requires one non-empty trajectory_id per row")
    return videos, trajectories


def _validate_query_signatures(
    signatures: Tensor,
    reference: Tensor,
    video_ids: tuple[str, ...],
    name: str,
) -> None:
    if (
        signatures.shape != (len(video_ids), 512)
        or not torch.is_floating_point(signatures)
        or signatures.dtype != reference.dtype
        or signatures.device != reference.device
    ):
        raise ValueError(f"{name} query_signatures must match inputs as floating [B, 512]")
    if signatures.device.type != "meta" and not bool(torch.isfinite(signatures).all()):
        raise ValueError(f"{name} query_signatures must be finite")


def _validate_reset_signature(
    module: nn.Module,
    video_id: str,
    trajectory_id: str,
    query_signature: Tensor,
    name: str,
) -> None:
    if not video_id or not trajectory_id:
        raise ValueError(f"{name} reset requires non-empty owner identifiers")
    if query_signature.shape != (512,) or not torch.is_floating_point(query_signature):
        raise ValueError(f"{name} reset query signature must be floating [512]")
    parameter = next(module.parameters())
    if query_signature.dtype != parameter.dtype or query_signature.device != parameter.device:
        raise ValueError(f"{name} reset signature must share module dtype/device")
    if query_signature.device.type != "meta" and not bool(torch.isfinite(query_signature).all()):
        raise ValueError(f"{name} reset signature must be finite")


def _normalize_e1_states(
    states: Sequence[E1RuntimeState | None] | None,
    batch_size: int,
) -> tuple[E1RuntimeState | None, ...]:
    if states is None:
        return (None,) * batch_size
    normalized = tuple(states)
    if len(normalized) != batch_size or any(
        state is not None and not isinstance(state, E1RuntimeState) for state in normalized
    ):
        raise ValueError("E1 requires one E1RuntimeState or None per batch row")
    return normalized


def _normalize_e2_states(
    states: Sequence[E2RuntimeState | None] | None,
    batch_size: int,
) -> tuple[E2RuntimeState | None, ...]:
    if states is None:
        return (None,) * batch_size
    normalized = tuple(states)
    if len(normalized) != batch_size or any(
        state is not None and not isinstance(state, E2RuntimeState) for state in normalized
    ):
        raise ValueError("E2 requires one E2RuntimeState or None per batch row")
    return normalized


def _validate_e1_prior_states(
    states: tuple[E1RuntimeState | None, ...],
    owners: tuple[tuple[str, ...], tuple[str, ...]],
    signatures: Tensor,
    reference: Tensor,
) -> None:
    for row, state in enumerate(states):
        if state is None:
            continue
        if state.video_id != owners[0][row] or state.trajectory_id != owners[1][row]:
            raise ValueError("E1 runtime owner must match its exact batch row")
        if (
            state.projected_history.dtype != reference.dtype
            or state.projected_history.device != reference.device
        ):
            raise ValueError("E1 runtime and inputs must share dtype/device")
        if not torch.equal(state.query_signature, signatures[row].detach()):
            raise ValueError("E1 runtime query signature drift requires reset")
    _assert_optional_e1_state_storage_isolated(states)


def _validate_e2_prior_states(
    states: tuple[E2RuntimeState | None, ...],
    owners: tuple[tuple[str, ...], tuple[str, ...]],
    signatures: Tensor,
    reference: Tensor,
) -> None:
    for row, state in enumerate(states):
        if state is None:
            continue
        if state.video_id != owners[0][row] or state.trajectory_id != owners[1][row]:
            raise ValueError("E2 runtime owner must match its exact batch row")
        if state.hidden.dtype != reference.dtype or state.hidden.device != reference.device:
            raise ValueError("E2 runtime and inputs must share dtype/device")
        if not torch.equal(state.query_signature, signatures[row].detach()):
            raise ValueError("E2 runtime query signature drift requires reset")
    _assert_optional_e2_state_storage_isolated(states)


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


def _timestamps_match(left: Tensor, right: Tensor) -> bool:
    if left.shape != right.shape:
        return False
    left_64 = left.to(dtype=torch.float64)
    right_64 = right.to(dtype=torch.float64)
    scale = torch.maximum(left_64.abs(), right_64.abs()).clamp_min(1.0)
    tolerance = 4.0 * torch.finfo(torch.float32).eps * scale
    return bool(torch.all((left_64 - right_64).abs() <= tolerance))


def _shares_storage(left: Tensor, right: Tensor) -> bool:
    if left.numel() == 0 or right.numel() == 0:
        return False
    if left.device.type == "meta" or right.device.type == "meta":
        return left is right
    return int(left.untyped_storage().data_ptr()) == int(right.untyped_storage().data_ptr())


def _shares_any_storage(tensors: Sequence[Tensor]) -> bool:
    for index, left in enumerate(tensors):
        for right in tensors[index + 1 :]:
            if _shares_storage(left, right):
                return True
    return False


def _e1_state_tensors(state: E1RuntimeState) -> tuple[Tensor, ...]:
    return (
        state.query_signature,
        state.projected_history,
        state.timestamps,
        state.position_ids,
    )


def _e2_state_tensors(state: E2RuntimeState) -> tuple[Tensor, ...]:
    return (
        state.query_signature,
        state.hidden,
        state.checkpoint_hidden,
        state.timestamps,
        state.position_ids,
    )


def _assert_e1_state_storage_isolated(states: Sequence[E1RuntimeState]) -> None:
    _assert_state_storage_isolated(tuple(_e1_state_tensors(state) for state in states), "E1")


def _assert_e2_state_storage_isolated(states: Sequence[E2RuntimeState]) -> None:
    _assert_state_storage_isolated(tuple(_e2_state_tensors(state) for state in states), "E2")


def _assert_optional_e1_state_storage_isolated(
    states: Sequence[E1RuntimeState | None],
) -> None:
    _assert_e1_state_storage_isolated(tuple(state for state in states if state is not None))


def _assert_optional_e2_state_storage_isolated(
    states: Sequence[E2RuntimeState | None],
) -> None:
    _assert_e2_state_storage_isolated(tuple(state for state in states if state is not None))


def _assert_state_storage_isolated(state_tensors: Sequence[tuple[Tensor, ...]], name: str) -> None:
    for left_index, left_group in enumerate(state_tensors):
        for right_group in state_tensors[left_index + 1 :]:
            if any(_shares_storage(left, right) for left in left_group for right in right_group):
                raise ValueError(f"{name} runtime batch rows must not share mutable storage")


def _validate_observation_heads_config(config: ObservationHeadsConfig) -> None:
    expected: dict[str, object] = {
        "temporal_input_conditioning": "inherited_query_conditioned_h_t",
        "raw_logits": True,
        "debug_probabilities": True,
        "output_valid_mask": True,
        "output_timestamps": True,
        "output_position_ids": True,
        "invalid_output_policy": "zero_tensors_negative_one_metadata",
        "online_frozen": True,
        "online_forward_no_grad": False,
        "detach_inputs": False,
        "hard_state_mutation": False,
    }
    _validate_config_fields(config, expected, "Observation Heads")


def _validate_o1_config(config: O1Config) -> None:
    expected: dict[str, object] = {
        "input_dim": 768,
        "query_dim": 512,
        "film_dim": 1536,
        "hidden_dims": (1024, 1024),
        "output_dim": 6,
        "layer_norm_eps": 1.0e-5,
        "activation": "silu",
        "film_mode": "one_plus_scale_and_shift",
        "output_names": O1SoftOutput.LOGIT_NAMES,
        "dropout": 0.0,
        "linear_bias": True,
        "parameter_count": 2_632_710,
    }
    _validate_config_fields(config, expected, "O1")


def _validate_o2_config(config: O2Config) -> None:
    expected: dict[str, object] = {
        "input_dim": 768,
        "hidden_dims": (1024, 1024),
        "identity_dim": 256,
        "score_dim": 2,
        "layer_norm_eps": 1.0e-5,
        "activation": "silu",
        "dropout": 0.0,
        "linear_bias": True,
        "identity_normalization": "l2_fp32_unit_basis_fallback",
        "normalization_eps": 1.0e-8,
        "score_names": O2SoftOutput.SCORE_NAMES,
        "parameter_count": 2_499_843,
    }
    _validate_config_fields(config, expected, "O2")


def _validate_e1_config(config: E1Config) -> None:
    expected: dict[str, object] = {
        "input_dim": 768,
        "channels": 512,
        "num_layers": 5,
        "kernel_size": 3,
        "dilations": (1, 2, 4, 8, 16),
        "output_dim": 3,
        "layer_norm_eps": 1.0e-5,
        "activation": "silu_filter_sigmoid_gate",
        "strict_causal": True,
        "batch_norm": False,
        "dropout": 0.0,
        "convolution_bias": True,
        "causal_padding": "left",
        "receptive_field": 63,
        "streaming_state_mode": "projected_history",
        "overlap_tubelets": 4,
        "history_tubelets": 66,
        "state_owner_keys": ("video_id", "trajectory_id", "query_signature"),
        "detach_runtime_default": True,
        "output_names": E1SoftOutput.LOGIT_NAMES,
        "parameter_count": 9_717_252,
    }
    _validate_config_fields(config, expected, "E1")


def _validate_e2_config(config: E2Config) -> None:
    expected: dict[str, object] = {
        "input_dim": 768,
        "hidden_dim": 768,
        "num_layers": 2,
        "event_output_dim": 4,
        "phase_output_dim": 4,
        "layer_norm_eps": 1.0e-5,
        "bidirectional": False,
        "batch_first": True,
        "bias": True,
        "dropout": 0.0,
        "streaming_state_mode": "hidden_with_rollback_checkpoints",
        "overlap_tubelets": 4,
        "checkpoint_tubelets": 5,
        "state_owner_keys": ("video_id", "trajectory_id", "query_signature"),
        "detach_runtime_default": True,
        "event_names": E2SoftOutput.EVENT_NAMES,
        "phase_names": E2SoftOutput.PHASE_NAMES,
        "parameter_count": 7_293_449,
    }
    _validate_config_fields(config, expected, "E2")


def _validate_config_fields(config: object, expected: dict[str, object], name: str) -> None:
    for field_name, required in expected.items():
        if getattr(config, field_name) != required:
            raise ValueError(f"P8 requires {name} {field_name}={required!r}")
