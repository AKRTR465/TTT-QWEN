"""Implement the typed P14 TTT, State, Answer, and Outer loss contracts.

Inputs: differentiable soft predictions, explicit dense labels/masks, and detached snapshots.
Outputs: FP32 per-row loss terms, validity/skip audits, metrics, and composed objectives.
Forbidden: hard Bank/FSM inputs, fabricated count labels, parameter updates, or O1 inner loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import PredictorConfig

_FP32_EPS = 1.0e-8
_NORM_ATOL = 5.0e-4
_PRED_WEIGHT = 1.0
_IDENTITY_WEIGHT = 0.5
_EVENT_WEIGHT = 0.5
_O1_UNLABELED_WEIGHT = 0.0
_TASK_WEIGHT = 1.0
_OPERATOR_WEIGHT = 1.0
_RETRIEVAL_WEIGHT = 1.0
_TIME_WEIGHT = 1.0
_AUXILIARY_OUTER_WEIGHT = 0.1


class LossSkipReason(StrEnum):
    """Auditable row-level reason for excluding a value from a reduction."""

    INSUFFICIENT_TIME = "insufficient_time"
    NO_CONTIGUOUS_PAIR = "no_contiguous_pair"
    NO_RELIABLE_MATCH = "no_reliable_match"
    NO_ALIGNED_EVENT = "no_aligned_event"
    NOT_APPLICABLE = "not_applicable"
    NO_TASK_LABEL = "no_task_label"
    NO_VALID_LABEL = "no_valid_label"
    NO_OPERATOR_LABEL = "no_operator_label"
    NO_RETRIEVAL_LABEL = "no_retrieval_label"
    NO_TIME_LABEL = "no_time_label"
    NO_SPAN_LABEL = "no_span_label"
    NO_ANSWER_TOKEN = "no_answer_token"
    NO_NUMBER_TOKEN = "no_number_token"
    NO_READER_COUNT = "no_reader_count"
    NO_VALID_SUPPORT = "no_valid_support"


class IdentityPairStatus(IntEnum):
    """Fixed non-learned disposition for one proposed overlap identity pair."""

    MATCHED = 0
    MISMATCH = 1
    DUPLICATE = 2
    LOW_CONFIDENCE = 3
    INVALID_SOURCE = 4
    PADDING = 5


@dataclass(frozen=True, slots=True)
class LossTerm:
    """One FP32 reduction with enough row detail to distinguish invalid from zero."""

    value: Tensor
    per_row: Tensor
    row_valid_mask: Tensor
    valid_counts: Tensor
    mask_counts: Tensor
    skip_reasons: tuple[LossSkipReason | None, ...]

    def __post_init__(self) -> None:
        _require_fp32_scalar(self.value, "loss value")
        batch_size = self.per_row.shape[0] if self.per_row.ndim == 1 else -1
        if self.per_row.dtype != torch.float32 or self.per_row.shape != (batch_size,):
            raise ValueError("loss per_row must be FP32 [B]")
        if self.row_valid_mask.shape != (batch_size,) or self.row_valid_mask.dtype != torch.bool:
            raise ValueError("loss row_valid_mask must be bool [B]")
        for counts, name in (
            (self.valid_counts, "valid_counts"),
            (self.mask_counts, "mask_counts"),
        ):
            if counts.shape != (batch_size,) or counts.dtype != torch.int64:
                raise ValueError(f"loss {name} must be int64 [B]")
        tensors = (
            self.per_row,
            self.row_valid_mask,
            self.valid_counts,
            self.mask_counts,
        )
        if any(tensor.device != self.value.device for tensor in tensors):
            raise ValueError("all LossTerm tensors must share one device")
        if len(self.skip_reasons) != batch_size:
            raise ValueError("loss skip_reasons must contain one entry per row")
        if any(
            (reason is None) != bool(self.row_valid_mask[row].item())
            for row, reason in enumerate(self.skip_reasons)
        ):
            raise ValueError("valid rows need no skip reason; invalid rows need one")
        if bool(torch.any(self.valid_counts < 0)) or bool(torch.any(self.mask_counts < 0)):
            raise ValueError("loss audit counts must be non-negative")
        if bool(torch.any(self.valid_counts > self.mask_counts)):
            raise ValueError("valid_counts cannot exceed mask_counts")
        if not torch.equal(self.row_valid_mask, self.valid_counts > 0):
            raise ValueError("row validity must exactly match positive valid_counts")
        if bool(torch.any(self.per_row[~self.row_valid_mask] != 0.0)):
            raise ValueError("invalid loss rows must have exact zero values")
        _require_finite(self.value, "loss value")
        _require_finite(self.per_row, "loss per_row")
        expected_value = (
            self.per_row[self.row_valid_mask].mean()
            if bool(self.row_valid_mask.any().item())
            else self.per_row.sum() * 0.0
        )
        if not torch.allclose(
            self.value.detach(), expected_value.detach(), atol=1.0e-6, rtol=1.0e-6
        ):
            raise ValueError("loss value must be the mean of union-valid per_row values")


@dataclass(frozen=True, slots=True)
class TemporalPredictionInput:
    hidden: Tensor
    valid_mask: Tensor
    position_ids: Tensor

    def __post_init__(self) -> None:
        if (
            self.hidden.ndim != 3
            or not torch.is_floating_point(self.hidden)
            or self.hidden.shape[0] <= 0
        ):
            raise ValueError("temporal hidden must be floating [B, T, D] with B > 0")
        shape = self.hidden.shape[:2]
        if self.valid_mask.shape != shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("temporal valid_mask must be bool [B, T]")
        if self.position_ids.shape != shape or self.position_ids.dtype != torch.int64:
            raise ValueError("temporal position_ids must be int64 [B, T]")
        _require_same_device(
            (self.hidden, self.valid_mask, self.position_ids), "temporal prediction input"
        )
        _require_finite(self.hidden, "temporal hidden")
        if self.hidden.device.type != "meta":
            if bool(torch.any(self.hidden[~self.valid_mask] != 0.0)):
                raise ValueError("invalid temporal hidden positions must be zero")
            if bool(torch.any(self.position_ids[~self.valid_mask] != -1)):
                raise ValueError("invalid temporal positions must use -1")


class TemporalPredictor(nn.Module):  # type: ignore[misc]
    """LayerNorm -> Linear -> SiLU -> Linear next-tubelet predictor."""

    def __init__(self, config: PredictorConfig) -> None:
        super().__init__()
        if float(config.layer_norm_eps) != 1.0e-5:
            raise ValueError("Predictor layer_norm_eps is frozen at 1e-5")
        if config.activation != "silu":
            raise ValueError("Predictor activation is frozen at silu")
        if not config.linear_bias:
            raise ValueError("Predictor Linear layers require bias")
        self.input_dim = int(config.input_dim)
        self.output_dim = int(config.output_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(
                self.input_dim,
                eps=float(config.layer_norm_eps),
                elementwise_affine=True,
                bias=config.linear_bias,
            ),
            nn.Linear(self.input_dim, int(config.hidden_dim), bias=config.linear_bias),
            nn.SiLU(),
            nn.Linear(int(config.hidden_dim), self.output_dim, bias=config.linear_bias),
        )
        actual_parameter_count = sum(parameter.numel() for parameter in self.parameters())
        if int(config.parameter_count) != actual_parameter_count:
            raise ValueError("Predictor parameter_count does not match its configured topology")
        if actual_parameter_count != 2_363_136:
            raise ValueError("P14 Predictor must contain exactly 2,363,136 parameters")

    def forward(self, hidden: Tensor) -> Tensor:
        if (
            hidden.ndim != 3
            or hidden.shape[-1] != self.input_dim
            or not torch.is_floating_point(hidden)
        ):
            raise ValueError(f"predictor input must be floating [B, T, {self.input_dim}]")
        _require_finite(hidden, "predictor input")
        output = self.network(hidden)
        _require_finite(output, "predictor output")
        return output


def build_temporal_predictor(config: PredictorConfig) -> TemporalPredictor:
    """Build from only the three configured dimensions; eps and biases are frozen by P14."""

    return TemporalPredictor(config)


@dataclass(frozen=True, slots=True)
class IdentityConsistencyInput:
    """O2 identity banks plus explicit proposed-pair dispositions."""

    current_predictions: Tensor
    previous_targets: Tensor
    current_valid_mask: Tensor
    previous_valid_mask: Tensor
    current_indices: Tensor
    previous_indices: Tensor
    statuses: Tensor
    current_position_ids: Tensor
    previous_position_ids: Tensor
    current_timestamps: Tensor
    previous_timestamps: Tensor

    def __post_init__(self) -> None:
        for tensor, name in (
            (self.current_predictions, "current_predictions"),
            (self.previous_targets, "previous_targets"),
        ):
            if (
                tensor.ndim != 3
                or tensor.shape[1] <= 0
                or tensor.shape[-1] != 256
                or not torch.is_floating_point(tensor)
            ):
                raise ValueError(f"identity {name} must be floating [B, N, 256]")
            _require_finite(tensor, f"identity {name}")
        batch_size = self.current_predictions.shape[0]
        if self.previous_targets.shape[0] != batch_size:
            raise ValueError("current and previous identity banks must share B")
        if self.current_valid_mask.shape != self.current_predictions.shape[:2]:
            raise ValueError("current identity valid mask has the wrong shape")
        if self.previous_valid_mask.shape != self.previous_targets.shape[:2]:
            raise ValueError("previous identity valid mask has the wrong shape")
        if (
            self.current_valid_mask.dtype != torch.bool
            or self.previous_valid_mask.dtype != torch.bool
        ):
            raise TypeError("identity valid masks must use bool dtype")
        pair_shape = self.current_indices.shape
        if (
            len(pair_shape) != 2
            or pair_shape[0] != batch_size
            or self.previous_indices.shape != pair_shape
            or self.statuses.shape != pair_shape
        ):
            raise ValueError("identity pair indices/statuses must share [B, M]")
        if any(
            tensor.dtype != torch.int64
            for tensor in (self.current_indices, self.previous_indices, self.statuses)
        ):
            raise TypeError("identity pair indices/statuses must use int64 dtype")
        if (
            self.current_position_ids.shape != pair_shape
            or self.previous_position_ids.shape != pair_shape
            or self.current_position_ids.dtype != torch.int64
            or self.previous_position_ids.dtype != torch.int64
        ):
            raise ValueError("identity pair positions must be int64 [B, M]")
        if (
            self.current_timestamps.shape != pair_shape
            or self.previous_timestamps.shape != pair_shape
            or not torch.is_floating_point(self.current_timestamps)
            or not torch.is_floating_point(self.previous_timestamps)
        ):
            raise ValueError("identity pair timestamps must be floating [B, M]")
        _require_same_device(
            (
                self.current_predictions,
                self.previous_targets,
                self.current_valid_mask,
                self.previous_valid_mask,
                self.current_indices,
                self.previous_indices,
                self.statuses,
                self.current_position_ids,
                self.previous_position_ids,
                self.current_timestamps,
                self.previous_timestamps,
            ),
            "identity consistency input",
        )
        _require_finite(self.current_timestamps, "current identity timestamps")
        _require_finite(self.previous_timestamps, "previous identity timestamps")
        if self.current_predictions.device.type == "meta":
            return
        allowed = torch.tensor(
            [int(status) for status in IdentityPairStatus], device=self.statuses.device
        )
        if bool(torch.any(~torch.isin(self.statuses, allowed))):
            raise ValueError("identity pair statuses contain an unknown value")
        padding = self.statuses == int(IdentityPairStatus.PADDING)
        invalid_source = self.statuses == int(IdentityPairStatus.INVALID_SOURCE)
        if bool(
            torch.any(
                padding
                & (
                    (self.current_indices != -1)
                    | (self.previous_indices != -1)
                    | (self.current_position_ids != -1)
                    | (self.previous_position_ids != -1)
                    | (self.current_timestamps != -1.0)
                    | (self.previous_timestamps != -1.0)
                )
            )
        ):
            raise ValueError("padding identity pairs must use -1 sentinels")
        if bool(
            torch.any(
                invalid_source
                & (
                    (self.current_indices != -1)
                    | (self.previous_indices != -1)
                    | (self.current_position_ids != -1)
                    | (self.previous_position_ids != -1)
                    | (self.current_timestamps != -1.0)
                    | (self.previous_timestamps != -1.0)
                )
            )
        ):
            raise ValueError("invalid-source identity pairs must use -1 sentinels")
        source_required = ~(padding | invalid_source)
        current_width = self.current_predictions.shape[1]
        previous_width = self.previous_targets.shape[1]
        if bool(
            torch.any(
                source_required
                & (
                    (self.current_indices < 0)
                    | (self.current_indices >= current_width)
                    | (self.previous_indices < 0)
                    | (self.previous_indices >= previous_width)
                )
            )
        ):
            raise ValueError("non-padding identity pair indices are out of range")
        safe_current = self.current_indices.clamp_min(0)
        safe_previous = self.previous_indices.clamp_min(0)
        current_refs = torch.gather(self.current_valid_mask, 1, safe_current)
        previous_refs = torch.gather(self.previous_valid_mask, 1, safe_previous)
        if bool(torch.any(source_required & (~current_refs | ~previous_refs))):
            raise ValueError("identity pairs may reference only valid O2 slots")
        if bool(
            torch.any(
                source_required
                & (
                    (self.current_position_ids < 0)
                    | (self.previous_position_ids < 0)
                    | (self.current_timestamps < 0.0)
                    | (self.previous_timestamps < 0.0)
                )
            )
        ):
            raise ValueError("valid identity sources require non-negative time metadata")
        _require_unit_norm(self.current_predictions, self.current_valid_mask, "current O2 identity")
        _require_unit_norm(self.previous_targets, self.previous_valid_mask, "previous O2 identity")
        matched = self.statuses == int(IdentityPairStatus.MATCHED)
        if bool(torch.any(matched & (self.current_position_ids != self.previous_position_ids))):
            raise ValueError("matched identity positions must be equal")
        if bool(
            torch.any(
                matched & ((self.current_timestamps - self.previous_timestamps).abs() > 1.0e-6)
            )
        ):
            raise ValueError("matched identity timestamps must agree within 1e-6")
        for row in range(batch_size):
            current = self.current_indices[row, matched[row]].tolist()
            previous = self.previous_indices[row, matched[row]].tolist()
            if len(set(current)) != len(current) or len(set(previous)) != len(previous):
                raise ValueError("MATCHED identity pairs must be one-to-one")
            current_positions = self.current_position_ids[row, matched[row]].tolist()
            previous_positions = self.previous_position_ids[row, matched[row]].tolist()
            if len(set(current_positions)) != len(current_positions) or len(
                set(previous_positions)
            ) != len(previous_positions):
                raise ValueError("MATCHED identity positions must be unique per row")


@dataclass(frozen=True, slots=True)
class IdentityConsistencyAudit:
    matched_counts: Tensor
    mismatch_counts: Tensor
    duplicate_counts: Tensor
    low_confidence_counts: Tensor
    invalid_source_counts: Tensor
    padding_counts: Tensor

    def __post_init__(self) -> None:
        fields = (
            self.matched_counts,
            self.mismatch_counts,
            self.duplicate_counts,
            self.low_confidence_counts,
            self.invalid_source_counts,
            self.padding_counts,
        )
        shape = fields[0].shape
        if len(shape) != 1 or any(
            field.shape != shape or field.dtype != torch.int64 for field in fields
        ):
            raise ValueError("identity audit counts must be aligned int64 [B]")
        _require_same_device(fields, "identity consistency audit")
        if any(bool(torch.any(field < 0)) for field in fields):
            raise ValueError("identity audit counts must be non-negative")


@dataclass(frozen=True, slots=True)
class IdentityLossOutput:
    term: LossTerm
    audit: IdentityConsistencyAudit


@dataclass(frozen=True, slots=True)
class E1ConsistencyInput:
    current_probabilities: Tensor
    previous_target_probabilities: Tensor
    pair_mask: Tensor
    alignment_mask: Tensor
    current_position_ids: Tensor
    previous_position_ids: Tensor
    current_timestamps: Tensor
    previous_timestamps: Tensor

    def __post_init__(self) -> None:
        _validate_overlap_probabilities(
            self.current_probabilities,
            self.previous_target_probabilities,
            self.pair_mask,
            self.alignment_mask,
            self.current_position_ids,
            self.previous_position_ids,
            self.current_timestamps,
            self.previous_timestamps,
            width=3,
            name="E1",
            phase=False,
        )


@dataclass(frozen=True, slots=True)
class E2ConsistencyInput:
    current_event_probabilities: Tensor
    previous_event_target_probabilities: Tensor
    current_phase_probabilities: Tensor
    previous_phase_target_probabilities: Tensor
    pair_mask: Tensor
    alignment_mask: Tensor
    current_position_ids: Tensor
    previous_position_ids: Tensor
    current_timestamps: Tensor
    previous_timestamps: Tensor

    def __post_init__(self) -> None:
        args = (
            self.pair_mask,
            self.alignment_mask,
            self.current_position_ids,
            self.previous_position_ids,
            self.current_timestamps,
            self.previous_timestamps,
        )
        _validate_overlap_probabilities(
            self.current_event_probabilities,
            self.previous_event_target_probabilities,
            *args,
            width=4,
            name="E2 event",
            phase=False,
        )
        _validate_overlap_probabilities(
            self.current_phase_probabilities,
            self.previous_phase_target_probabilities,
            *args,
            width=4,
            name="E2 phase",
            phase=True,
        )


@dataclass(frozen=True, slots=True)
class EventConsistencyInput:
    e1: E1ConsistencyInput
    e2: E2ConsistencyInput

    def __post_init__(self) -> None:
        if self.e1.current_probabilities.shape[0] != self.e2.current_event_probabilities.shape[0]:
            raise ValueError("E1 and E2 consistency batches must share B")
        if self.e1.current_probabilities.device != self.e2.current_event_probabilities.device:
            raise ValueError("E1 and E2 consistency inputs must share one device")


@dataclass(frozen=True, slots=True)
class EventLossOutput:
    e1: LossTerm
    e2: LossTerm
    total: LossTerm

    def __post_init__(self) -> None:
        expected_per_row = self.e1.per_row + self.e2.per_row
        if not torch.allclose(self.total.per_row, expected_per_row, atol=1.0e-6, rtol=1.0e-6):
            raise ValueError("event total per_row must equal E1 + E2")


@dataclass(frozen=True, slots=True)
class TTTLossInput:
    temporal: TemporalPredictionInput
    identity: IdentityConsistencyInput
    event: EventConsistencyInput

    def __post_init__(self) -> None:
        batch_size = self.temporal.hidden.shape[0]
        if self.identity.current_predictions.shape[0] != batch_size:
            raise ValueError("temporal and identity TTT inputs must share B")
        if self.event.e1.current_probabilities.shape[0] != batch_size:
            raise ValueError("temporal and event TTT inputs must share B")
        if self.temporal.hidden.device != self.identity.current_predictions.device:
            raise ValueError("all TTT inputs must share one device")
        if self.temporal.hidden.device != self.event.e1.current_probabilities.device:
            raise ValueError("all TTT inputs must share one device")


@dataclass(frozen=True, slots=True)
class TTTLossOutput:
    pred: LossTerm
    identity: LossTerm
    e1_event: LossTerm
    e2_event: LossTerm
    event: LossTerm
    total: Tensor
    per_row_total: Tensor
    update_valid_mask: Tensor
    identity_audit: IdentityConsistencyAudit
    pred_weight: float = _PRED_WEIGHT
    identity_weight: float = _IDENTITY_WEIGHT
    event_weight: float = _EVENT_WEIGHT
    o1_unlabeled_weight: float = _O1_UNLABELED_WEIGHT

    def __post_init__(self) -> None:
        _require_fp32_scalar(self.total, "TTT total")
        batch_size = self.pred.per_row.shape[0]
        if self.per_row_total.shape != (batch_size,) or self.per_row_total.dtype != torch.float32:
            raise ValueError("TTT per_row_total must be FP32 [B]")
        if (
            self.update_valid_mask.shape != (batch_size,)
            or self.update_valid_mask.dtype != torch.bool
        ):
            raise ValueError("TTT update_valid_mask must be bool [B]")
        if (
            self.per_row_total.device != self.total.device
            or self.update_valid_mask.device != self.total.device
        ):
            raise ValueError("TTT outputs must share one device")
        if (self.pred_weight, self.identity_weight, self.event_weight) != (1.0, 0.5, 0.5):
            raise ValueError("P14 TTT weights are frozen at 1/0.5/0.5")
        if self.o1_unlabeled_weight != 0.0:
            raise ValueError("O1 unlabeled loss is forbidden")
        _require_finite(self.total, "TTT total")
        _require_finite(self.per_row_total, "TTT per_row_total")
        if bool(torch.any(self.per_row_total[~self.update_valid_mask] != 0.0)):
            raise ValueError("invalid TTT rows must have exact zero values")
        expected_total = (
            self.per_row_total[self.update_valid_mask].mean()
            if bool(self.update_valid_mask.any().item())
            else self.per_row_total.sum() * 0.0
        )
        if not torch.allclose(
            self.total.detach(), expected_total.detach(), atol=1.0e-6, rtol=1.0e-6
        ):
            raise ValueError("TTT total must be the mean of union-valid per_row_total")


def compute_temporal_prediction_loss(
    predictor: TemporalPredictor, inputs: TemporalPredictionInput
) -> LossTerm:
    """Predict the next contiguous valid tubelet and detach only its target."""

    if inputs.hidden.shape[-1] != predictor.input_dim:
        raise ValueError("temporal hidden size does not match PredictorConfig.input_dim")
    if predictor.output_dim != inputs.hidden.shape[-1]:
        raise ValueError("PredictorConfig.output_dim must match temporal hidden size")
    batch_size, length, _ = inputs.hidden.shape
    if length < 2:
        zero = _differentiable_zero(inputs.hidden)
        zeros = torch.zeros(batch_size, dtype=torch.int64, device=inputs.hidden.device)
        return _make_term_from_rows(
            torch.zeros(batch_size, dtype=torch.float32, device=inputs.hidden.device) + zero,
            zeros,
            zeros,
            tuple(LossSkipReason.INSUFFICIENT_TIME for _ in range(batch_size)),
        )
    predictions = predictor(inputs.hidden[:, :-1])
    targets = inputs.hidden[:, 1:].detach()
    candidate_mask = inputs.valid_mask[:, :-1] & inputs.valid_mask[:, 1:]
    contiguous_mask = candidate_mask & (
        inputs.position_ids[:, 1:] == inputs.position_ids[:, :-1] + 1
    )
    item_losses = (predictions.float() - targets.float()).square().mean(dim=-1)
    valid_counts = contiguous_mask.sum(dim=1, dtype=torch.int64)
    mask_counts = candidate_mask.sum(dim=1, dtype=torch.int64)
    reasons = tuple(
        None
        if int(valid_counts[row].item()) > 0
        else (
            LossSkipReason.INSUFFICIENT_TIME
            if int(mask_counts[row].item()) == 0
            else LossSkipReason.NO_CONTIGUOUS_PAIR
        )
        for row in range(batch_size)
    )
    return _reduce_items(item_losses, contiguous_mask, mask_counts, reasons)


def compute_identity_consistency_loss(inputs: IdentityConsistencyInput) -> IdentityLossOutput:
    """Train current O2 identities toward detached previous snapshots for reliable pairs."""

    matched = inputs.statuses == int(IdentityPairStatus.MATCHED)
    safe_current = inputs.current_indices.clamp_min(0)
    safe_previous = inputs.previous_indices.clamp_min(0)
    feature_dim = inputs.current_predictions.shape[-1]
    current = torch.gather(
        inputs.current_predictions,
        1,
        safe_current.unsqueeze(-1).expand(-1, -1, feature_dim),
    )
    previous = torch.gather(
        inputs.previous_targets,
        1,
        safe_previous.unsqueeze(-1).expand(-1, -1, feature_dim),
    ).detach()
    item_losses = 1.0 - (current.float() * previous.float()).sum(dim=-1)
    valid_counts = matched.sum(dim=1, dtype=torch.int64)
    mask_counts = (inputs.statuses != int(IdentityPairStatus.PADDING)).sum(dim=1, dtype=torch.int64)
    reasons = tuple(
        None if int(count.item()) > 0 else LossSkipReason.NO_RELIABLE_MATCH
        for count in valid_counts
    )
    term = _reduce_items(item_losses, matched, mask_counts, reasons)
    audit = IdentityConsistencyAudit(
        matched_counts=valid_counts,
        mismatch_counts=(inputs.statuses == int(IdentityPairStatus.MISMATCH)).sum(
            dim=1, dtype=torch.int64
        ),
        duplicate_counts=(inputs.statuses == int(IdentityPairStatus.DUPLICATE)).sum(
            dim=1, dtype=torch.int64
        ),
        low_confidence_counts=(inputs.statuses == int(IdentityPairStatus.LOW_CONFIDENCE)).sum(
            dim=1, dtype=torch.int64
        ),
        invalid_source_counts=(inputs.statuses == int(IdentityPairStatus.INVALID_SOURCE)).sum(
            dim=1, dtype=torch.int64
        ),
        padding_counts=(inputs.statuses == int(IdentityPairStatus.PADDING)).sum(
            dim=1, dtype=torch.int64
        ),
    )
    return IdentityLossOutput(term=term, audit=audit)


def compute_event_consistency_loss(inputs: EventConsistencyInput) -> EventLossOutput:
    """Compare aligned E1/E2 soft outputs against detached previous snapshots."""

    e1_items = (
        (
            inputs.e1.current_probabilities.float()
            - inputs.e1.previous_target_probabilities.detach().float()
        )
        .square()
        .mean(dim=-1)
    )
    e1 = _reduce_overlap_items(e1_items, inputs.e1.pair_mask, inputs.e1.alignment_mask)

    event_mse = (
        (
            inputs.e2.current_event_probabilities.float()
            - inputs.e2.previous_event_target_probabilities.detach().float()
        )
        .square()
        .mean(dim=-1)
    )
    target_phase = inputs.e2.previous_phase_target_probabilities.detach().float()
    target_phase = target_phase.clamp_min(_FP32_EPS)
    target_phase = target_phase / target_phase.sum(dim=-1, keepdim=True)
    current_phase = inputs.e2.current_phase_probabilities.float().clamp_min(_FP32_EPS)
    current_phase = current_phase / current_phase.sum(dim=-1, keepdim=True)
    phase_kl = (target_phase * (target_phase.log() - current_phase.log())).sum(dim=-1)
    e2_items = event_mse + phase_kl
    e2 = _reduce_overlap_items(e2_items, inputs.e2.pair_mask, inputs.e2.alignment_mask)

    per_row = e1.per_row + e2.per_row
    valid_counts = e1.valid_counts + e2.valid_counts
    mask_counts = e1.mask_counts + e2.mask_counts
    row_valid = valid_counts > 0
    reasons = tuple(
        None if bool(row_valid[row].item()) else LossSkipReason.NO_ALIGNED_EVENT
        for row in range(per_row.shape[0])
    )
    total = _make_term_from_rows(per_row, valid_counts, mask_counts, reasons)
    return EventLossOutput(e1=e1, e2=e2, total=total)


def compute_ttt_loss(predictor: TemporalPredictor, inputs: TTTLossInput) -> TTTLossOutput:
    pred = compute_temporal_prediction_loss(predictor, inputs.temporal)
    identity_output = compute_identity_consistency_loss(inputs.identity)
    event_output = compute_event_consistency_loss(inputs.event)
    per_row = (
        _PRED_WEIGHT * pred.per_row
        + _IDENTITY_WEIGHT * identity_output.term.per_row
        + _EVENT_WEIGHT * event_output.total.per_row
    )
    update_valid = (
        pred.row_valid_mask
        | identity_output.term.row_valid_mask
        | event_output.total.row_valid_mask
    )
    total = per_row[update_valid].mean() if bool(update_valid.any().item()) else per_row.sum() * 0.0
    return TTTLossOutput(
        pred=pred,
        identity=identity_output.term,
        e1_event=event_output.e1,
        e2_event=event_output.e2,
        event=event_output.total,
        total=total,
        per_row_total=per_row,
        update_valid_mask=update_valid,
        identity_audit=identity_output.audit,
    )


@dataclass(frozen=True, slots=True)
class O1StateTarget:
    """P15-provided pre-matched dense slot labels; P14 never fabricates slot matching."""

    row_indices: Tensor
    logits: Tensor
    targets: Tensor
    slot_mask: Tensor

    def __post_init__(self) -> None:
        _validate_dense_binary_target(
            self.row_indices, self.logits, self.targets, self.slot_mask, 6, "O1"
        )


@dataclass(frozen=True, slots=True)
class O2StateTarget:
    row_indices: Tensor
    identity_predictions: Tensor
    identity_targets: Tensor
    score_logits: Tensor
    score_targets: Tensor
    slot_mask: Tensor

    def __post_init__(self) -> None:
        _validate_row_indices(self.row_indices, self.identity_predictions.shape[0], "O2")
        if (
            self.identity_predictions.ndim != 3
            or self.identity_predictions.shape[-1] != 256
            or not torch.is_floating_point(self.identity_predictions)
            or self.identity_targets.shape != self.identity_predictions.shape
            or not torch.is_floating_point(self.identity_targets)
        ):
            raise ValueError("O2 identities must be aligned floating [R, N, 256]")
        shape = self.identity_predictions.shape[:2]
        if (
            self.score_logits.shape != (*shape, 2)
            or self.score_targets.shape != self.score_logits.shape
            or not torch.is_floating_point(self.score_logits)
            or not torch.is_floating_point(self.score_targets)
        ):
            raise ValueError("O2 score logits/targets must be floating [R, N, 2]")
        if self.slot_mask.shape != shape or self.slot_mask.dtype != torch.bool:
            raise ValueError("O2 slot_mask must be bool [R, N]")
        tensors = (
            self.row_indices,
            self.identity_predictions,
            self.identity_targets,
            self.score_logits,
            self.score_targets,
            self.slot_mask,
        )
        _require_same_device(tensors, "O2 State target")
        for tensor, name in (
            (self.identity_predictions, "O2 identity predictions"),
            (self.identity_targets, "O2 identity targets"),
            (self.score_logits, "O2 score logits"),
            (self.score_targets, "O2 score targets"),
        ):
            _require_finite(tensor, name)
        _require_probability_targets(self.score_targets, "O2 score targets")
        _require_unit_norm(self.identity_predictions, self.slot_mask, "O2 identity predictions")
        _require_unit_norm(self.identity_targets, self.slot_mask, "O2 identity targets")


@dataclass(frozen=True, slots=True)
class E1StateTarget:
    row_indices: Tensor
    logits: Tensor
    targets: Tensor
    time_mask: Tensor

    def __post_init__(self) -> None:
        _validate_dense_binary_target(
            self.row_indices, self.logits, self.targets, self.time_mask, 3, "E1"
        )


@dataclass(frozen=True, slots=True)
class E2StateTarget:
    """Dense E2 labels whose phase CE is the soft-FSM proxy, never a hard FSM input."""

    row_indices: Tensor
    event_logits: Tensor
    event_targets: Tensor
    phase_logits: Tensor
    phase_targets: Tensor
    time_mask: Tensor

    def __post_init__(self) -> None:
        _validate_row_indices(self.row_indices, self.event_logits.shape[0], "E2")
        if (
            self.event_logits.ndim != 3
            or self.event_logits.shape[-1] != 4
            or not torch.is_floating_point(self.event_logits)
            or self.event_targets.shape != self.event_logits.shape
            or not torch.is_floating_point(self.event_targets)
            or self.phase_logits.shape != self.event_logits.shape
            or not torch.is_floating_point(self.phase_logits)
        ):
            raise ValueError("E2 event/phase tensors must be floating [R, T, 4]")
        shape = self.event_logits.shape[:2]
        if self.phase_targets.shape != shape or self.phase_targets.dtype != torch.int64:
            raise ValueError("E2 phase_targets must be int64 [R, T]")
        if self.time_mask.shape != shape or self.time_mask.dtype != torch.bool:
            raise ValueError("E2 time_mask must be bool [R, T]")
        tensors = (
            self.row_indices,
            self.event_logits,
            self.event_targets,
            self.phase_logits,
            self.phase_targets,
            self.time_mask,
        )
        _require_same_device(tensors, "E2 State target")
        _require_finite(self.event_logits, "E2 event logits")
        _require_finite(self.event_targets, "E2 event targets")
        _require_finite(self.phase_logits, "E2 phase logits")
        _require_probability_targets(self.event_targets, "E2 event targets")
        if self.phase_targets.device.type != "meta":
            valid = self.time_mask
            if bool(torch.any((self.phase_targets[valid] < 0) | (self.phase_targets[valid] >= 4))):
                raise ValueError("valid E2 phase targets must be within [0, 4)")
            if bool(torch.any(~valid & (self.phase_targets != -100))):
                raise ValueError("masked E2 phase targets must use -100")


@dataclass(frozen=True, slots=True)
class OperatorLossInput:
    logits: Tensor
    targets: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        batch_size = self.logits.shape[0] if self.logits.ndim == 2 else -1
        if self.logits.shape != (batch_size, 9) or not torch.is_floating_point(self.logits):
            raise ValueError("operator logits must be floating [B, 9]")
        if self.targets.shape != (batch_size,) or self.targets.dtype != torch.int64:
            raise ValueError("operator targets must be int64 [B]")
        if self.valid_mask.shape != (batch_size,) or self.valid_mask.dtype != torch.bool:
            raise ValueError("operator valid_mask must be bool [B]")
        _require_same_device((self.logits, self.targets, self.valid_mask), "operator loss input")
        _require_finite(self.logits, "operator logits")
        if self.targets.device.type != "meta" and bool(
            torch.any(self.valid_mask & ((self.targets < 0) | (self.targets >= 9)))
        ):
            raise ValueError("valid operator targets must be within [0, 9)")


@dataclass(frozen=True, slots=True)
class RetrievalLossInput:
    logits: Tensor
    targets: Tensor
    present_mask: Tensor
    label_mask: Tensor

    def __post_init__(self) -> None:
        if self.logits.ndim != 2 or not torch.is_floating_point(self.logits):
            raise ValueError("retrieval logits must be floating [B, N]")
        if self.targets.shape != self.logits.shape or not torch.is_floating_point(self.targets):
            raise ValueError("retrieval targets must be floating [B, N]")
        if (
            self.present_mask.shape != self.logits.shape
            or self.label_mask.shape != self.logits.shape
            or self.present_mask.dtype != torch.bool
            or self.label_mask.dtype != torch.bool
        ):
            raise ValueError("retrieval masks must be bool [B, N]")
        _require_same_device(
            (self.logits, self.targets, self.present_mask, self.label_mask),
            "retrieval loss input",
        )
        _require_finite(self.logits, "retrieval logits")
        _require_finite(self.targets, "retrieval targets")
        _require_probability_targets(self.targets, "retrieval targets")
        if self.logits.device.type != "meta" and bool(
            torch.any(self.label_mask & ~self.present_mask)
        ):
            raise ValueError("retrieval labels cannot target padded records")


@dataclass(frozen=True, slots=True)
class TimeLossInput:
    mode_logits: Tensor
    mode_targets: Tensor
    mode_valid_mask: Tensor
    span_start_logits: Tensor
    span_end_logits: Tensor
    span_start_targets: Tensor
    span_end_targets: Tensor
    token_valid_mask: Tensor

    def __post_init__(self) -> None:
        batch_size = self.mode_logits.shape[0] if self.mode_logits.ndim == 2 else -1
        if self.mode_logits.shape != (batch_size, 4) or not torch.is_floating_point(
            self.mode_logits
        ):
            raise ValueError("time mode logits must be floating [B, 4]")
        if self.mode_targets.shape != (batch_size,) or self.mode_targets.dtype != torch.int64:
            raise ValueError("time mode targets must be int64 [B]")
        if self.mode_valid_mask.shape != (batch_size,) or self.mode_valid_mask.dtype != torch.bool:
            raise ValueError("time mode valid_mask must be bool [B]")
        if (
            self.span_start_logits.ndim != 2
            or self.span_start_logits.shape != self.span_end_logits.shape
            or self.span_start_logits.shape[0] != batch_size
            or not torch.is_floating_point(self.span_start_logits)
            or not torch.is_floating_point(self.span_end_logits)
        ):
            raise ValueError("time span logits must be aligned floating [B, L]")
        for target, name in (
            (self.span_start_targets, "span_start_targets"),
            (self.span_end_targets, "span_end_targets"),
        ):
            if target.shape != (batch_size,) or target.dtype != torch.int64:
                raise ValueError(f"time {name} must be int64 [B]")
        if (
            self.token_valid_mask.shape != self.span_start_logits.shape
            or self.token_valid_mask.dtype != torch.bool
        ):
            raise ValueError("time token_valid_mask must be bool [B, L]")
        tensors = (
            self.mode_logits,
            self.mode_targets,
            self.mode_valid_mask,
            self.span_start_logits,
            self.span_end_logits,
            self.span_start_targets,
            self.span_end_targets,
            self.token_valid_mask,
        )
        _require_same_device(tensors, "time loss input")
        _require_finite(self.mode_logits, "time mode logits")
        _require_finite(self.span_start_logits, "time span start logits")
        _require_finite(self.span_end_logits, "time span end logits")
        if self.mode_targets.device.type == "meta":
            return
        if bool(
            torch.any(self.mode_valid_mask & ((self.mode_targets < 0) | (self.mode_targets >= 4)))
        ):
            raise ValueError("valid time mode targets must be within [0, 4)")
        start_ignored = self.span_start_targets == -100
        end_ignored = self.span_end_targets == -100
        if not torch.equal(start_ignored, end_ignored):
            raise ValueError("time span start/end targets must be ignored together")
        span_valid = ~start_ignored
        width = self.span_start_logits.shape[1]
        if bool(
            torch.any(
                span_valid
                & (
                    (self.span_start_targets < 0)
                    | (self.span_start_targets >= width)
                    | (self.span_end_targets < 0)
                    | (self.span_end_targets >= width)
                    | (self.span_start_targets > self.span_end_targets)
                )
            )
        ):
            raise ValueError("valid time spans must satisfy 0 <= start <= end < L")
        safe_start = self.span_start_targets.clamp_min(0)
        safe_end = self.span_end_targets.clamp_min(0)
        start_present = torch.gather(self.token_valid_mask, 1, safe_start.unsqueeze(1)).squeeze(1)
        end_present = torch.gather(self.token_valid_mask, 1, safe_end.unsqueeze(1)).squeeze(1)
        if bool(torch.any(span_valid & (~start_present | ~end_present))):
            raise ValueError("time span targets must point to valid query tokens")


@dataclass(frozen=True, slots=True)
class StateLossInput:
    batch_size: int
    o1: O1StateTarget | None = None
    o2: O2StateTarget | None = None
    e1: E1StateTarget | None = None
    e2: E2StateTarget | None = None
    operator: OperatorLossInput | None = None
    retrieval: RetrievalLossInput | None = None
    time: TimeLossInput | None = None

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("State loss batch_size must be a positive integer")
        components = (self.o1, self.o2, self.e1, self.e2, self.operator, self.retrieval, self.time)
        if all(component is None for component in components):
            raise ValueError("State loss requires at least one explicit supervised input")
        row_sets: list[set[int]] = []
        for target in (self.o1, self.o2, self.e1, self.e2):
            if target is None:
                continue
            rows = target.row_indices.tolist()
            if any(row < 0 or row >= self.batch_size for row in rows):
                raise ValueError("State task row index is outside batch_size")
            row_set = set(rows)
            if any(row_set & existing for existing in row_sets):
                raise ValueError("each State row may supervise exactly one observation head")
            row_sets.append(row_set)
        for component, name in (
            (self.operator, "operator"),
            (self.retrieval, "retrieval"),
            (self.time, "time"),
        ):
            if component is not None and _component_batch_size(component) != self.batch_size:
                raise ValueError(f"State {name} batch size does not match batch_size")
        reference = _state_reference(self)
        for supervised_component in components:
            if (
                supervised_component is not None
                and _component_reference(supervised_component).device != reference.device
            ):
                raise ValueError("all State loss inputs must share one device")


@dataclass(frozen=True, slots=True)
class TimeLossOutput:
    mode: LossTerm
    start: LossTerm
    end: LossTerm
    total: Tensor
    per_row_total: Tensor
    row_valid_mask: Tensor

    def __post_init__(self) -> None:
        _require_fp32_scalar(self.total, "time total")
        batch_size = self.mode.per_row.shape[0]
        if self.per_row_total.shape != (batch_size,) or self.per_row_total.dtype != torch.float32:
            raise ValueError("time per_row_total must be FP32 [B]")
        if self.row_valid_mask.shape != (batch_size,) or self.row_valid_mask.dtype != torch.bool:
            raise ValueError("time row_valid_mask must be bool [B]")
        _require_finite(self.total, "time total")
        _require_finite(self.per_row_total, "time per_row_total")


@dataclass(frozen=True, slots=True)
class StateLossOutput:
    o1: LossTerm
    o2: LossTerm
    e1: LossTerm
    e2: LossTerm
    task: LossTerm
    operator: LossTerm
    retrieval: LossTerm
    time: TimeLossOutput
    total: Tensor
    per_row_total: Tensor
    row_valid_mask: Tensor
    task_weight: float = _TASK_WEIGHT
    operator_weight: float = _OPERATOR_WEIGHT
    retrieval_weight: float = _RETRIEVAL_WEIGHT
    time_weight: float = _TIME_WEIGHT

    def __post_init__(self) -> None:
        _require_fp32_scalar(self.total, "State total")
        batch_size = self.task.per_row.shape[0]
        if self.per_row_total.shape != (batch_size,) or self.per_row_total.dtype != torch.float32:
            raise ValueError("State per_row_total must be FP32 [B]")
        if self.row_valid_mask.shape != (batch_size,) or self.row_valid_mask.dtype != torch.bool:
            raise ValueError("State row_valid_mask must be bool [B]")
        if (
            self.task_weight,
            self.operator_weight,
            self.retrieval_weight,
            self.time_weight,
        ) != (1.0, 1.0, 1.0, 1.0):
            raise ValueError("P14 State weights are frozen at one")
        _require_finite(self.total, "State total")
        _require_finite(self.per_row_total, "State per_row_total")


def compute_state_loss(inputs: StateLossInput) -> StateLossOutput:
    reference = _state_reference(inputs)
    batch_size = inputs.batch_size
    o1 = (
        _invalid_term(batch_size, reference, LossSkipReason.NOT_APPLICABLE)
        if inputs.o1 is None
        else _scatter_term(_compute_o1_state_term(inputs.o1), inputs.o1.row_indices, batch_size)
    )
    o2 = (
        _invalid_term(batch_size, reference, LossSkipReason.NOT_APPLICABLE)
        if inputs.o2 is None
        else _scatter_term(_compute_o2_state_term(inputs.o2), inputs.o2.row_indices, batch_size)
    )
    e1 = (
        _invalid_term(batch_size, reference, LossSkipReason.NOT_APPLICABLE)
        if inputs.e1 is None
        else _scatter_term(_compute_e1_state_term(inputs.e1), inputs.e1.row_indices, batch_size)
    )
    e2 = (
        _invalid_term(batch_size, reference, LossSkipReason.NOT_APPLICABLE)
        if inputs.e2 is None
        else _scatter_term(_compute_e2_state_term(inputs.e2), inputs.e2.row_indices, batch_size)
    )
    head_terms = [o1, o2, e1, e2]
    task_per_row = o1.per_row + o2.per_row + e1.per_row + e2.per_row
    task_counts = o1.valid_counts + o2.valid_counts + e1.valid_counts + e2.valid_counts
    task_masks = o1.mask_counts + o2.mask_counts + e1.mask_counts + e2.mask_counts
    targeted = torch.zeros(batch_size, dtype=torch.bool, device=reference.device)
    targeted_reasons: list[LossSkipReason | None] = [LossSkipReason.NO_TASK_LABEL] * batch_size
    for term, target in zip(head_terms, (inputs.o1, inputs.o2, inputs.e1, inputs.e2), strict=True):
        if target is None:
            continue
        for row in target.row_indices.tolist():
            targeted[row] = True
            targeted_reasons[row] = term.skip_reasons[row]
    task_reasons = tuple(
        None
        if int(task_counts[row].item()) > 0
        else (targeted_reasons[row] if bool(targeted[row].item()) else LossSkipReason.NO_TASK_LABEL)
        for row in range(batch_size)
    )
    task = _make_term_from_rows(task_per_row, task_counts, task_masks, task_reasons)

    operator = (
        _compute_operator_term(inputs.operator)
        if inputs.operator is not None
        else _invalid_term(batch_size, reference, LossSkipReason.NO_OPERATOR_LABEL)
    )
    retrieval = (
        _compute_retrieval_term(inputs.retrieval)
        if inputs.retrieval is not None
        else _invalid_term(batch_size, reference, LossSkipReason.NO_RETRIEVAL_LABEL)
    )
    time = (
        _compute_time_loss(inputs.time)
        if inputs.time is not None
        else _invalid_time_output(batch_size, reference)
    )
    per_row = task.per_row + operator.per_row + retrieval.per_row + time.per_row_total
    row_valid = (
        task.row_valid_mask
        | operator.row_valid_mask
        | retrieval.row_valid_mask
        | time.row_valid_mask
    )
    total = task.value + operator.value + retrieval.value + time.total
    return StateLossOutput(
        o1=o1,
        o2=o2,
        e1=e1,
        e2=e2,
        task=task,
        operator=operator,
        retrieval=retrieval,
        time=time,
        total=total,
        per_row_total=per_row,
        row_valid_mask=row_valid,
    )


@dataclass(frozen=True, slots=True)
class ReaderCountMetricInput:
    predicted_counts: Tensor
    target_counts: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        batch_size = self.predicted_counts.shape[0] if self.predicted_counts.ndim == 1 else -1
        for tensor, name in (
            (self.predicted_counts, "predicted_counts"),
            (self.target_counts, "target_counts"),
        ):
            if tensor.shape != (batch_size,) or tensor.dtype != torch.int64:
                raise ValueError(f"Reader {name} must be int64 [B]")
        if self.valid_mask.shape != (batch_size,) or self.valid_mask.dtype != torch.bool:
            raise ValueError("Reader count valid_mask must be bool [B]")
        _require_same_device(
            (self.predicted_counts, self.target_counts, self.valid_mask), "Reader count metric"
        )


@dataclass(frozen=True, slots=True)
class AnswerLossInput:
    logits: Tensor
    labels: Tensor
    number_token_mask: Tensor
    reader_counts: ReaderCountMetricInput | None = None

    def __post_init__(self) -> None:
        if (
            self.logits.ndim != 3
            or self.logits.shape[0] <= 0
            or self.logits.shape[1] < 2
            or self.logits.shape[2] <= 1
            or not torch.is_floating_point(self.logits)
        ):
            raise ValueError("answer logits must be floating [B, L>=2, V>1]")
        shape = self.logits.shape[:2]
        if self.labels.shape != shape or self.labels.dtype != torch.int64:
            raise ValueError("answer labels must be int64 [B, L]")
        if self.number_token_mask.shape != shape or self.number_token_mask.dtype != torch.bool:
            raise ValueError("answer number_token_mask must be bool [B, L]")
        _require_same_device(
            (self.logits, self.labels, self.number_token_mask), "Answer loss input"
        )
        _require_finite(self.logits, "answer logits")
        if self.logits.device.type != "meta":
            supervised = self.labels != -100
            vocab_size = self.logits.shape[-1]
            if bool(torch.any(supervised & ((self.labels < 0) | (self.labels >= vocab_size)))):
                raise ValueError("supervised answer labels must be within vocabulary")
            if bool(torch.any(self.number_token_mask & ~supervised)):
                raise ValueError("number_token_mask must be a subset of supervised labels")
            if bool(torch.any(self.number_token_mask[:, 0])):
                raise ValueError("the first label cannot be predicted by causal shift")
        if self.reader_counts is not None:
            if self.reader_counts.predicted_counts.shape[0] != self.logits.shape[0]:
                raise ValueError("Reader count metric batch size must match Answer loss")
            if self.reader_counts.predicted_counts.device != self.logits.device:
                raise ValueError("Reader count metric must share the Answer device")


@dataclass(frozen=True, slots=True)
class AnswerLossOutput:
    loss: LossTerm
    teacher_forced_token_accuracy: LossTerm
    number_token_accuracy: LossTerm
    answer_exact_match: LossTerm
    reader_exact_count_accuracy: LossTerm


def compute_answer_loss(inputs: AnswerLossInput) -> AnswerLossOutput:
    shift_logits = inputs.logits[:, :-1].float()
    shift_labels = inputs.labels[:, 1:]
    valid = shift_labels != -100
    safe_labels = torch.where(valid, shift_labels, torch.zeros_like(shift_labels))
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        safe_labels.reshape(-1),
        reduction="none",
    ).reshape_as(shift_labels)
    mask_counts = torch.full(
        (shift_logits.shape[0],),
        shift_logits.shape[1],
        dtype=torch.int64,
        device=shift_logits.device,
    )
    loss_reasons = tuple(
        None if bool(valid[row].any().item()) else LossSkipReason.NO_ANSWER_TOKEN
        for row in range(valid.shape[0])
    )
    loss = _reduce_items(token_losses, valid, mask_counts, loss_reasons)

    predictions = shift_logits.argmax(dim=-1)
    correct = predictions == shift_labels
    token_accuracy = _metric_from_items(
        correct,
        valid,
        mask_counts,
        LossSkipReason.NO_ANSWER_TOKEN,
    )
    number_mask = inputs.number_token_mask[:, 1:] & valid
    number_counts = number_mask.sum(dim=1, dtype=torch.int64)
    number_accuracy = _metric_from_items(
        correct,
        number_mask,
        number_counts,
        LossSkipReason.NO_NUMBER_TOKEN,
    )
    supervised_counts = valid.sum(dim=1, dtype=torch.int64)
    exact_per_row = ((~valid) | correct).all(dim=1).to(torch.float32)
    exact_per_row = torch.where(
        supervised_counts > 0, exact_per_row, torch.zeros_like(exact_per_row)
    )
    exact_reasons = tuple(
        None if int(count.item()) > 0 else LossSkipReason.NO_ANSWER_TOKEN
        for count in supervised_counts
    )
    exact_match = _make_term_from_rows(
        exact_per_row,
        (supervised_counts > 0).to(torch.int64),
        torch.ones_like(supervised_counts),
        exact_reasons,
    )
    if inputs.reader_counts is None:
        reader_accuracy = _invalid_term(
            inputs.logits.shape[0], inputs.logits, LossSkipReason.NO_READER_COUNT
        )
    else:
        reader = inputs.reader_counts
        reader_correct = reader.predicted_counts == reader.target_counts
        counts = reader.valid_mask.to(torch.int64)
        per_row = (reader_correct & reader.valid_mask).to(torch.float32)
        reasons = tuple(
            None if bool(value.item()) else LossSkipReason.NO_READER_COUNT
            for value in reader.valid_mask
        )
        reader_accuracy = _make_term_from_rows(
            per_row,
            counts,
            torch.ones_like(counts),
            reasons,
        )
    return AnswerLossOutput(
        loss=loss,
        teacher_forced_token_accuracy=token_accuracy,
        number_token_accuracy=number_accuracy,
        answer_exact_match=exact_match,
        reader_exact_count_accuracy=reader_accuracy,
    )


@dataclass(frozen=True, slots=True)
class OuterLossInput:
    answer_after: AnswerLossOutput
    state_after: StateLossOutput
    support_ttt: tuple[TTTLossOutput, ...]

    def __post_init__(self) -> None:
        if self.answer_after.loss.value.device != self.state_after.total.device:
            raise ValueError("after-update Answer and State losses must share one device")
        for support in self.support_ttt:
            if support.total.device != self.state_after.total.device:
                raise ValueError("support TTT losses must share the after-update device")


@dataclass(frozen=True, slots=True)
class OuterLossOutput:
    answer_after: Tensor
    state_after: Tensor
    auxiliary_ttt: LossTerm
    outer: Tensor
    total: Tensor
    auxiliary_weight: float = _AUXILIARY_OUTER_WEIGHT

    def __post_init__(self) -> None:
        for value, name in (
            (self.answer_after, "outer answer_after"),
            (self.state_after, "outer state_after"),
            (self.outer, "outer loss"),
            (self.total, "total loss"),
        ):
            _require_fp32_scalar(value, name)
            _require_finite(value, name)
        if self.auxiliary_weight != 0.1:
            raise ValueError("P14 auxiliary outer weight is frozen at 0.1")


def compute_outer_loss(inputs: OuterLossInput) -> OuterLossOutput:
    return compose_outer_loss_terms(
        answer_after=inputs.answer_after.loss.value,
        state_after=inputs.state_after.total,
        support_ttt=inputs.support_ttt,
    )


def compose_outer_loss_terms(
    *,
    answer_after: Tensor,
    state_after: Tensor,
    support_ttt: tuple[TTTLossOutput, ...],
) -> OuterLossOutput:
    """Compose already-reduced Answer/State terms without changing support-TTT semantics."""

    if answer_after.device != state_after.device:
        raise ValueError("composed outer Answer and State losses must share one device")
    auxiliary = _support_ttt_term(support_ttt, answer_after + state_after)
    outer = answer_after + state_after
    total = outer + _AUXILIARY_OUTER_WEIGHT * auxiliary.value
    return OuterLossOutput(
        answer_after=answer_after,
        state_after=state_after,
        auxiliary_ttt=auxiliary,
        outer=outer,
        total=total,
    )


@dataclass(frozen=True, slots=True)
class TrainingLossInput:
    """One current TTT batch plus optional earlier/additional support TTT outputs."""

    ttt: TTTLossInput
    state_after: StateLossInput
    answer_after: AnswerLossInput
    support_ttt: tuple[TTTLossOutput, ...]


@dataclass(frozen=True, slots=True)
class TrainingLossOutput:
    ttt: TTTLossOutput
    state: StateLossOutput
    answer: AnswerLossOutput
    outer: OuterLossOutput
    total: Tensor

    def __post_init__(self) -> None:
        _require_fp32_scalar(self.total, "training total")
        _require_finite(self.total, "training total")


def compute_losses(
    inputs: TrainingLossInput | None = None,
    *,
    predictor: TemporalPredictor | None = None,
) -> TrainingLossOutput:
    """Compute the complete P14 objective from explicit typed inputs.

    The no-argument guard preserves the staged-entrypoint audit: callers must supply typed inputs
    and the registered Predictor rather than receive fabricated labels.
    """

    if inputs is None:
        raise ValueError("compute_losses requires explicit TrainingLossInput")
    if predictor is None:
        raise ValueError("compute_losses requires the registered TemporalPredictor")
    ttt = compute_ttt_loss(predictor, inputs.ttt)
    state = compute_state_loss(inputs.state_after)
    answer = compute_answer_loss(inputs.answer_after)
    outer = compute_outer_loss(
        OuterLossInput(
            answer_after=answer,
            state_after=state,
            support_ttt=(ttt, *inputs.support_ttt),
        )
    )
    return TrainingLossOutput(ttt=ttt, state=state, answer=answer, outer=outer, total=outer.total)


def _compute_o1_state_term(target: O1StateTarget) -> LossTerm:
    losses = F.binary_cross_entropy_with_logits(
        target.logits.float(), target.targets.detach().float(), reduction="none"
    ).mean(dim=-1)
    return _reduce_dense_target(losses, target.slot_mask)


def _compute_o2_state_term(target: O2StateTarget) -> LossTerm:
    cosine = 1.0 - (
        target.identity_predictions.float() * target.identity_targets.detach().float()
    ).sum(dim=-1)
    score = F.binary_cross_entropy_with_logits(
        target.score_logits.float(), target.score_targets.detach().float(), reduction="none"
    ).mean(dim=-1)
    return _reduce_dense_target(cosine + score, target.slot_mask)


def _compute_e1_state_term(target: E1StateTarget) -> LossTerm:
    losses = F.binary_cross_entropy_with_logits(
        target.logits.float(), target.targets.detach().float(), reduction="none"
    ).mean(dim=-1)
    return _reduce_dense_target(losses, target.time_mask)


def _compute_e2_state_term(target: E2StateTarget) -> LossTerm:
    event = F.binary_cross_entropy_with_logits(
        target.event_logits.float(), target.event_targets.detach().float(), reduction="none"
    ).mean(dim=-1)
    safe_phase = torch.where(
        target.time_mask, target.phase_targets, torch.zeros_like(target.phase_targets)
    )
    phase = F.cross_entropy(
        target.phase_logits.float().reshape(-1, 4),
        safe_phase.reshape(-1),
        reduction="none",
    ).reshape_as(target.phase_targets)
    return _reduce_dense_target(event + phase, target.time_mask)


def _compute_operator_term(inputs: OperatorLossInput) -> LossTerm:
    safe_targets = torch.where(inputs.valid_mask, inputs.targets, torch.zeros_like(inputs.targets))
    losses = F.cross_entropy(inputs.logits.float(), safe_targets, reduction="none")
    counts = inputs.valid_mask.to(torch.int64)
    reasons = tuple(
        None if bool(value.item()) else LossSkipReason.NO_OPERATOR_LABEL
        for value in inputs.valid_mask
    )
    return _make_term_from_rows(
        torch.where(inputs.valid_mask, losses, torch.zeros_like(losses)),
        counts,
        torch.ones_like(counts),
        reasons,
    )


def _compute_retrieval_term(inputs: RetrievalLossInput) -> LossTerm:
    mask = inputs.present_mask & inputs.label_mask
    losses = F.binary_cross_entropy_with_logits(
        inputs.logits.float(), inputs.targets.detach().float(), reduction="none"
    )
    mask_counts = inputs.present_mask.sum(dim=1, dtype=torch.int64)
    reasons = tuple(
        None if bool(mask[row].any().item()) else LossSkipReason.NO_RETRIEVAL_LABEL
        for row in range(mask.shape[0])
    )
    return _reduce_items(losses, mask, mask_counts, reasons)


def _compute_time_loss(inputs: TimeLossInput) -> TimeLossOutput:
    mode_targets = torch.where(
        inputs.mode_valid_mask, inputs.mode_targets, torch.zeros_like(inputs.mode_targets)
    )
    mode_losses = F.cross_entropy(inputs.mode_logits.float(), mode_targets, reduction="none")
    mode_counts = inputs.mode_valid_mask.to(torch.int64)
    mode_reasons = tuple(
        None if bool(value.item()) else LossSkipReason.NO_TIME_LABEL
        for value in inputs.mode_valid_mask
    )
    mode = _make_term_from_rows(
        torch.where(inputs.mode_valid_mask, mode_losses, torch.zeros_like(mode_losses)),
        mode_counts,
        torch.ones_like(mode_counts),
        mode_reasons,
    )
    span_valid = inputs.span_start_targets != -100
    start = _masked_span_ce(
        inputs.span_start_logits,
        inputs.span_start_targets,
        inputs.token_valid_mask,
        span_valid,
    )
    end = _masked_span_ce(
        inputs.span_end_logits,
        inputs.span_end_targets,
        inputs.token_valid_mask,
        span_valid,
    )
    return TimeLossOutput(
        mode=mode,
        start=start,
        end=end,
        total=mode.value + start.value + end.value,
        per_row_total=mode.per_row + start.per_row + end.per_row,
        row_valid_mask=mode.row_valid_mask | start.row_valid_mask | end.row_valid_mask,
    )


def _masked_span_ce(
    logits: Tensor, targets: Tensor, token_mask: Tensor, row_valid: Tensor
) -> LossTerm:
    batch_size = logits.shape[0]
    losses = torch.zeros(batch_size, dtype=torch.float32, device=logits.device)
    valid_rows = torch.nonzero(row_valid, as_tuple=False).flatten()
    if valid_rows.numel():
        selected_logits = logits.index_select(0, valid_rows).float()
        selected_mask = token_mask.index_select(0, valid_rows)
        masked_logits = selected_logits.masked_fill(~selected_mask, -torch.inf)
        selected_targets = targets.index_select(0, valid_rows)
        selected_losses = F.cross_entropy(masked_logits, selected_targets, reduction="none")
        losses = losses.index_copy(0, valid_rows, selected_losses)
    counts = row_valid.to(torch.int64)
    reasons = tuple(
        None if bool(value.item()) else LossSkipReason.NO_SPAN_LABEL for value in row_valid
    )
    return _make_term_from_rows(losses, counts, torch.ones_like(counts), reasons)


def _reduce_dense_target(losses: Tensor, mask: Tensor) -> LossTerm:
    counts = mask.sum(dim=1, dtype=torch.int64)
    reasons = tuple(
        None if int(count.item()) > 0 else LossSkipReason.NO_VALID_LABEL for count in counts
    )
    return _reduce_items(losses, mask, torch.full_like(counts, mask.shape[1]), reasons)


def _scatter_term(local: LossTerm, row_indices: Tensor, batch_size: int) -> LossTerm:
    device = local.value.device
    per_row = torch.zeros(batch_size, dtype=torch.float32, device=device).index_copy(
        0, row_indices, local.per_row
    )
    valid_counts = torch.zeros(batch_size, dtype=torch.int64, device=device).index_copy(
        0, row_indices, local.valid_counts
    )
    mask_counts = torch.zeros(batch_size, dtype=torch.int64, device=device).index_copy(
        0, row_indices, local.mask_counts
    )
    reasons: list[LossSkipReason | None] = [LossSkipReason.NOT_APPLICABLE] * batch_size
    for local_row, global_row in enumerate(row_indices.tolist()):
        reasons[global_row] = local.skip_reasons[local_row]
    return LossTerm(
        value=local.value,
        per_row=per_row,
        row_valid_mask=valid_counts > 0,
        valid_counts=valid_counts,
        mask_counts=mask_counts,
        skip_reasons=tuple(reasons),
    )


def _support_ttt_term(support: tuple[TTTLossOutput, ...], reference: Tensor) -> LossTerm:
    if not support:
        return _invalid_term(0, reference, LossSkipReason.NO_VALID_SUPPORT)
    per_row = torch.cat(tuple(item.per_row_total for item in support), dim=0)
    valid = torch.cat(tuple(item.update_valid_mask for item in support), dim=0)
    per_row = torch.where(valid, per_row, torch.zeros_like(per_row))
    counts = valid.to(torch.int64)
    reasons = tuple(
        None if bool(value.item()) else LossSkipReason.NO_VALID_SUPPORT for value in valid
    )
    return _make_term_from_rows(per_row, counts, torch.ones_like(counts), reasons)


def _metric_from_items(
    correct: Tensor,
    valid_mask: Tensor,
    mask_counts: Tensor,
    invalid_reason: LossSkipReason,
) -> LossTerm:
    counts = valid_mask.sum(dim=1, dtype=torch.int64)
    per_row = (correct & valid_mask).sum(dim=1).float() / counts.clamp_min(1).float()
    per_row = torch.where(counts > 0, per_row, torch.zeros_like(per_row))
    reasons = tuple(None if int(count.item()) > 0 else invalid_reason for count in counts)
    return _make_term_from_rows(per_row, counts, mask_counts, reasons)


def _reduce_overlap_items(losses: Tensor, pair_mask: Tensor, valid_mask: Tensor) -> LossTerm:
    mask_counts = pair_mask.sum(dim=1, dtype=torch.int64)
    reasons = tuple(
        None if bool(valid_mask[row].any().item()) else LossSkipReason.NO_ALIGNED_EVENT
        for row in range(valid_mask.shape[0])
    )
    return _reduce_items(losses, valid_mask, mask_counts, reasons)


def _reduce_items(
    item_losses: Tensor,
    valid_mask: Tensor,
    mask_counts: Tensor,
    reasons: tuple[LossSkipReason | None, ...],
) -> LossTerm:
    if item_losses.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
        raise ValueError("item losses and bool valid mask must share [B, N]")
    losses = item_losses.float()
    _require_finite(losses, "item losses")
    valid_counts = valid_mask.sum(dim=1, dtype=torch.int64)
    per_row = (losses * valid_mask).sum(dim=1) / valid_counts.clamp_min(1).float()
    per_row = torch.where(valid_counts > 0, per_row, torch.zeros_like(per_row))
    return _make_term_from_rows(per_row, valid_counts, mask_counts, reasons)


def _make_term_from_rows(
    per_row: Tensor,
    valid_counts: Tensor,
    mask_counts: Tensor,
    reasons: tuple[LossSkipReason | None, ...],
) -> LossTerm:
    per_row_fp32 = per_row.float()
    valid = valid_counts > 0
    value = per_row_fp32[valid].mean() if bool(valid.any().item()) else per_row_fp32.sum() * 0.0
    return LossTerm(
        value=value,
        per_row=per_row_fp32,
        row_valid_mask=valid,
        valid_counts=valid_counts,
        mask_counts=mask_counts,
        skip_reasons=reasons,
    )


def _invalid_term(batch_size: int, reference: Tensor, reason: LossSkipReason) -> LossTerm:
    zero = _differentiable_zero(reference)
    per_row = torch.zeros(batch_size, dtype=torch.float32, device=reference.device) + zero
    counts = torch.zeros(batch_size, dtype=torch.int64, device=reference.device)
    return LossTerm(
        value=zero,
        per_row=per_row,
        row_valid_mask=torch.zeros(batch_size, dtype=torch.bool, device=reference.device),
        valid_counts=counts,
        mask_counts=counts.clone(),
        skip_reasons=tuple(reason for _ in range(batch_size)),
    )


def _invalid_time_output(batch_size: int, reference: Tensor) -> TimeLossOutput:
    mode = _invalid_term(batch_size, reference, LossSkipReason.NO_TIME_LABEL)
    start = _invalid_term(batch_size, reference, LossSkipReason.NO_SPAN_LABEL)
    end = _invalid_term(batch_size, reference, LossSkipReason.NO_SPAN_LABEL)
    return TimeLossOutput(
        mode=mode,
        start=start,
        end=end,
        total=mode.value + start.value + end.value,
        per_row_total=mode.per_row + start.per_row + end.per_row,
        row_valid_mask=mode.row_valid_mask,
    )


def _validate_dense_binary_target(
    row_indices: Tensor,
    logits: Tensor,
    targets: Tensor,
    mask: Tensor,
    width: int,
    name: str,
) -> None:
    rows = logits.shape[0] if logits.ndim == 3 else -1
    _validate_row_indices(row_indices, rows, name)
    if logits.shape != (rows, logits.shape[1], width) or not torch.is_floating_point(logits):
        raise ValueError(f"{name} logits must be floating [R, N, {width}]")
    if targets.shape != logits.shape or not torch.is_floating_point(targets):
        raise ValueError(f"{name} targets must match logits as floating [R, N, {width}]")
    if mask.shape != logits.shape[:2] or mask.dtype != torch.bool:
        raise ValueError(f"{name} mask must be bool [R, N]")
    _require_same_device((row_indices, logits, targets, mask), f"{name} State target")
    _require_finite(logits, f"{name} logits")
    _require_finite(targets, f"{name} targets")
    _require_probability_targets(targets, f"{name} targets")


def _validate_row_indices(indices: Tensor, rows: int, name: str) -> None:
    if indices.shape != (rows,) or indices.dtype != torch.int64:
        raise ValueError(f"{name} row_indices must be int64 [R]")
    if indices.device.type != "meta" and len(set(indices.tolist())) != rows:
        raise ValueError(f"{name} row_indices must be unique")


def _validate_overlap_probabilities(
    current: Tensor,
    previous: Tensor,
    pair_mask: Tensor,
    alignment_mask: Tensor,
    current_positions: Tensor,
    previous_positions: Tensor,
    current_timestamps: Tensor,
    previous_timestamps: Tensor,
    *,
    width: int,
    name: str,
    phase: bool,
) -> None:
    if current.ndim != 3 or current.shape[-1] != width or not torch.is_floating_point(current):
        raise ValueError(f"{name} current probabilities must be floating [B, M, {width}]")
    if previous.shape != current.shape or not torch.is_floating_point(previous):
        raise ValueError(f"{name} previous target probabilities must match current")
    shape = current.shape[:2]
    if (
        pair_mask.shape != shape
        or alignment_mask.shape != shape
        or pair_mask.dtype != torch.bool
        or alignment_mask.dtype != torch.bool
    ):
        raise ValueError(f"{name} pair/alignment masks must be bool [B, M]")
    if current_positions.shape != shape or previous_positions.shape != shape:
        raise ValueError(f"{name} overlap positions must be [B, M]")
    if current_positions.dtype != torch.int64 or previous_positions.dtype != torch.int64:
        raise TypeError(f"{name} overlap positions must use int64 dtype")
    if current_timestamps.shape != shape or previous_timestamps.shape != shape:
        raise ValueError(f"{name} overlap timestamps must be [B, M]")
    if not torch.is_floating_point(current_timestamps) or not torch.is_floating_point(
        previous_timestamps
    ):
        raise TypeError(f"{name} overlap timestamps must be floating")
    tensors = (
        current,
        previous,
        pair_mask,
        alignment_mask,
        current_positions,
        previous_positions,
        current_timestamps,
        previous_timestamps,
    )
    _require_same_device(tensors, f"{name} consistency input")
    _require_finite(current, f"{name} current probabilities")
    _require_finite(previous, f"{name} target probabilities")
    _require_finite(current_timestamps, f"{name} current timestamps")
    _require_finite(previous_timestamps, f"{name} previous timestamps")
    _require_probability_targets(current, f"{name} current probabilities")
    _require_probability_targets(previous, f"{name} target probabilities")
    if current.device.type == "meta":
        return
    if bool(torch.any(alignment_mask & ~pair_mask)):
        raise ValueError(f"{name} alignment_mask must be a subset of pair_mask")
    if bool(torch.any(alignment_mask & (current_positions != previous_positions))):
        raise ValueError(f"{name} aligned positions must be identical")
    aligned_time_difference = (current_timestamps - previous_timestamps).abs()
    if bool(torch.any(alignment_mask & (aligned_time_difference > 1.0e-6))):
        raise ValueError(f"{name} aligned timestamps must be equal")
    for row in range(current.shape[0]):
        current_aligned = current_positions[row, alignment_mask[row]].tolist()
        previous_aligned = previous_positions[row, alignment_mask[row]].tolist()
        if len(set(current_aligned)) != len(current_aligned) or len(set(previous_aligned)) != len(
            previous_aligned
        ):
            raise ValueError(f"{name} aligned positions must be unique per row")
    if phase:
        for tensor, label in ((current, "current"), (previous, "target")):
            selected = tensor[alignment_mask]
            if selected.numel() and not torch.allclose(
                selected.float().sum(dim=-1),
                torch.ones(selected.shape[0], device=selected.device),
                atol=_NORM_ATOL,
                rtol=_NORM_ATOL,
            ):
                raise ValueError(f"{name} {label} distributions must sum to one")


def _require_unit_norm(values: Tensor, mask: Tensor, name: str) -> None:
    if values.device.type == "meta":
        return
    selected = values[mask]
    if not selected.numel():
        return
    norms = torch.linalg.vector_norm(selected.float(), dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=_NORM_ATOL,
        rtol=_NORM_ATOL,
    ):
        raise ValueError(f"{name} must be unit L2 normalized")


def _require_probability_targets(values: Tensor, name: str) -> None:
    if values.device.type != "meta" and bool(torch.any((values < 0.0) | (values > 1.0))):
        raise ValueError(f"{name} must stay within [0, 1]")


def _state_reference(inputs: StateLossInput) -> Tensor:
    for component in (
        inputs.o1,
        inputs.o2,
        inputs.e1,
        inputs.e2,
        inputs.operator,
        inputs.retrieval,
        inputs.time,
    ):
        if component is not None:
            return _component_reference(component)
    raise AssertionError("StateLossInput validation requires a component")


def _component_reference(component: object) -> Tensor:
    if isinstance(component, O1StateTarget | E1StateTarget):
        return component.logits
    if isinstance(component, O2StateTarget):
        return component.identity_predictions
    if isinstance(component, E2StateTarget):
        return component.event_logits
    if isinstance(component, OperatorLossInput):
        return component.logits
    if isinstance(component, RetrievalLossInput):
        return component.logits
    if isinstance(component, TimeLossInput):
        return component.mode_logits
    raise TypeError("unsupported State loss component")


def _component_batch_size(component: object) -> int:
    reference = _component_reference(component)
    return int(reference.shape[0])


def _differentiable_zero(reference: Tensor) -> Tensor:
    return reference.float().sum() * 0.0


def _require_fp32_scalar(value: Tensor, name: str) -> None:
    if value.ndim != 0 or value.dtype != torch.float32:
        raise ValueError(f"{name} must be an FP32 scalar tensor")


def _require_finite(value: Tensor, name: str) -> None:
    if value.device.type != "meta" and not bool(torch.isfinite(value.detach()).all().item()):
        raise ValueError(f"{name} must be finite")


def _require_same_device(tensors: tuple[Tensor, ...], name: str) -> None:
    if tensors and any(tensor.device != tensors[0].device for tensor in tensors[1:]):
        raise ValueError(f"{name} tensors must share one device")
