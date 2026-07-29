"""Bank-conditioned state-write Test-Time Training contracts.

The hard State Bank remains authoritative and detached.  Its pre-write semantic
records condition the Fast Adapter key, while the current model-predicted active
state-write head supplies the label-free inner target.  This module owns no
runtime state mutation and performs no optimizer step.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from ttt_svcbench_qwen.state_bank import HeadType, StateBankView

ASSOCIATIVE_CONTRACT = "bank_conditioned_state_write_v2"
ASSOCIATIVE_CONTRACT_VERSION = 2
BANK_EMBEDDING_DIM = 512
ASSOCIATIVE_DIM = 768
_NORMALIZE_EPSILON = 1.0e-6


class StateWriteSourceView(Protocol):
    """Minimal soft-write surface needed by the inner target selector."""

    @property
    def o1_present_mask(self) -> Tensor: ...

    @property
    def o2_present_mask(self) -> Tensor: ...

    @property
    def e1_present_mask(self) -> Tensor: ...

    @property
    def e2_present_mask(self) -> Tensor: ...

    @property
    def o1_sources(self) -> Tensor: ...

    @property
    def o2_sources(self) -> Tensor: ...

    @property
    def e1_sources(self) -> Tensor: ...

    @property
    def e2_sources(self) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class FastAssociativeContext:
    """Immutable, pre-write Bank context consumed by one visual forward."""

    combined_query: Tensor
    bank_record_counts: Tensor
    bank_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.combined_query.ndim != 2
            or self.combined_query.shape[0] <= 0
            or self.combined_query.shape[1] != BANK_EMBEDDING_DIM
            or not torch.is_floating_point(self.combined_query)
        ):
            raise ValueError("combined_query must be floating [B, 512]")
        if self.bank_record_counts.shape != self.combined_query.shape[:1]:
            raise ValueError("bank_record_counts must align to context batch")
        if self.bank_record_counts.dtype != torch.int64:
            raise TypeError("bank_record_counts must be int64")
        if self.bank_record_counts.device != self.combined_query.device:
            raise ValueError("associative context tensors must share one device")
        if len(self.bank_versions) != self.combined_query.shape[0]:
            raise ValueError("bank_versions must align to context batch")
        if any(type(version) is not int or version < 0 for version in self.bank_versions):
            raise ValueError("bank_versions must be non-negative integers")
        if bool(torch.any(self.bank_record_counts < 0)):
            raise ValueError("bank_record_counts must be non-negative")
        if self.combined_query.device.type != "meta" and not bool(
            torch.isfinite(self.combined_query).all()
        ):
            raise ValueError("combined_query must be finite")

    def to(self, reference: Tensor) -> FastAssociativeContext:
        """Move only the immutable tensor view to the visual reference."""

        return FastAssociativeContext(
            combined_query=self.combined_query.to(
                device=reference.device,
                dtype=reference.dtype,
            ),
            bank_record_counts=self.bank_record_counts.to(device=reference.device),
            bank_versions=self.bank_versions,
        )


@dataclass(frozen=True, slots=True)
class AssociativeTTTIntermediates:
    """Ephemeral key/prediction tensors captured by one adapter call."""

    keys: Tensor
    predictions: Tensor
    valid_mask: Tensor
    bank_record_counts: Tensor
    bank_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        shape = self.keys.shape
        if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] != ASSOCIATIVE_DIM:
            raise ValueError("associative keys must be non-empty [B, N, 768]")
        tensors = (self.keys, self.predictions)
        if any(tensor.shape != shape or not torch.is_floating_point(tensor) for tensor in tensors):
            raise ValueError("associative key/prediction tensors must align")
        if any(
            tensor.dtype != self.keys.dtype or tensor.device != self.keys.device
            for tensor in tensors[1:]
        ):
            raise ValueError("associative tensors must share dtype/device")
        if self.valid_mask.shape != shape[:2] or self.valid_mask.dtype != torch.bool:
            raise ValueError("associative valid_mask must be bool [B, N]")
        if self.valid_mask.device != self.keys.device:
            raise ValueError("associative valid_mask must share the tensor device")
        if (
            self.bank_record_counts.shape != shape[:1]
            or self.bank_record_counts.dtype != torch.int64
            or self.bank_record_counts.device != self.keys.device
        ):
            raise ValueError("bank_record_counts must be int64 [B] on the tensor device")
        if len(self.bank_versions) != shape[0]:
            raise ValueError("bank_versions must align to the associative batch")
        if self.keys.device.type != "meta" and any(
            not bool(torch.isfinite(tensor).all()) for tensor in tensors
        ):
            raise ValueError("associative tensors must be finite")


@dataclass(frozen=True, slots=True)
class AssociativeScaleAudit:
    key_sum_squares: Tensor
    value_sum_squares: Tensor
    prediction_sum_squares: Tensor
    error_sum_squares: Tensor
    element_count: Tensor
    key_max_abs: Tensor
    value_max_abs: Tensor
    prediction_max_abs: Tensor
    error_max_abs: Tensor

    def __post_init__(self) -> None:
        fp32 = (
            self.key_sum_squares,
            self.value_sum_squares,
            self.prediction_sum_squares,
            self.error_sum_squares,
            self.key_max_abs,
            self.value_max_abs,
            self.prediction_max_abs,
            self.error_max_abs,
        )
        if any(value.shape != () or value.dtype != torch.float32 for value in fp32):
            raise ValueError("associative scale values must be detached FP32 scalars")
        if self.element_count.shape != () or self.element_count.dtype != torch.int64:
            raise ValueError("associative element_count must be a detached int64 scalar")
        if any(
            value.requires_grad or value.grad_fn is not None
            for value in (*fp32, self.element_count)
        ):
            raise ValueError("associative scale audit must be detached")


@dataclass(frozen=True, slots=True)
class AssociativeTargetAudit:
    """Detached state-write target selection and cosine statistics."""

    active_head_counts: tuple[int, int, int, int]
    valid_target_counts: tuple[int, int, int, int]
    unsupported_count: int
    empty_target_count: int
    prediction_target_cosine_sum: Tensor
    prediction_target_cosine_count: Tensor

    def __post_init__(self) -> None:
        counts = (
            *self.active_head_counts,
            *self.valid_target_counts,
            self.unsupported_count,
            self.empty_target_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("associative target audit counts must be non-negative integers")
        if (
            self.prediction_target_cosine_sum.shape != ()
            or self.prediction_target_cosine_sum.dtype != torch.float32
            or self.prediction_target_cosine_count.shape != ()
            or self.prediction_target_cosine_count.dtype != torch.int64
        ):
            raise ValueError("associative target cosine audit must use detached scalars")
        if (
            self.prediction_target_cosine_sum.requires_grad
            or self.prediction_target_cosine_sum.grad_fn is not None
            or self.prediction_target_cosine_count.requires_grad
            or self.prediction_target_cosine_count.grad_fn is not None
        ):
            raise ValueError("associative target cosine audit must be detached")


@dataclass(frozen=True, slots=True)
class AssociativeTTTLossOutput:
    total: Tensor
    per_row_total: Tensor
    update_valid_mask: Tensor
    valid_token_counts: Tensor
    scale_audit: AssociativeScaleAudit
    target_audit: AssociativeTargetAudit

    def __post_init__(self) -> None:
        batch_size = self.per_row_total.shape[0]
        if self.total.shape != () or self.total.dtype != torch.float32:
            raise ValueError("associative total must be an FP32 scalar")
        if self.per_row_total.ndim != 1 or self.per_row_total.dtype != torch.float32:
            raise ValueError("associative per_row_total must be FP32 [B]")
        if (
            self.update_valid_mask.shape != (batch_size,)
            or self.update_valid_mask.dtype != torch.bool
        ):
            raise ValueError("associative update_valid_mask must be bool [B]")
        if (
            self.valid_token_counts.shape != (batch_size,)
            or self.valid_token_counts.dtype != torch.int64
        ):
            raise ValueError("associative valid_token_counts must be int64 [B]")
        if any(
            value.device != self.total.device
            for value in (self.per_row_total, self.update_valid_mask, self.valid_token_counts)
        ):
            raise ValueError("associative loss outputs must share one device")


def build_fast_associative_context(
    query_target: Tensor,
    bank_view: StateBankView,
) -> FastAssociativeContext:
    """Pool every pre-write present+valid semantic record with parameter-free attention."""

    if (
        query_target.ndim != 2
        or query_target.shape[0] <= 0
        or query_target.shape[1] != BANK_EMBEDDING_DIM
        or not torch.is_floating_point(query_target)
    ):
        raise ValueError("query_target must be floating [B, 512]")
    if bank_view.embeddings.shape[0] != query_target.shape[0]:
        raise ValueError("State Bank view and query batch must align")
    embeddings = bank_view.embeddings.to(
        device=query_target.device,
        dtype=query_target.dtype,
    ).detach()
    valid = (
        bank_view.present_mask & bank_view.record_valid_mask
    ).to(device=query_target.device)
    counts = valid.sum(dim=1, dtype=torch.int64)
    if embeddings.shape[1] == 0:
        pooled = torch.zeros_like(query_target)
    else:
        scores = torch.einsum("bd,bnd->bn", query_target, embeddings)
        scores = scores / math.sqrt(BANK_EMBEDDING_DIM)
        row_nonempty = valid.any(dim=1, keepdim=True)
        safe_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        safe_scores = torch.where(row_nonempty, safe_scores, torch.zeros_like(safe_scores))
        weights = torch.softmax(safe_scores, dim=1) * valid.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = torch.einsum("bn,bnd->bd", weights, embeddings)
    combined = query_target + pooled
    return FastAssociativeContext(
        combined_query=combined,
        bank_record_counts=counts,
        bank_versions=bank_view.bank_versions,
    )


def compute_associative_ttt_loss(
    inputs: AssociativeTTTIntermediates,
    state_write: StateWriteSourceView,
    head_types: Sequence[HeadType | None],
) -> AssociativeTTTLossOutput:
    """Compute the sole inner loss against the predicted active-head write source."""

    batch_size = inputs.predictions.shape[0]
    heads = tuple(head_types)
    if len(heads) != batch_size:
        raise ValueError("active head metadata must align to the associative batch")
    targets, target_valid, target_audit_counts = _select_state_write_targets(
        state_write,
        heads,
        reference=inputs.predictions,
    )
    mask = inputs.valid_mask
    counts = mask.sum(dim=1, dtype=torch.int64)
    visual_valid = counts > 0
    valid_rows = visual_valid & target_valid
    effective_counts = torch.where(target_valid, counts, torch.zeros_like(counts))
    fp32_mask = mask.unsqueeze(-1).to(dtype=torch.float32)
    denominator = counts.clamp_min(1).to(dtype=torch.float32).unsqueeze(-1)
    pooled_keys = (inputs.keys.float() * fp32_mask).sum(dim=1) / denominator
    pooled_predictions = (
        (inputs.predictions.float() * fp32_mask).sum(dim=1) / denominator
    )
    normalized_predictions = _smooth_normalize(pooled_predictions)
    normalized_targets = _smooth_normalize(targets.detach().float())
    cosine = (normalized_predictions * normalized_targets).sum(dim=-1)
    per_row = torch.where(valid_rows, 1.0 - cosine, torch.zeros_like(cosine))
    total = per_row[valid_rows].mean() if bool(valid_rows.any().item()) else per_row.sum() * 0.0

    row_mask = valid_rows.unsqueeze(-1).to(dtype=torch.float32)
    keys = pooled_keys.detach() * row_mask
    values = normalized_targets.detach() * row_mask
    predictions = normalized_predictions.detach() * row_mask
    errors = (predictions - values).detach()
    element_count = (valid_rows.sum(dtype=torch.int64) * inputs.keys.shape[-1]).detach()
    zero = total.detach().new_zeros(())

    def max_abs(value: Tensor) -> Tensor:
        return value.abs().amax().detach() if value.numel() else zero.clone()

    audit = AssociativeScaleAudit(
        key_sum_squares=keys.square().sum().detach(),
        value_sum_squares=values.square().sum().detach(),
        prediction_sum_squares=predictions.square().sum().detach(),
        error_sum_squares=errors.square().sum().detach(),
        element_count=element_count,
        key_max_abs=max_abs(keys),
        value_max_abs=max_abs(values),
        prediction_max_abs=max_abs(predictions),
        error_max_abs=max_abs(errors),
    )
    active_counts, valid_counts, unsupported_count, empty_target_count = target_audit_counts
    target_audit = AssociativeTargetAudit(
        active_head_counts=active_counts,
        valid_target_counts=valid_counts,
        unsupported_count=unsupported_count,
        empty_target_count=empty_target_count,
        prediction_target_cosine_sum=cosine[valid_rows].detach().float().sum(),
        prediction_target_cosine_count=valid_rows.sum(dtype=torch.int64).detach(),
    )
    return AssociativeTTTLossOutput(
        total=total,
        per_row_total=per_row,
        update_valid_mask=valid_rows,
        valid_token_counts=effective_counts,
        scale_audit=audit,
        target_audit=target_audit,
    )


def _select_state_write_targets(
    state_write: StateWriteSourceView,
    head_types: tuple[HeadType | None, ...],
    *,
    reference: Tensor,
) -> tuple[
    Tensor,
    Tensor,
    tuple[tuple[int, int, int, int], tuple[int, int, int, int], int, int],
]:
    """Select one detached 768-d source per row from the predicted active head."""

    required = (
        "o1_sources",
        "o1_present_mask",
        "o2_sources",
        "o2_present_mask",
        "e1_sources",
        "e1_present_mask",
        "e2_sources",
        "e2_present_mask",
    )
    if state_write is None or any(not hasattr(state_write, name) for name in required):
        raise TypeError("state-write associative target requires StageASoftWriteOutput")
    batch_size = len(head_types)
    targets = torch.zeros(
        (batch_size, ASSOCIATIVE_DIM),
        dtype=torch.float32,
        device=reference.device,
    )
    valid = torch.zeros(batch_size, dtype=torch.bool, device=reference.device)
    head_order = (HeadType.O1, HeadType.O2, HeadType.E1, HeadType.E2)
    active_counts = [0, 0, 0, 0]
    valid_counts = [0, 0, 0, 0]
    unsupported_count = 0
    for row, head in enumerate(head_types):
        if head is None:
            unsupported_count += 1
            continue
        if head not in head_order:
            raise ValueError(f"unsupported associative active head: {head!r}")
        head_index = head_order.index(head)
        active_counts[head_index] += 1
        source, source_valid = _state_write_source_for_row(state_write, head, row)
        if not source_valid:
            continue
        if source.shape != (ASSOCIATIVE_DIM,) or not torch.is_floating_point(source):
            raise ValueError("active-head state-write source must be floating [768]")
        if source.device != reference.device:
            source = source.to(device=reference.device)
        detached = source.detach().float()
        if not bool(torch.isfinite(detached).all()):
            raise ValueError("active-head state-write source must be finite")
        targets[row] = detached
        valid[row] = True
        valid_counts[head_index] += 1
    empty_target_count = batch_size - sum(valid_counts) - unsupported_count
    active_count_tuple = (
        active_counts[0],
        active_counts[1],
        active_counts[2],
        active_counts[3],
    )
    valid_count_tuple = (
        valid_counts[0],
        valid_counts[1],
        valid_counts[2],
        valid_counts[3],
    )
    return (
        targets,
        valid,
        (
            active_count_tuple,
            valid_count_tuple,
            unsupported_count,
            empty_target_count,
        ),
    )


def _smooth_normalize(value: Tensor) -> Tensor:
    """Normalize with a finite first/second derivative at the zero vector."""

    inverse_norm = torch.rsqrt(
        value.square().sum(dim=-1, keepdim=True) + _NORMALIZE_EPSILON**2
    )
    return value * inverse_norm


def _state_write_source_for_row(
    state_write: StateWriteSourceView,
    head: HeadType,
    row: int,
) -> tuple[Tensor, bool]:
    if head is HeadType.O1:
        present = state_write.o1_present_mask[row]
        return state_write.o1_sources[row], bool(present.item())
    if head is HeadType.O2:
        present = state_write.o2_present_mask[row]
        sources = state_write.o2_sources[row]
        if not bool(present.any().item()):
            return sources.float().sum(dim=0) * 0.0, False
        fp32_mask = present.unsqueeze(-1).to(dtype=torch.float32)
        pooled = (sources.float() * fp32_mask).sum(dim=0) / present.sum().to(torch.float32)
        return pooled, True
    if head is HeadType.E1:
        present = state_write.e1_present_mask[row]
        return state_write.e1_sources[row], bool(present.any().item())
    if head is HeadType.E2:
        present = state_write.e2_present_mask[row]
        return state_write.e2_sources[row], bool(present.any().item())
    raise ValueError(f"unsupported associative active head: {head!r}")
