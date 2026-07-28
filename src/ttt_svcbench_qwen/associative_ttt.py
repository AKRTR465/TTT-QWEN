"""Bank-conditioned associative Test-Time Training contracts.

The hard State Bank remains authoritative and detached.  Its pre-write semantic
records condition the Fast Adapter key, while the current raw Main Merger
tokens provide a label-free visual value target.  This module owns no runtime
state mutation and performs no optimizer step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ttt_svcbench_qwen.state_bank import StateBankView

ASSOCIATIVE_CONTRACT = "bank_conditioned_visual_v1"
ASSOCIATIVE_CONTRACT_VERSION = 1
BANK_EMBEDDING_DIM = 512
ASSOCIATIVE_DIM = 768


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
    """Ephemeral key/value/prediction tensors captured by one adapter call."""

    keys: Tensor
    values: Tensor
    predictions: Tensor
    valid_mask: Tensor
    bank_record_counts: Tensor
    bank_versions: tuple[int, ...]

    def __post_init__(self) -> None:
        shape = self.keys.shape
        if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] != ASSOCIATIVE_DIM:
            raise ValueError("associative keys must be non-empty [B, N, 768]")
        tensors = (self.keys, self.values, self.predictions)
        if any(tensor.shape != shape or not torch.is_floating_point(tensor) for tensor in tensors):
            raise ValueError("associative key/value/prediction tensors must align")
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
class AssociativeTTTLossOutput:
    total: Tensor
    per_row_total: Tensor
    update_valid_mask: Tensor
    valid_token_counts: Tensor
    scale_audit: AssociativeScaleAudit

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
) -> AssociativeTTTLossOutput:
    """Compute the sole inner loss: masked FP32 visual association MSE."""

    error = inputs.predictions.float() - inputs.values.float()
    item_losses = error.square().mean(dim=-1)
    mask = inputs.valid_mask
    counts = mask.sum(dim=1, dtype=torch.int64)
    valid_rows = counts > 0
    per_row = (
        (item_losses * mask.to(dtype=item_losses.dtype)).sum(dim=1)
        / counts.clamp_min(1).to(dtype=torch.float32)
    )
    total = per_row[valid_rows].mean() if bool(valid_rows.any().item()) else per_row.sum() * 0.0

    expanded_mask = mask.unsqueeze(-1).to(dtype=torch.float32)
    keys = inputs.keys.detach().float() * expanded_mask
    values = inputs.values.detach().float() * expanded_mask
    predictions = inputs.predictions.detach().float() * expanded_mask
    errors = error.detach() * expanded_mask
    element_count = (counts.sum() * inputs.keys.shape[-1]).detach()
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
    return AssociativeTTTLossOutput(
        total=total,
        per_row_total=per_row,
        update_valid_mask=valid_rows,
        valid_token_counts=counts,
        scale_audit=audit,
    )
