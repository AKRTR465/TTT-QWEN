"""Closed-form parallel delta-rule writes for the per-video slot memory.

Inputs: one :class:`FastMemoryState` per video and one adapter-produced
:class:`MemoryWriteBatch` of unit slot keys/values, gated write gains, and one
forget factor.
Outputs: a new next-chunk memory generation plus finite write audits.
Forbidden: optimizer state, gradient steps, arbitrary parameter lists, in-place
mutation, Bank/FSM logic, or silent meta-gradient detaches.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor

from ttt_svcbench_qwen.fast_ttt import FastMemoryState, MemoryWriteBatch, SlotGeometryProbe
from ttt_svcbench_qwen.tensor_contracts import tensor_storage_key

_COSINE_EPSILON = 1.0e-12


class MemoryWriteSkipReason(StrEnum):
    NO_VALID_SLOT = "no_valid_slot"
    NONFINITE_KEY_VALUE = "nonfinite_key_value"


class GradientMode(StrEnum):
    """Exact autograd contract used by one delta-rule write."""

    ONLINE_LEAF = "online_leaf"
    META_LINEAR_RECURRENCE = "meta_linear_recurrence"


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    fast_state: FastMemoryState
    did_write: bool
    slots_written: int
    eta_sum: float
    eta_renormalized: bool
    pre_write_cosine_mean: float
    post_write_cosine_mean: float
    write_norm: float
    memory_norm: float
    skip_reason: MemoryWriteSkipReason | None
    skip_detail: str | None
    gradient_mode: GradientMode
    key_pairwise_cosine_mean: float = 0.0
    value_pairwise_cosine_mean: float = 0.0
    delta_pairwise_cosine_mean: float = 0.0
    slot_geometry: SlotGeometryProbe | None = None

    def __post_init__(self) -> None:
        if type(self.did_write) is not bool or type(self.eta_renormalized) is not bool:
            raise TypeError("memory write result flags must be bool")
        if not isinstance(self.gradient_mode, GradientMode):
            raise TypeError("memory write result gradient_mode must be GradientMode")
        if self.slot_geometry is not None and not isinstance(
            self.slot_geometry, SlotGeometryProbe
        ):
            raise TypeError("memory write slot_geometry must be a SlotGeometryProbe")
        if type(self.slots_written) is not int or self.slots_written < 0:
            raise ValueError("slots_written must be a non-negative exact integer")
        values = (
            self.eta_sum,
            self.write_norm,
            self.memory_norm,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("memory write scalar audits must be finite and non-negative")
        cosines = (
            self.pre_write_cosine_mean,
            self.post_write_cosine_mean,
            self.key_pairwise_cosine_mean,
            self.value_pairwise_cosine_mean,
            self.delta_pairwise_cosine_mean,
        )
        if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in cosines):
            raise ValueError("memory write cosine audits must be finite in [-1, 1]")
        if self.did_write:
            if self.skip_reason is not None or self.skip_detail is not None:
                raise ValueError("successful writes cannot carry skip metadata")
            if self.slots_written <= 0:
                raise ValueError("successful writes require at least one valid slot")
        else:
            if self.skip_reason is None or not self.skip_detail:
                raise ValueError("skipped writes require a reason and detail")
            zero_audits = (
                float(self.slots_written),
                self.eta_sum,
                self.write_norm,
                self.key_pairwise_cosine_mean,
                self.value_pairwise_cosine_mean,
                self.delta_pairwise_cosine_mean,
            )
            if any(value != 0.0 for value in zero_audits):
                raise ValueError("skipped writes must report zero slot/eta/write audits")
        mode_is_differentiable = self.gradient_mode is GradientMode.META_LINEAR_RECURRENCE
        if mode_is_differentiable is not self.fast_state.differentiable:
            raise ValueError("memory write gradient mode must match FastMemoryState")


@dataclass(frozen=True, slots=True)
class MemoryTruncateAudit:
    """Detached evidence for one truncated-meta memory boundary."""

    write_version: int
    write_count: int
    skip_count: int
    max_abs_value_drift: float
    old_graph_truncated: bool
    storage_isolated: bool

    def __post_init__(self) -> None:
        counters = (self.write_version, self.write_count, self.skip_count)
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("memory truncation counters must be non-negative integers")
        if self.write_version != self.write_count:
            raise ValueError("memory truncation version must equal accepted writes")
        if not math.isfinite(self.max_abs_value_drift) or self.max_abs_value_drift != 0.0:
            raise ValueError("memory truncation must preserve values bitwise")
        if not self.old_graph_truncated:
            raise ValueError("memory truncation must cut the old inner graph")
        if not self.storage_isolated:
            raise ValueError("memory truncation tensors must use isolated storage")


def apply_memory_write(
    *,
    fast_state: FastMemoryState,
    batch: MemoryWriteBatch,
    row: int,
) -> MemoryWriteResult:
    """Return one new memory generation without mutating the current chunk state."""

    if not isinstance(fast_state, FastMemoryState):
        raise TypeError("memory write requires one FastMemoryState")
    if not isinstance(batch, MemoryWriteBatch):
        raise TypeError("memory write requires MemoryWriteBatch")
    batch_size = int(batch.keys.shape[0])
    if type(row) is not int or row < 0 or row >= batch_size:
        raise IndexError("memory write row must be an exact in-range batch index")
    if batch.keys.device != fast_state.m.device:
        raise ValueError("memory write payload and state must share one device")
    gradient_mode = (
        GradientMode.META_LINEAR_RECURRENCE
        if fast_state.differentiable
        else GradientMode.ONLINE_LEAF
    )
    mask = batch.slot_mask[row]
    slot_count = int(mask.sum().item())
    if slot_count == 0:
        return _skip_result(
            fast_state,
            reason=MemoryWriteSkipReason.NO_VALID_SLOT,
            detail="chunk_produced_no_valid_slot",
            gradient_mode=gradient_mode,
        )
    keys = batch.keys[row]
    values = batch.values[row]
    etas = batch.etas[row]
    row_payload_finite = bool(
        torch.isfinite(keys).all()
        and torch.isfinite(values).all()
        and torch.isfinite(etas).all()
        and torch.isfinite(batch.beta).all()
    )
    if not row_payload_finite:
        return _skip_result(
            fast_state,
            reason=MemoryWriteSkipReason.NONFINITE_KEY_VALUE,
            detail="write_payload_is_not_finite",
            gradient_mode=gradient_mode,
        )

    write_context = (
        torch.enable_grad() if fast_state.differentiable else torch.no_grad()
    )
    with write_context:
        recall = keys @ fast_state.m.transpose(0, 1)
        delta = values - recall
        update = torch.einsum("s,sd,se->de", etas, delta, keys)
        next_m = (1.0 - batch.beta) * fast_state.m + update
    if not bool(torch.isfinite(next_m.detach()).all()):
        return _skip_result(
            fast_state,
            reason=MemoryWriteSkipReason.NONFINITE_KEY_VALUE,
            detail="memory_after_write_is_not_finite",
            gradient_mode=gradient_mode,
        )
    with torch.no_grad():
        pre_write_cosine = _masked_cosine_mean(recall.detach(), values.detach(), mask)
        post_recall = keys.detach() @ next_m.detach().transpose(0, 1)
        post_write_cosine = _masked_cosine_mean(post_recall, values.detach(), mask)
        write_norm = float(
            torch.linalg.matrix_norm(next_m.detach() - fast_state.m.detach()).cpu().item()
        )
        memory_norm = float(torch.linalg.matrix_norm(next_m.detach()).cpu().item())
        eta_sum = float(etas.detach().sum().cpu().item())
        key_pairwise_cosine = _pairwise_offdiag_cosine_mean(keys.detach(), mask)
        value_pairwise_cosine = _pairwise_offdiag_cosine_mean(values.detach(), mask)
        delta_pairwise_cosine = _pairwise_offdiag_cosine_mean(delta.detach(), mask)
    next_state = FastMemoryState(
        m=_next_memory(next_m, differentiable=fast_state.differentiable),
        write_version=fast_state.write_version + 1,
        write_count=fast_state.write_count + 1,
        skip_count=fast_state.skip_count,
        differentiable=fast_state.differentiable,
    )
    return MemoryWriteResult(
        fast_state=next_state,
        did_write=True,
        slots_written=slot_count,
        eta_sum=eta_sum,
        eta_renormalized=batch.eta_renormalized[row],
        pre_write_cosine_mean=pre_write_cosine,
        post_write_cosine_mean=post_write_cosine,
        write_norm=write_norm,
        memory_norm=memory_norm,
        skip_reason=None,
        skip_detail=None,
        gradient_mode=gradient_mode,
        key_pairwise_cosine_mean=key_pairwise_cosine,
        value_pairwise_cosine_mean=value_pairwise_cosine,
        delta_pairwise_cosine_mean=delta_pairwise_cosine,
        slot_geometry=(
            batch.slot_geometry_probes[row] if batch.slot_geometry_probes else None
        ),
    )


def apply_memory_writes(
    *,
    fast_states: Sequence[FastMemoryState],
    batch: MemoryWriteBatch,
) -> tuple[MemoryWriteResult, ...]:
    """Apply independent row writes to a storage-isolated batch of video states."""

    states = tuple(fast_states)
    if not all(isinstance(state, FastMemoryState) for state in states):
        raise TypeError("memory write batch bridge requires only FastMemoryState values")
    if not isinstance(batch, MemoryWriteBatch):
        raise TypeError("memory write batch bridge requires MemoryWriteBatch")
    if len(states) != int(batch.keys.shape[0]):
        raise ValueError("memory write batch and fast states must have identical B")
    if len({state.differentiable for state in states}) > 1:
        raise ValueError("one memory write batch cannot mix differentiable and online states")
    if len({tensor_storage_key(state.m) for state in states}) != len(states):
        raise ValueError("batched memory states must use storage-isolated tensors")
    return tuple(
        apply_memory_write(fast_state=state, batch=batch, row=row)
        for row, state in enumerate(states)
    )


def truncate_memory_state(
    state: FastMemoryState,
) -> tuple[FastMemoryState, MemoryTruncateAudit]:
    """Cut the K-step inner graph at a segment boundary, preserving values bitwise.

    Unlike the retired straight-through re-anchor there is no W0 ancestry to
    restore: the memory is zero-initialized per video, so the truncated leaf is
    the complete authoritative value.
    """

    if not isinstance(state, FastMemoryState):
        raise TypeError("truncate_memory_state requires one FastMemoryState")
    if not state.differentiable:
        raise ValueError("only differentiable meta-training memory may be truncated")
    next_m = state.m.detach().clone().requires_grad_(True)
    next_state = FastMemoryState(
        m=next_m,
        write_version=state.write_version,
        write_count=state.write_count,
        skip_count=state.skip_count,
        differentiable=True,
    )
    if state.m.device.type == "meta":
        drift = 0.0
    else:
        drift = float((next_m.detach() - state.m.detach()).abs().max().cpu().item())
    audit = MemoryTruncateAudit(
        write_version=state.write_version,
        write_count=state.write_count,
        skip_count=state.skip_count,
        max_abs_value_drift=drift,
        old_graph_truncated=next_m.grad_fn is None and next_m is not state.m,
        storage_isolated=tensor_storage_key(next_m) != tensor_storage_key(state.m),
    )
    return next_state, audit


def _skip_result(
    fast_state: FastMemoryState,
    *,
    reason: MemoryWriteSkipReason,
    detail: str,
    gradient_mode: GradientMode,
) -> MemoryWriteResult:
    next_state = FastMemoryState(
        m=_next_memory(fast_state.m, differentiable=fast_state.differentiable),
        write_version=fast_state.write_version,
        write_count=fast_state.write_count,
        skip_count=fast_state.skip_count + 1,
        differentiable=fast_state.differentiable,
    )
    with torch.no_grad():
        memory_norm = (
            0.0
            if fast_state.m.device.type == "meta"
            else float(torch.linalg.matrix_norm(fast_state.m.detach()).cpu().item())
        )
    return MemoryWriteResult(
        fast_state=next_state,
        did_write=False,
        slots_written=0,
        eta_sum=0.0,
        eta_renormalized=False,
        pre_write_cosine_mean=0.0,
        post_write_cosine_mean=0.0,
        write_norm=0.0,
        memory_norm=memory_norm,
        skip_reason=reason,
        skip_detail=detail,
        gradient_mode=gradient_mode,
    )


def _next_memory(value: Tensor, *, differentiable: bool) -> Tensor:
    if differentiable:
        return value.clone()
    return value.detach().clone().requires_grad_(True)


def _pairwise_offdiag_cosine_mean(rows: Tensor, mask: Tensor) -> float:
    """Mean off-diagonal cosine over the valid payload rows; a pure probe scalar."""

    selected = rows[mask]
    count = int(selected.shape[0])
    if count < 2:
        return 0.0
    normalized = selected / selected.norm(dim=-1, keepdim=True).clamp_min(_COSINE_EPSILON)
    gram = normalized @ normalized.transpose(0, 1)
    total = float(gram.sum().cpu().item())
    trace = float(gram.diagonal().sum().cpu().item())
    value = (total - trace) / float(count * (count - 1))
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def _masked_cosine_mean(predictions: Tensor, targets: Tensor, mask: Tensor) -> float:
    selected_predictions = predictions[mask]
    selected_targets = targets[mask]
    if selected_predictions.numel() == 0:
        return 0.0
    numerator = (selected_predictions * selected_targets).sum(dim=-1)
    denominator = (
        selected_predictions.norm(dim=-1) * selected_targets.norm(dim=-1)
    ).clamp_min(_COSINE_EPSILON)
    value = float((numerator / denominator).mean().cpu().item())
    if not math.isfinite(value):
        raise ValueError("memory write cosine audit must be finite")
    return max(-1.0, min(1.0, value))
