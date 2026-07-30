"""Bank-conditioned key-context contracts for the slot-memory Fast Adapter.

The hard State Bank remains authoritative and detached.  Its pre-write semantic
records condition the Fast Adapter key space; the per-video delta-rule memory in
``memory_write`` consumes the resulting token keys.  This module owns no runtime
state mutation and performs no memory write.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from ttt_svcbench_qwen.state_bank import StateBankView

ASSOCIATIVE_CONTRACT = "bank_conditioned_slot_memory_v3"
ASSOCIATIVE_CONTRACT_VERSION = 3
BANK_EMBEDDING_DIM = 512
ASSOCIATIVE_DIM = 768


class StateWriteSourceView(Protocol):
    """Minimal soft-write surface still produced by the heads for the Bank path."""

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
