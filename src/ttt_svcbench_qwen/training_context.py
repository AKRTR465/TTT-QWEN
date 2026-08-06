"""Shared activation-lifetime controls for A2 and A5 training."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from types import TracebackType
from typing import cast

import torch

_DEFAULT_QUERY_OFFLOAD_MAX_GB = 8
_MINIMUM_OFFLOAD_TENSOR_BYTES = 1 << 20


@dataclass(slots=True)
class _OffloadedActivation:
    device: torch.device
    tensor: torch.Tensor | None
    budget: QueryActivationOffloadBudget
    nbytes: int
    restored_tensor: torch.Tensor | None = None
    released: bool = False

    def restore(self) -> torch.Tensor:
        if self.restored_tensor is not None:
            return self.restored_tensor
        if self.tensor is None:
            raise RuntimeError("offloaded Query activation has no restorable tensor")
        # Autograd is allowed to unpack one saved tensor more than once during a single
        # backward. Cache the first blocking device restore for those repeated reads. The
        # no-retain Query graph owns this wrapper, so both the cached device tensor and this
        # object disappear at the end of the Query backward.
        restored = self.tensor.to(self.device, non_blocking=False)
        self.tensor = None
        if not self.released:
            self.budget.release(self.nbytes)
            self.released = True
        self.restored_tensor = restored
        return self.restored_tensor

    def release(self) -> None:
        self.tensor = None
        self.restored_tensor = None
        if not self.released:
            self.budget.release(self.nbytes)
            self.released = True

    def __del__(self) -> None:
        self.release()


@dataclass(slots=True)
class QueryActivationOffloadBudget:
    maximum_bytes: int
    claimed_bytes: int = 0
    peak_claimed_bytes: int = 0
    total_claimed_bytes: int = 0

    def __post_init__(self) -> None:
        if type(self.maximum_bytes) is not int or self.maximum_bytes <= 0:
            raise ValueError("activation offload maximum_bytes must be a positive integer")

    @classmethod
    def from_environment(cls) -> QueryActivationOffloadBudget:
        return cls(maximum_bytes=_query_offload_budget_bytes())

    def claim(self, nbytes: int) -> bool:
        if nbytes <= 0:
            raise ValueError("activation offload claims must be positive")
        if self.claimed_bytes + nbytes > self.maximum_bytes:
            return False
        self.claimed_bytes += nbytes
        self.peak_claimed_bytes = max(self.peak_claimed_bytes, self.claimed_bytes)
        self.total_claimed_bytes += nbytes
        return True

    def release(self, nbytes: int) -> None:
        if nbytes <= 0 or nbytes > self.claimed_bytes:
            raise ValueError("activation offload release does not match a live claim")
        self.claimed_bytes -= nbytes


class QueryActivationOffloadScope(AbstractContextManager[object]):
    """One Query-local saved-tensor scope with an explicit post-backward release."""

    def __init__(
        self,
        budget: QueryActivationOffloadBudget,
        *,
        context_maximum_bytes: int,
    ) -> None:
        if context_maximum_bytes <= 0:
            raise ValueError("context_maximum_bytes must be positive")
        self.budget = budget
        self.context_maximum_bytes = context_maximum_bytes
        self.context_claimed_bytes = 0
        self._activations: list[_OffloadedActivation] = []
        self._hooks: AbstractContextManager[object] | None = None
        self._active = False
        self._released = False

    def _pack(self, tensor: torch.Tensor) -> torch.Tensor | _OffloadedActivation:
        if not self._active or self._released:
            raise RuntimeError("Query activation scope is not accepting saved tensors")
        nbytes = tensor.numel() * tensor.element_size()
        should_offload = (
            tensor.device.type == "cuda"
            and tensor.layout is torch.strided
            and tensor.requires_grad
            and not tensor.is_leaf
            and nbytes >= _MINIMUM_OFFLOAD_TENSOR_BYTES
            and self.context_claimed_bytes + nbytes <= self.context_maximum_bytes
        )
        if not should_offload or not self.budget.claim(nbytes):
            return tensor
        packed = torch.empty(
            tensor.size(),
            dtype=tensor.dtype,
            layout=tensor.layout,
            pin_memory=True,
        )
        packed.copy_(tensor)
        self.context_claimed_bytes += nbytes
        activation = _OffloadedActivation(tensor.device, packed, self.budget, nbytes)
        self._activations.append(activation)
        return activation

    @staticmethod
    def _unpack(packed: torch.Tensor | _OffloadedActivation) -> torch.Tensor:
        if isinstance(packed, _OffloadedActivation):
            return packed.restore()
        return packed

    def __enter__(self) -> object:
        if self._active or self._hooks is not None:
            raise RuntimeError("Query activation scope may be entered only once")
        hooks = cast(
            AbstractContextManager[object],
            torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack),
        )
        self._hooks = hooks
        self._active = True
        hooks.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        hooks = self._hooks
        if hooks is None or not self._active:
            raise RuntimeError("Query activation scope exit does not match an active enter")
        try:
            return hooks.__exit__(exc_type, exc_value, traceback)
        finally:
            self._active = False
            if exc_type is not None:
                self.release()

    def release(self) -> None:
        """Release every saved tensor after the Query-local backward has returned."""

        if self._active:
            raise RuntimeError("cannot release Query activations while packing is active")
        if self._released:
            return
        for activation in self._activations:
            activation.release()
        self._activations.clear()
        self._released = True


def _query_offload_budget_bytes() -> int:
    raw = os.environ.get(
        "TTT_QUERY_ACTIVATION_OFFLOAD_MAX_GB",
        str(_DEFAULT_QUERY_OFFLOAD_MAX_GB),
    )
    try:
        max_gb = int(raw)
    except ValueError as error:
        raise ValueError(
            "TTT_QUERY_ACTIVATION_OFFLOAD_MAX_GB must be a positive integer"
        ) from error
    if max_gb <= 0:
        raise ValueError("TTT_QUERY_ACTIVATION_OFFLOAD_MAX_GB must be a positive integer")
    return max_gb * (1 << 30)


def query_activation_context(
    enabled: bool,
    *,
    shared_budget: QueryActivationOffloadBudget | None = None,
    context_maximum_bytes: int | None = None,
) -> AbstractContextManager[object]:
    if not enabled or not torch.cuda.is_available():
        return nullcontext()
    budget = shared_budget or QueryActivationOffloadBudget.from_environment()
    context_limit = budget.maximum_bytes if context_maximum_bytes is None else context_maximum_bytes
    return QueryActivationOffloadScope(
        budget,
        context_maximum_bytes=context_limit,
    )
