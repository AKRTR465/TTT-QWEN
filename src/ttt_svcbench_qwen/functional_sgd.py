"""Define the functional single-step SGD update result.

Inputs: finite L_TTT gradients for exactly two fast matrices and the frozen SGD config.
Outputs: next-chunk fast weights or an explicit audited skip result.
Forbidden: Bank/FSM logic, momentum, multi-step updates, or mutation of slow parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

from ttt_svcbench_qwen.fast_ttt import FastWeightsState


class UpdateSkipReason(StrEnum):
    NO_VALID_TERM = "no_valid_term"
    INSUFFICIENT_TIME = "insufficient_time"
    NONFINITE_LOSS = "nonfinite_loss"
    NONFINITE_GRADIENT = "nonfinite_gradient"
    INVALID_AFTER_CLIP = "invalid_after_clip"


@dataclass(frozen=True, slots=True)
class FunctionalSGDResult:
    fast_state: FastWeightsState
    did_update: bool
    gradient_norm: float | None
    clipped_gradient_norm: float | None
    skip_reason: UpdateSkipReason | None

    def __post_init__(self) -> None:
        if self.did_update == (self.skip_reason is not None):
            raise ValueError("updated results need no skip reason; skipped results need one")
        for norm in (self.gradient_norm, self.clipped_gradient_norm):
            if norm is not None and norm < 0.0:
                raise ValueError("gradient norms must be non-negative")


def functional_sgd_step(*_args: object, **_kwargs: object) -> NoReturn:
    """P14 owns finite checks, clipping, and the differentiable one-step update."""

    raise NotImplementedError("Functional SGD implementation is deferred to P14")
