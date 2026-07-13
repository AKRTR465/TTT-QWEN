"""Define TTT, State, Answer, Outer, and total loss result contracts.

Inputs: detach-before-write soft branches, valid masks, task labels, and after-update queries.
Outputs: finite scalar losses plus explicit validity flags and metric components.
Forbidden: in-place parameter updates, hard Bank/FSM gradients, or unlabeled O1 consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import torch
from torch import Tensor


def _validate_scalar(value: Tensor, name: str) -> None:
    if value.ndim != 0 or not torch.is_floating_point(value):
        raise ValueError(f"{name} must be a floating scalar tensor")


@dataclass(frozen=True, slots=True)
class TTTLossOutput:
    pred: Tensor
    identity: Tensor
    event: Tensor
    total: Tensor
    pred_valid: bool
    identity_valid: bool
    event_valid: bool

    def __post_init__(self) -> None:
        for value, name in (
            (self.pred, "pred"),
            (self.identity, "identity"),
            (self.event, "event"),
            (self.total, "total"),
        ):
            _validate_scalar(value, name)


@dataclass(frozen=True, slots=True)
class TrainingLossOutput:
    ttt: TTTLossOutput
    state: Tensor
    answer: Tensor
    outer: Tensor
    total: Tensor

    def __post_init__(self) -> None:
        for value, name in (
            (self.state, "state"),
            (self.answer, "answer"),
            (self.outer, "outer"),
            (self.total, "total"),
        ):
            _validate_scalar(value, name)


def compute_losses(*_args: object, **_kwargs: object) -> NoReturn:
    """P14 owns all loss formulas and valid-mask reductions."""

    raise NotImplementedError("Loss implementation is deferred to P14")
