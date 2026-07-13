"""Define Stage A-D episode and training-step result contracts.

Inputs: stage-specific support/query episodes, validated config, model, and outer optimizer.
Outputs: auditable losses, metrics, checkpoint metadata, and update statistics.
Forbidden: duplicating module algorithms, hidden labels in support loss, or clean-test selection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

from ttt_svcbench_qwen.data import assert_runtime_payload_safe
from ttt_svcbench_qwen.losses import TrainingLossOutput


class TrainingStage(StrEnum):
    A = "stage_a"
    B = "stage_b"
    C = "stage_c"
    D = "stage_d"


@dataclass(frozen=True, slots=True)
class TrainingStepOutput:
    stage: TrainingStage
    losses: TrainingLossOutput
    global_step: int
    metrics: tuple[tuple[str, float], ...]
    checkpoint_path: str | None

    def __post_init__(self) -> None:
        if self.global_step < 0:
            raise ValueError("global_step must be non-negative")


def build_trainer(*_args: object, **_kwargs: object) -> NoReturn:
    """P15-P19 own the stage-specific training orchestration."""

    raise NotImplementedError("Trainer implementation is deferred to P15-P19")


def assert_trainer_runtime_payload(payload: Mapping[str, object]) -> None:
    """P2 leakage guard applied before any trainer/model handoff."""

    assert_runtime_payload_safe(payload, layer="Trainer")
