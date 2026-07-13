"""Define per-video runtime ownership and inference-result contracts.

Inputs: causal chunks, questions, legal query times, validated config, and one video runtime.
Outputs: one Reader-backed answer plus reset/update/state/retrieval audit metadata.
Forbidden: training labels, cross-video state, future frames, or repeated updates during decode.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn

from ttt_svcbench_qwen.data import assert_runtime_payload_safe
from ttt_svcbench_qwen.fast_ttt import FastWeightsState, OptimizerRuntimeState
from ttt_svcbench_qwen.identity_bank import IdentityBankRuntimeState
from ttt_svcbench_qwen.state_bank import HeadType, StateBankRuntimeState
from ttt_svcbench_qwen.state_encoder import SpatialSlotRuntimeState, TemporalCache
from ttt_svcbench_qwen.state_reader import ReaderResult


@dataclass(frozen=True, slots=True)
class PerVideoRuntimeState:
    video_id: str
    trajectory_id: str
    fast_weights: FastWeightsState
    optimizer: OptimizerRuntimeState
    slot_state: SpatialSlotRuntimeState | None
    temporal_cache: TemporalCache
    state_bank: StateBankRuntimeState
    identity_bank: IdentityBankRuntimeState
    fsm_state: tuple[tuple[HeadType, str], ...]
    reader_audit: tuple[ReaderResult, ...]
    released: bool

    def __post_init__(self) -> None:
        if not self.video_id or not self.trajectory_id:
            raise ValueError("per-video runtime identifiers must be non-empty")
        if self.state_bank.video_id != self.video_id:
            raise ValueError("State Bank video_id does not match runtime ownership")
        if self.state_bank.trajectory_id != self.trajectory_id:
            raise ValueError("State Bank trajectory_id does not match runtime ownership")
        if self.slot_state is not None and self.slot_state.video_id != self.video_id:
            raise ValueError("spatial slot state video_id does not match runtime ownership")
        if self.temporal_cache.hidden.shape[0] != 1:
            raise ValueError("per-video temporal cache must have batch size 1")
        if self.temporal_cache.video_ids != (self.video_id,):
            raise ValueError("temporal cache video_ids do not match runtime ownership")
        if self.temporal_cache.trajectory_ids != (self.trajectory_id,):
            raise ValueError("temporal cache trajectory_ids do not match runtime ownership")


@dataclass(frozen=True, slots=True)
class InferenceResult:
    answer_text: str
    reader_result: ReaderResult
    runtime_state: PerVideoRuntimeState
    audit_fields: tuple[tuple[str, str | int | float | bool | None], ...]

    def __post_init__(self) -> None:
        if not self.answer_text:
            raise ValueError("inference answer_text must be non-empty")


def run_inference(*_args: object, **_kwargs: object) -> NoReturn:
    """P18 owns reset, causal chunk order, one-time prefill, and release."""

    raise NotImplementedError("Inference protocol implementation is deferred to P18")


def assert_inference_runtime_payload(payload: Mapping[str, object]) -> None:
    """P2 leakage guard applied before any inference/model handoff."""

    assert_runtime_payload_safe(payload, layer="Inference")
