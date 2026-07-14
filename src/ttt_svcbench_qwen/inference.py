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
from ttt_svcbench_qwen.observation_heads import E1RuntimeState, E2RuntimeState
from ttt_svcbench_qwen.state_bank import StateBankRuntimeState
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
    e1_state: E1RuntimeState | None
    e2_state: E2RuntimeState | None
    state_bank: StateBankRuntimeState
    identity_bank: IdentityBankRuntimeState
    reader_audit: tuple[ReaderResult, ...]
    released: bool

    def __post_init__(self) -> None:
        if not self.video_id or not self.trajectory_id:
            raise ValueError("per-video runtime identifiers must be non-empty")
        if type(self.released) is not bool:
            raise TypeError("per-video runtime released flag must be bool")
        if self.state_bank.video_id != self.video_id:
            raise ValueError("State Bank video_id does not match runtime ownership")
        if self.state_bank.trajectory_id != self.trajectory_id:
            raise ValueError("State Bank trajectory_id does not match runtime ownership")
        if self.state_bank.released != self.released:
            raise ValueError("State Bank and per-video release state must agree")
        if self.identity_bank.video_id != self.video_id:
            raise ValueError("Identity Bank video_id does not match runtime ownership")
        if self.identity_bank.trajectory_id != self.trajectory_id:
            raise ValueError("Identity Bank trajectory_id does not match runtime ownership")
        if self.identity_bank.released != self.released:
            raise ValueError("Identity Bank and per-video release state must agree")
        if self.slot_state is not None and self.slot_state.video_id != self.video_id:
            raise ValueError("spatial slot state video_id does not match runtime ownership")
        if self.temporal_cache.hidden.shape[0] != 1:
            raise ValueError("per-video temporal cache must have batch size 1")
        if self.temporal_cache.video_ids != (self.video_id,):
            raise ValueError("temporal cache video_ids do not match runtime ownership")
        if self.temporal_cache.trajectory_ids != (self.trajectory_id,):
            raise ValueError("temporal cache trajectory_ids do not match runtime ownership")
        for name, state in (("E1", self.e1_state), ("E2", self.e2_state)):
            if state is None:
                continue
            if state.video_id != self.video_id or state.trajectory_id != self.trajectory_id:
                raise ValueError(f"{name} state does not match runtime ownership")
            if (
                state.query_signature.dtype != self.temporal_cache.hidden.dtype
                or state.query_signature.device != self.temporal_cache.hidden.device
                or not state.query_signature.equal(self.temporal_cache.query_signatures[0])
            ):
                raise ValueError(f"{name} state query signature does not match temporal cache")
            if state.total_seen != int(self.temporal_cache.total_seen[0].item()):
                raise ValueError(f"{name} state position does not match temporal cache")


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
