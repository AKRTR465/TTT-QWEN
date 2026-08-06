"""Shared constructors for the typed runtime objects the test suite builds by hand.

Every factory reproduces the dtype, device, storage-isolation and metadata semantics
of the per-module builders it replaces. Axes on which those builders diverged (cache
``total_seen``, supplied vs derived metadata, stream audit state lengths, record
ownership) are explicit keyword overrides, so no call site silently inherits a
different invariant.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E2RuntimeState,
    StreamReplayAudit,
)
from ttt_svcbench_qwen.query_encoder import (
    OPERATOR_TO_HEAD_TYPE,
    Operator,
    OperatorRouterOutput,
    QueryEmbeddingOutput,
    QueryEncoderOutput,
    TimeResolution,
    TimeResolutionStatus,
    TimeResolverLogits,
    TimeResolverOutput,
    TimeWindow,
    TimeWindowMode,
)
from ttt_svcbench_qwen.state_bank import HeadType, StatePayload, StateRecord
from ttt_svcbench_qwen.state_encoder import (
    SpatialEncoderOutput,
    SpatialSlotRuntimeState,
    TemporalCache,
)

HIDDEN_DIM = 768
QUERY_DIM = 512
E1_HISTORY_CAPACITY = 66
E2_CHECKPOINT_CAPACITY = 5
E2_LAYERS = 2


def make_temporal_cache(
    *,
    hidden: Tensor,
    video_ids: tuple[str, ...],
    trajectory_ids: tuple[str, ...],
    query_signatures: Tensor | None = None,
    timestamps: Tensor | None = None,
    position_ids: Tensor | None = None,
    valid_mask: Tensor | None = None,
    total_seen: Tensor | None = None,
) -> TemporalCache:
    """Build a zero-filled cache with an empty replay context.

    ``hidden`` fixes batch size, cache width, dtype and device for every K/V layer.
    Metadata defaults to a dense ``arange`` prefix whose ``total_seen`` is the cache
    width; override any of it when a call site pins different semantics.
    """
    batch_size, width = hidden.shape[:2]
    device = hidden.device
    dtype = hidden.dtype

    def layers(length: int) -> tuple[Tensor, ...]:
        return tuple(
            torch.zeros(batch_size, 12, length, 64, dtype=dtype, device=device) for _ in range(6)
        )

    return TemporalCache(
        hidden=hidden.detach().clone(),
        layer_keys=layers(width),
        layer_values=layers(width),
        replay_layer_keys=layers(0),
        replay_layer_values=layers(0),
        timestamps=(
            torch.arange(width, dtype=torch.float64, device=device).expand(batch_size, -1).clone()
            if timestamps is None
            else timestamps.clone()
        ),
        replay_timestamps=torch.zeros(batch_size, 0, dtype=torch.float64, device=device),
        position_ids=(
            torch.arange(width, dtype=torch.int64, device=device).expand(batch_size, -1).clone()
            if position_ids is None
            else position_ids.clone()
        ),
        replay_position_ids=torch.zeros(batch_size, 0, dtype=torch.int64, device=device),
        valid_mask=(
            torch.ones(batch_size, width, dtype=torch.bool, device=device)
            if valid_mask is None
            else valid_mask.clone()
        ),
        replay_valid_mask=torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
        video_ids=video_ids,
        trajectory_ids=trajectory_ids,
        query_signatures=(
            torch.zeros(batch_size, QUERY_DIM, dtype=dtype, device=device)
            if query_signatures is None
            else query_signatures.detach().clone()
        ),
        total_seen=(
            torch.full((batch_size,), width, dtype=torch.int64, device=device)
            if total_seen is None
            else total_seen
        ),
    )


def make_spatial_output(
    slots: Tensor,
    *,
    video_ids: tuple[str, ...],
    processed_tubelets: int = 3,
) -> SpatialEncoderOutput:
    """Build an all-valid, zero-overflow spatial output with one state per row."""
    batch_size, width = slots.shape[:2]
    mask = torch.ones((batch_size, width), dtype=torch.bool)
    next_states = tuple(
        SpatialSlotRuntimeState(
            video_id=video_ids[row],
            slots=slots[row].detach().clone(),
            slot_valid_mask=mask[row].clone(),
            slot_confidence=torch.ones(width),
            active_slot_overflow_count=0,
            overflow_event_count=0,
            processed_tubelets=processed_tubelets,
        )
        for row in range(batch_size)
    )
    return SpatialEncoderOutput(
        slots=slots,
        slot_valid_mask=mask,
        active_slot_overflow_count=torch.zeros(batch_size, dtype=torch.int64),
        slot_confidence=torch.ones((batch_size, width)),
        next_states=next_states,
    )


def make_e1_state(
    *,
    video_id: str = "video-a",
    trajectory_id: str = "trajectory-a",
    query_signature: Tensor | None = None,
    total_seen: int = 0,
    timestamps: Tensor | None = None,
    position_ids: Tensor | None = None,
) -> E1RuntimeState:
    """Build a zero-history E1 state whose length is ``min(total_seen, 66)``."""
    length = min(total_seen, E1_HISTORY_CAPACITY)
    positions = (
        torch.arange(total_seen - length, total_seen, dtype=torch.int64)
        if position_ids is None
        else position_ids
    )
    return E1RuntimeState(
        video_id=video_id,
        trajectory_id=trajectory_id,
        query_signature=torch.zeros(QUERY_DIM) if query_signature is None else query_signature,
        projected_history=torch.zeros(length, 512),
        timestamps=positions.to(dtype=torch.float64) if timestamps is None else timestamps,
        position_ids=positions,
        total_seen=total_seen,
    )


def make_e2_state(
    *,
    video_id: str = "video-a",
    trajectory_id: str = "trajectory-a",
    query_signature: Tensor | None = None,
    total_seen: int = 0,
    timestamps: Tensor | None = None,
    position_ids: Tensor | None = None,
) -> E2RuntimeState:
    """Build a zero-hidden E2 state with ``min(total_seen, 5)`` checkpoints."""
    length = min(total_seen, E2_CHECKPOINT_CAPACITY)
    positions = (
        torch.arange(total_seen - length, total_seen, dtype=torch.int64)
        if position_ids is None
        else position_ids
    )
    checkpoints = torch.zeros(length, E2_LAYERS, HIDDEN_DIM)
    return E2RuntimeState(
        video_id=video_id,
        trajectory_id=trajectory_id,
        query_signature=torch.zeros(QUERY_DIM) if query_signature is None else query_signature,
        hidden=(checkpoints[-1].clone() if length else torch.zeros(E2_LAYERS, HIDDEN_DIM)),
        checkpoint_hidden=checkpoints,
        timestamps=positions.to(dtype=torch.float64) if timestamps is None else timestamps,
        position_ids=positions,
        total_seen=total_seen,
    )


def make_stream_audit(
    head: str,
    batch_size: int,
    width: int,
    *,
    state_length: int | None = None,
) -> StreamReplayAudit:
    """Build a no-overlap stream audit; ``state_length`` defaults to ``width``."""
    resolved = width if state_length is None else state_length
    return StreamReplayAudit(
        head,
        (width,) * batch_size,
        (0,) * batch_size,
        (resolved,) * batch_size,
    )


def make_query_output(
    operators: tuple[Operator, ...],
    *,
    q_target: Tensor,
) -> QueryEncoderOutput:
    """Build a confidently routed query whose window is a valid 0..2s history span."""
    batch_size = len(operators)
    raw_indices = torch.tensor(
        [tuple(Operator).index(operator) for operator in operators], dtype=torch.int64
    )
    logits = torch.full((batch_size, len(tuple(Operator))), -5.0)
    logits[torch.arange(batch_size), raw_indices] = 5.0
    route = OperatorRouterOutput(
        logits=logits,
        confidence=torch.ones(batch_size),
        raw_indices=raw_indices,
        hard_operators=operators,
        head_types=tuple(OPERATOR_TO_HEAD_TYPE[operator] for operator in operators),
        confidence_gate_applied=False,
    )
    time_logits = TimeResolverLogits(
        mode_logits=torch.zeros((batch_size, 4)),
        mode_confidence=torch.ones(batch_size),
        mode_indices=torch.ones(batch_size, dtype=torch.int64),
        span_start_logits=torch.zeros((batch_size, 1)),
        span_end_logits=torch.zeros((batch_size, 1)),
        padding_mask=torch.zeros((batch_size, 1), dtype=torch.bool),
    )
    resolutions = tuple(
        TimeResolution(
            window=TimeWindow(TimeWindowMode.HISTORY, 2.0, 0.0, 2.0, True),
            status=TimeResolutionStatus.OK,
            reason="synthetic_explicit",
            mode_confidence=1.0,
            numeric_span=None,
            parsed_values_seconds=(),
            used_operator_default=True,
        )
        for _ in range(batch_size)
    )
    return QueryEncoderOutput(
        embeddings=QueryEmbeddingOutput(
            token_states=torch.zeros((batch_size, 1, HIDDEN_DIM)),
            pooling_weights=torch.ones((batch_size, 1)),
            q_target=q_target,
            q_operator=q_target.clone(),
            q_time=q_target.clone(),
            padding_mask=torch.zeros((batch_size, 1), dtype=torch.bool),
        ),
        route=route,
        time=TimeResolverOutput(time_logits, resolutions),
        hard_operators=operators,
        head_types=route.head_types,
    )


def make_state_record(
    record_id: str,
    head_type: HeadType,
    payload: StatePayload,
    *,
    semantic_embedding: Tensor,
    video_id: str = "video-0",
    trajectory_id: str = "trajectory-0",
    timestamp: float | None = 0.0,
    time_range: tuple[float, float] | None = None,
    valid: bool = True,
    confidence: float = 0.9,
) -> StateRecord:
    """Build a point-in-time, valid record; every contract field stays overridable."""
    return StateRecord(
        record_id=record_id,
        video_id=video_id,
        trajectory_id=trajectory_id,
        head_type=head_type,
        semantic_embedding=semantic_embedding,
        timestamp=timestamp,
        time_range=time_range,
        valid=valid,
        confidence=confidence,
        payload=payload,
    )
