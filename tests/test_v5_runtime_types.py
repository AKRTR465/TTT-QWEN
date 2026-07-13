from __future__ import annotations

import pytest
import torch

from ttt_svcbench_qwen.fast_ttt import FastWeightsState, OptimizerRuntimeState
from ttt_svcbench_qwen.identity_bank import (
    CandidateIdentity,
    ConfirmedIdentity,
    HotCacheEntry,
    IdentityBankRuntimeState,
)
from ttt_svcbench_qwen.inference import PerVideoRuntimeState
from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E1SoftOutput,
    E2RuntimeState,
    E2SoftOutput,
    O1SoftOutput,
    O2SoftOutput,
    ObservationOutputs,
    StreamReplayAudit,
)
from ttt_svcbench_qwen.query_encoder import (
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
from ttt_svcbench_qwen.qwen_adapter import (
    MergedVideoMetadata,
    QwenVisualOutput,
    VideoBatch,
)
from ttt_svcbench_qwen.state_bank import (
    HeadType,
    O1Payload,
    StateBankRuntimeState,
    StateRecord,
)
from ttt_svcbench_qwen.state_encoder import (
    SpatialEncoderOutput,
    TemporalCache,
    TemporalEncoderOutput,
)
from ttt_svcbench_qwen.state_reader import ReaderResult, ReaderStatus
from ttt_svcbench_qwen.state_retriever import RetrievalStatus, RetrieverOutput


def make_video_batch() -> VideoBatch:
    return VideoBatch(
        pixel_values_videos=torch.zeros(16, 1536),
        video_grid_thw=torch.tensor([[2, 2, 2], [2, 2, 2]], dtype=torch.int64),
        timestamps=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        query_time=torch.tensor([1.0, 1.0]),
        valid_mask=torch.ones(2, 2, dtype=torch.bool),
        video_ids=("video-a", "video-b"),
        trajectory_ids=("trajectory-a", "trajectory-b"),
    )


def test_video_batch_and_qwen_visual_contracts_validate_shape_dtype_and_ids() -> None:
    batch = make_video_batch()
    main = torch.zeros(2, 2, 4096)
    packed_deepstack = torch.zeros(4, 4096)
    output = QwenVisualOutput(
        main_visual_embeddings=main,
        deepstack_features=(
            packed_deepstack.clone(),
            packed_deepstack.clone(),
            packed_deepstack.clone(),
        ),
        visual_valid_mask=torch.ones(2, 2, dtype=torch.bool),
        metadata=MergedVideoMetadata(
            video_grid_thw=batch.video_grid_thw,
            merged_grid_thw=torch.tensor([[2, 1, 1], [2, 1, 1]], dtype=torch.int64),
            spatial_merge_size=2,
            token_counts=(2, 2),
            token_offsets=(0, 2, 4),
        ),
    )

    assert batch.patch_offsets == (0, 8, 16)
    assert output.main_visual_embeddings.shape == (2, 2, 4096)
    with pytest.raises(ValueError, match="valid_mask"):
        VideoBatch(
            pixel_values_videos=batch.pixel_values_videos,
            video_grid_thw=batch.video_grid_thw,
            timestamps=batch.timestamps,
            query_time=batch.query_time,
            valid_mask=torch.ones(2, 3, dtype=torch.bool),
            video_ids=batch.video_ids,
            trajectory_ids=batch.trajectory_ids,
        )


def test_query_output_and_time_window_contracts_reject_future_time() -> None:
    padding_mask = torch.zeros(2, 5, dtype=torch.bool)
    embeddings = QueryEmbeddingOutput(
        token_states=torch.zeros(2, 5, 768),
        pooling_weights=torch.full((2, 5), 0.2),
        q_target=torch.zeros(2, 512),
        q_operator=torch.zeros(2, 512),
        q_time=torch.zeros(2, 512),
        padding_mask=padding_mask,
    )
    route = OperatorRouterOutput(
        logits=torch.zeros(2, 9),
        confidence=torch.zeros(2),
        raw_indices=torch.zeros(2, dtype=torch.int64),
        hard_operators=(Operator.O1_SNAP, Operator.O1_SNAP),
        head_types=(HeadType.O1, HeadType.O1),
        confidence_gate_applied=False,
    )
    time_logits = TimeResolverLogits(
        mode_logits=torch.zeros(2, 4),
        mode_confidence=torch.zeros(2),
        mode_indices=torch.zeros(2, dtype=torch.int64),
        span_start_logits=torch.zeros(2, 5),
        span_end_logits=torch.zeros(2, 5),
        padding_mask=padding_mask,
    )
    now_windows = tuple(
        TimeResolution(
            window=TimeWindow(TimeWindowMode.NOW, 2.0, None, 2.0, True),
            status=TimeResolutionStatus.OK,
            reason="runtime-contract",
            mode_confidence=0.0,
            numeric_span=None,
            parsed_values_seconds=(),
            used_operator_default=True,
        )
        for _ in range(2)
    )
    output = QueryEncoderOutput(
        embeddings=embeddings,
        route=route,
        time=TimeResolverOutput(logits=time_logits, resolutions=now_windows),
        hard_operators=(Operator.O1_SNAP, Operator.O1_SNAP),
        head_types=(HeadType.O1, HeadType.O1),
    )
    window = TimeWindow(
        mode=TimeWindowMode.HISTORY,
        query_time=8.0,
        start_time=0.0,
        end_time=8.0,
        valid=True,
    )

    assert output.q_target.shape == (2, 512)
    assert window.end_time == window.query_time
    with pytest.raises(ValueError, match="beyond query_time"):
        TimeWindow(
            mode=TimeWindowMode.HISTORY,
            query_time=8.0,
            start_time=0.0,
            end_time=9.0,
            valid=True,
        )


def test_encoder_cache_and_observation_output_contracts() -> None:
    spatial = SpatialEncoderOutput(
        slots=torch.zeros(2, 32, 768),
        slot_valid_mask=torch.ones(2, 32, dtype=torch.bool),
        active_slot_overflow_count=torch.zeros(2, dtype=torch.int64),
    )
    cache = TemporalCache(
        hidden=torch.zeros(2, 4, 768),
        layer_keys=tuple(torch.zeros(2, 12, 4, 64) for _ in range(6)),
        layer_values=tuple(torch.zeros(2, 12, 4, 64) for _ in range(6)),
        replay_layer_keys=tuple(torch.zeros(2, 12, 0, 64) for _ in range(6)),
        replay_layer_values=tuple(torch.zeros(2, 12, 0, 64) for _ in range(6)),
        timestamps=torch.arange(4, dtype=torch.float64).repeat(2, 1),
        replay_timestamps=torch.empty(2, 0, dtype=torch.float64),
        position_ids=torch.arange(4, dtype=torch.int64).repeat(2, 1),
        replay_position_ids=torch.empty(2, 0, dtype=torch.int64),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
        replay_valid_mask=torch.empty(2, 0, dtype=torch.bool),
        video_ids=("video-a", "video-b"),
        trajectory_ids=("trajectory-a", "trajectory-b"),
        query_signatures=torch.zeros(2, 512),
        total_seen=torch.full((2,), 4, dtype=torch.int64),
    )
    temporal = TemporalEncoderOutput(
        hidden=torch.zeros(2, 4, 768),
        timestamps=torch.arange(4, dtype=torch.float32).repeat(2, 1),
        position_ids=torch.arange(4, dtype=torch.int64).repeat(2, 1),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
        cache=cache,
    )
    slot_mask = torch.zeros(2, 32, dtype=torch.bool)
    temporal_mask = torch.zeros(2, 4, dtype=torch.bool)
    slot_timestamps = torch.full((2, 32), -1.0)
    temporal_timestamps = torch.full((2, 4), -1.0)
    slot_positions = torch.full((2, 32), -1, dtype=torch.int64)
    temporal_positions = torch.full((2, 4), -1, dtype=torch.int64)
    e1_states = tuple(
        E1RuntimeState(
            video_id=f"video-{row}",
            trajectory_id=f"trajectory-{row}",
            query_signature=torch.zeros(512),
            projected_history=torch.zeros(0, 512),
            timestamps=torch.zeros(0, dtype=torch.float64),
            position_ids=torch.zeros(0, dtype=torch.int64),
            total_seen=0,
        )
        for row in range(2)
    )
    e2_states = tuple(
        E2RuntimeState(
            video_id=f"video-{row}",
            trajectory_id=f"trajectory-{row}",
            query_signature=torch.zeros(512),
            hidden=torch.zeros(2, 768),
            checkpoint_hidden=torch.zeros(0, 2, 768),
            timestamps=torch.zeros(0, dtype=torch.float64),
            position_ids=torch.zeros(0, dtype=torch.int64),
            total_seen=0,
        )
        for row in range(2)
    )
    observations = ObservationOutputs(
        o1=O1SoftOutput(
            logits=torch.zeros(2, 32, 6),
            probabilities=torch.zeros(2, 32, 6),
            soft_count=torch.zeros(2),
            valid_mask=slot_mask,
            timestamps=slot_timestamps,
            position_ids=slot_positions,
        ),
        o2=O2SoftOutput(
            identity=torch.zeros(2, 32, 256),
            score_logits=torch.zeros(2, 32, 2),
            score_probabilities=torch.zeros(2, 32, 2),
            valid_mask=slot_mask,
            timestamps=slot_timestamps.clone(),
            position_ids=slot_positions.clone(),
        ),
        e1=E1SoftOutput(
            logits=torch.zeros(2, 4, 3),
            probabilities=torch.zeros(2, 4, 3),
            valid_mask=temporal_mask,
            timestamps=temporal_timestamps,
            position_ids=temporal_positions,
            next_states=e1_states,
            audit=StreamReplayAudit("e1", (0, 0), (0, 0), (0, 0)),
        ),
        e2=E2SoftOutput(
            event_logits=torch.zeros(2, 4, 4),
            phase_logits=torch.zeros(2, 4, 4),
            event_probabilities=torch.zeros(2, 4, 4),
            phase_probabilities=torch.zeros(2, 4, 4),
            valid_mask=temporal_mask.clone(),
            timestamps=temporal_timestamps.clone(),
            position_ids=temporal_positions.clone(),
            next_states=e2_states,
            audit=StreamReplayAudit("e2", (0, 0), (0, 0), (0, 0)),
        ),
    )

    assert spatial.slots.shape == (2, 32, 768)
    assert temporal.cache.hidden.shape[1] == 4
    assert observations.o2.identity.shape[-1] == 256


def test_typed_state_identity_retrieval_and_reader_contracts() -> None:
    prototype = torch.zeros(256)
    candidate = CandidateIdentity("candidate-1", prototype, 1, 8, 0.8)
    confirmed = ConfirmedIdentity("identity-1", prototype, 0.0, 2.0, 3)
    hot = HotCacheEntry("identity-1", prototype, 2.0)
    identities = IdentityBankRuntimeState((candidate,), (confirmed,), (hot,), 1, 0)
    record = StateRecord(
        record_id="record-1",
        video_id="video-a",
        trajectory_id="trajectory-a",
        head_type=HeadType.O1,
        semantic_embedding=torch.zeros(512),
        timestamp=2.0,
        time_range=None,
        valid=True,
        confidence=0.9,
        payload=O1Payload(2, 1, (0, 1)),
    )
    bank = StateBankRuntimeState("video-a", "trajectory-a", (record,), ())
    retrieval = RetrieverOutput(
        selected_record_ids=(("record-1",),),
        scores=torch.tensor([[0.9]]),
        selected_mask=torch.tensor([[True]]),
        status=(RetrievalStatus.OK,),
        n_state=torch.tensor([1]),
        n_retrieved=torch.tensor([1]),
    )
    window = TimeWindow(TimeWindowMode.HISTORY, 2.0, 0.0, 2.0, True)
    reader = ReaderResult(
        status=ReaderStatus.OK,
        exact_count=2,
        number_token_ids=(17,),
        selected_record_ids=("record-1",),
        operator=Operator.O1_SNAP,
        time_window=window,
        audit_fields=(("source", "hard-records"),),
    )

    assert identities.unique_count == 1
    assert bank.records[0].payload.current_visible_count == 2
    assert retrieval.n_retrieved.item() == 1
    assert reader.exact_count == 2


def test_per_video_runtime_covers_all_owned_state_and_rejects_cross_video_bank() -> None:
    w0_1 = torch.zeros(768, 768)
    w0_2 = torch.ones(768, 768)
    w_t_1 = w0_1.clone().requires_grad_(True)
    w_t_2 = w0_2.clone().requires_grad_(True)
    fast = FastWeightsState(w0_1, w0_2, w_t_1, w_t_2, 0, 0, 0)
    optimizer = OptimizerRuntimeState("sgd", 1.0e-4, 0.0, 0.0, 1, 1.0, 0, None)
    cache = TemporalCache(
        hidden=torch.zeros(1, 0, 768),
        layer_keys=tuple(torch.zeros(1, 12, 0, 64) for _ in range(6)),
        layer_values=tuple(torch.zeros(1, 12, 0, 64) for _ in range(6)),
        replay_layer_keys=tuple(torch.zeros(1, 12, 0, 64) for _ in range(6)),
        replay_layer_values=tuple(torch.zeros(1, 12, 0, 64) for _ in range(6)),
        timestamps=torch.zeros(1, 0, dtype=torch.float64),
        replay_timestamps=torch.zeros(1, 0, dtype=torch.float64),
        position_ids=torch.zeros(1, 0, dtype=torch.int64),
        replay_position_ids=torch.zeros(1, 0, dtype=torch.int64),
        valid_mask=torch.zeros(1, 0, dtype=torch.bool),
        replay_valid_mask=torch.zeros(1, 0, dtype=torch.bool),
        video_ids=("video-a",),
        trajectory_ids=("trajectory-a",),
        query_signatures=torch.zeros(1, 512),
        total_seen=torch.zeros(1, dtype=torch.int64),
    )
    bank = StateBankRuntimeState("video-a", "trajectory-a", (), ())
    identities = IdentityBankRuntimeState((), (), (), 0, 0)
    e1_state = E1RuntimeState(
        video_id="video-a",
        trajectory_id="trajectory-a",
        query_signature=torch.zeros(512),
        projected_history=torch.zeros(0, 512),
        timestamps=torch.zeros(0, dtype=torch.float64),
        position_ids=torch.zeros(0, dtype=torch.int64),
        total_seen=0,
    )
    e2_state = E2RuntimeState(
        video_id="video-a",
        trajectory_id="trajectory-a",
        query_signature=torch.zeros(512),
        hidden=torch.zeros(2, 768),
        checkpoint_hidden=torch.zeros(0, 2, 768),
        timestamps=torch.zeros(0, dtype=torch.float64),
        position_ids=torch.zeros(0, dtype=torch.int64),
        total_seen=0,
    )
    runtime = PerVideoRuntimeState(
        video_id="video-a",
        trajectory_id="trajectory-a",
        fast_weights=fast,
        optimizer=optimizer,
        slot_state=None,
        temporal_cache=cache,
        e1_state=e1_state,
        e2_state=e2_state,
        state_bank=bank,
        identity_bank=identities,
        fsm_state=(),
        reader_audit=(),
        released=False,
    )

    assert runtime.fast_weights.fast_version == 0
    with pytest.raises(ValueError, match="video_id"):
        PerVideoRuntimeState(
            video_id="video-b",
            trajectory_id="trajectory-a",
            fast_weights=fast,
            optimizer=optimizer,
            slot_state=None,
            temporal_cache=cache,
            e1_state=None,
            e2_state=None,
            state_bank=bank,
            identity_bank=identities,
            fsm_state=(),
            reader_audit=(),
            released=False,
        )

    mismatched_e1 = E1RuntimeState(
        video_id="video-a",
        trajectory_id="trajectory-a",
        query_signature=torch.ones(512),
        projected_history=torch.zeros(0, 512),
        timestamps=torch.zeros(0, dtype=torch.float64),
        position_ids=torch.zeros(0, dtype=torch.int64),
        total_seen=0,
    )
    with pytest.raises(ValueError, match="E1 state query signature"):
        PerVideoRuntimeState(
            video_id="video-a",
            trajectory_id="trajectory-a",
            fast_weights=fast,
            optimizer=optimizer,
            slot_state=None,
            temporal_cache=cache,
            e1_state=mismatched_e1,
            e2_state=e2_state,
            state_bank=bank,
            identity_bank=identities,
            fsm_state=(),
            reader_audit=(),
            released=False,
        )
