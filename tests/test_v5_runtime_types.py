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
    E1SoftOutput,
    E2SoftOutput,
    O1SoftOutput,
    O2SoftOutput,
    ObservationOutputs,
)
from ttt_svcbench_qwen.query_encoder import (
    Operator,
    QueryEncoderOutput,
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
    output = QueryEncoderOutput(
        q_target=torch.zeros(2, 512),
        q_operator=torch.zeros(2, 512),
        q_time=torch.zeros(2, 512),
        operator_logits=torch.zeros(2, 9),
        operator_confidence=torch.zeros(2),
        padding_mask=torch.zeros(2, 5, dtype=torch.bool),
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
        timestamps=torch.zeros(2, 4),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
        video_ids=("video-a", "video-b"),
    )
    temporal = TemporalEncoderOutput(
        hidden=torch.zeros(2, 4, 768),
        timestamps=torch.zeros(2, 4),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
        cache=cache,
    )
    observations = ObservationOutputs(
        o1=O1SoftOutput(torch.zeros(2, 32, 6)),
        o2=O2SoftOutput(torch.zeros(2, 32, 256), torch.zeros(2, 32, 2)),
        e1=E1SoftOutput(torch.zeros(2, 4, 3)),
        e2=E2SoftOutput(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4)),
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
    matrix = torch.zeros(768, 768)
    fast = FastWeightsState(matrix, matrix, matrix, matrix, 0, 0, 0)
    optimizer = OptimizerRuntimeState("sgd", 1.0e-4, 0.0, 0.0, 1, 1.0, 0, None)
    cache = TemporalCache(
        hidden=torch.zeros(1, 0, 768),
        timestamps=torch.zeros(1, 0),
        valid_mask=torch.zeros(1, 0, dtype=torch.bool),
        video_ids=("video-a",),
    )
    bank = StateBankRuntimeState("video-a", "trajectory-a", (), ())
    identities = IdentityBankRuntimeState((), (), (), 0, 0)
    runtime = PerVideoRuntimeState(
        video_id="video-a",
        trajectory_id="trajectory-a",
        fast_weights=fast,
        optimizer=optimizer,
        slot_state=None,
        temporal_cache=cache,
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
            state_bank=bank,
            identity_bank=identities,
            fsm_state=(),
            reader_audit=(),
            released=False,
        )
