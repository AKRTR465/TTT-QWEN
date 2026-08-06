from __future__ import annotations

import pytest
import torch

from tests.support.runtime_factories import (
    make_e1_state,
    make_e2_state,
    make_state_record,
    make_temporal_cache,
)
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.fast_ttt import FastMemoryState
from ttt_svcbench_qwen.identity_bank import (
    CandidateIdentity,
    ConfirmedIdentity,
    HotCacheEntry,
    build_identity_bank,
)
from ttt_svcbench_qwen.model import RuntimeOwner, TrajectoryRuntimeState
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
)
from ttt_svcbench_qwen.state_reader import ReaderResult, ReaderStatus
from ttt_svcbench_qwen.state_retriever import (
    RetrievalFilterAudit,
    RetrievalReason,
    RetrievalStatus,
    RetrieverOutput,
)


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


def test_typed_state_identity_retrieval_and_reader_contracts() -> None:
    prototype = torch.zeros(256)
    prototype[0] = 1.0
    semantic = torch.zeros(512)
    semantic[0] = 1.0
    candidate = CandidateIdentity("candidate-1", prototype, 1, 8, 0.8)
    confirmed = ConfirmedIdentity("identity-1", prototype, 0.0, 2.0, 3)
    hot = HotCacheEntry("identity-1", prototype, 2)
    identities = build_identity_bank(load_config()).reset(
        "video-a",
        "trajectory-a",
        hot_cache_enabled=False,
    )
    record = make_state_record(
        "record-1",
        HeadType.O1,
        O1Payload(2, 1, (0, 1)),
        semantic_embedding=semantic,
        video_id="video-a",
        trajectory_id="trajectory-a",
        timestamp=2.0,
    )
    bank = StateBankRuntimeState("video-a", "trajectory-a", (record,), ())
    window = TimeWindow(TimeWindowMode.HISTORY, 2.0, 0.0, 2.0, True)
    resolution = TimeResolution(
        window=window,
        status=TimeResolutionStatus.OK,
        reason="synthetic-ok",
        mode_confidence=1.0,
        numeric_span=None,
        parsed_values_seconds=(),
        used_operator_default=True,
    )
    retrieval = RetrieverOutput(
        selected_record_ids=(("record-1",),),
        selected_scores=((0.5,),),
        selected_records=((record,),),
        candidate_record_ids=(("record-1",),),
        candidate_records=((record,),),
        candidate_head_types=((HeadType.O1,),),
        state_embeddings=semantic.reshape(1, 1, 512),
        scores=torch.tensor([[0.5]]),
        present_mask=torch.tensor([[True]]),
        record_valid_mask=torch.tensor([[True]]),
        retrieval_eligible_mask=torch.tensor([[True]]),
        causal_mask=torch.tensor([[True]]),
        predicted_head_mask=torch.tensor([[True]]),
        selected_mask=torch.tensor([[True]]),
        status=(RetrievalStatus.OK,),
        reason=(RetrievalReason.MATCHED,),
        hard_operators=(Operator.O1_SNAP,),
        time_resolutions=(resolution,),
        n_state=torch.tensor([1]),
        n_retrieved=torch.tensor([1]),
        audit=(
            RetrievalFilterAudit(
                n_state=1,
                head_partition_excluded_count=0,
                query_rejected_count=0,
                owner_mismatch_count=0,
                invalid_count=0,
                retrieval_ineligible_count=0,
                future_count=0,
                outside_window_count=0,
                below_similarity_count=0,
                selected_count=1,
            ),
        ),
        video_ids=("video-a",),
        trajectory_ids=("trajectory-a",),
        bank_video_ids=("video-a",),
        bank_trajectory_ids=("trajectory-a",),
        bank_versions=(bank.version,),
    )
    reader = ReaderResult(
        status=ReaderStatus.OK,
        exact_count=2,
        number_token_ids=(17,),
        selected_record_ids=("record-1",),
        operator=Operator.O1_SNAP,
        time_window=window,
        audit_fields=(
            ("source", "retrieved_typed_records"),
            ("operator", "o1-snap"),
            ("retrieval_status", "ok"),
            ("retrieval_reason", "matched"),
            ("n_state", 1),
            ("n_retrieved", 1),
            ("input_record_count", 1),
            ("bank_version", bank.version),
            ("time_resolution_status", "ok"),
            ("window_start", 0.0),
            ("window_end", 2.0),
            ("reader_reason", "exact_typed_payload_arithmetic"),
            ("arithmetic", "o1_current_visible_count"),
            ("contributing_count", 1),
            ("operand_current_visible_count", 2),
            ("operand_baseline_count", 1),
            ("operand_baseline_initialized", True),
            ("operand_baseline_position_id", None),
            ("computed_exact_count", 2),
            ("number_text", "2"),
        ),
    )

    assert identities.unique_count == 0
    assert candidate.observation_count == 1
    assert confirmed.observation_count == 3
    assert hot.last_accessed_position_id == 2
    assert bank.records[0].payload.current_visible_count == 2
    assert retrieval.n_retrieved.item() == 1
    assert reader.exact_count == 2


def test_per_video_runtime_covers_all_owned_state_and_rejects_cross_video_bank() -> None:
    fast = FastMemoryState(
        m=torch.zeros((768, 768), dtype=torch.float32, requires_grad=True),
        write_version=0,
        write_count=0,
        skip_count=0,
    )
    cache = make_temporal_cache(
        hidden=torch.zeros(1, 0, 768),
        video_ids=("video-a",),
        trajectory_ids=("trajectory-a",),
    )
    bank = StateBankRuntimeState("video-a", "trajectory-a", (), ())
    identity_operator = build_identity_bank(load_config())
    identities = identity_operator.reset(
        "video-a",
        "trajectory-a",
        hot_cache_enabled=False,
    )
    e1_state = make_e1_state()
    e2_state = make_e2_state()
    owner = RuntimeOwner(("video-a",), ("trajectory-a",))
    values = {
        "owner": owner,
        "next_chunk_index": 0,
        "fast_weights": fast,
        "slot_state": None,
        "temporal_cache": cache,
        "e1_state": e1_state,
        "e2_state": e2_state,
        "state_bank": bank,
        "identity_bank": identities,
        "reader_audit": (),
        "released": False,
    }
    runtime = TrajectoryRuntimeState(**values)

    assert runtime.fast_weights.write_version == 0
    with pytest.raises(ValueError, match="State Bank ownership"):
        TrajectoryRuntimeState(
            **{
                **values,
                "owner": RuntimeOwner(("video-b",), ("trajectory-a",)),
                "e1_state": None,
                "e2_state": None,
            }
        )

    mismatched_identities = identity_operator.reset(
        "video-b",
        "trajectory-a",
        hot_cache_enabled=False,
    )
    with pytest.raises(ValueError, match="Identity Bank ownership"):
        TrajectoryRuntimeState(**{**values, "identity_bank": mismatched_identities})

    with pytest.raises(ValueError, match="Identity Bank release"):
        TrajectoryRuntimeState(**{**values, "identity_bank": identity_operator.release(identities)})

    mismatched_e1 = make_e1_state(query_signature=torch.ones(512))
    with pytest.raises(ValueError, match="E1 state query signature"):
        TrajectoryRuntimeState(**{**values, "e1_state": mismatched_e1})
