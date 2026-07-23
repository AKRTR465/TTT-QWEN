from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from ttt_svcbench_qwen.data import RUNTIME_ALLOWLIST, RuntimeQueryInput
from ttt_svcbench_qwen.model import BatchRuntimeState, TrajectoryRuntimeState

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_query_input_is_the_single_internal_query_contract() -> None:
    query = RuntimeQueryInput(
        video_id="video-a",
        trajectory_id="trajectory-a",
        query_id="query-a",
        query_index=0,
        video=Path("video.mp4"),
        question="How many?",
        query_time=2.0,
        explicit_time_values=(),
    )

    assert set(query.as_payload()) == RUNTIME_ALLOWLIST
    assert tuple(field.name for field in fields(RuntimeQueryInput)) == (
        "video_id",
        "trajectory_id",
        "query_id",
        "query_index",
        "video",
        "question",
        "query_time",
        "explicit_time_values",
        "episode_nonce",
    )


def test_runtime_state_has_one_trajectory_and_one_batch_representation() -> None:
    assert tuple(field.name for field in fields(BatchRuntimeState)) == ("rows",)
    assert tuple(field.name for field in fields(TrajectoryRuntimeState)) == (
        "owner",
        "next_chunk_index",
        "slot_state",
        "temporal_cache",
        "e1_state",
        "e2_state",
        "state_bank",
        "identity_bank",
        "retrieval_history",
        "fast_weights",
        "optimizer",
        "reader_audit",
        "online_overlap_memory",
        "released",
    )


def test_removed_runtime_and_query_bridges_do_not_reappear() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src" / "ttt_svcbench_qwen").glob("*.py"))
    )
    for removed in (
        "RuntimeModelInput",
        "RuntimeQuerySpec",
        "CurrentQueryInput",
        "StageABatchRuntime",
        "StageARuntimeBridge",
        "PerVideoRuntimeState",
        "MetaModelRuntime",
    ):
        assert removed not in source
