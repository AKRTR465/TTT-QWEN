from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from torch import Tensor

from tests.support import parameter_count
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.qwen_adapter import MergedVideoMetadata
from ttt_svcbench_qwen.state_encoder import (
    TemporalCache,
    TemporalEncoderOutput,
    TemporalEventEncoder,
    build_temporal_encoder,
)

ROOT = Path(__file__).resolve().parents[1]
EXACT_PARAMETER_COUNT = 48_438_272
HIDDEN_DIM = 768
QUERY_DIM = 512


def make_metadata(time_count: int, height: int = 1, width: int = 1) -> MergedVideoMetadata:
    merged = torch.tensor([[time_count, height, width]], dtype=torch.int64)
    raw = merged.clone()
    raw[:, 1:] *= 2
    token_count = time_count * height * width
    return MergedVideoMetadata(
        video_grid_thw=raw,
        merged_grid_thw=merged,
        spatial_merge_size=2,
        token_counts=(token_count,),
        token_offsets=(0, token_count),
    )


def make_source(
    time_count: int,
    *,
    height: int = 1,
    width: int = 1,
    seed: int = 0,
    requires_grad: bool = False,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(
        time_count,
        height,
        width,
        4096,
        generator=generator,
    )
    return source.requires_grad_(requires_grad)


def make_query(seed: int = 0, *, requires_grad: bool = False) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(1, QUERY_DIM, generator=generator)
    return query.requires_grad_(requires_grad)


def encode_positions(
    encoder: TemporalEventEncoder,
    source: Tensor,
    positions: Sequence[int],
    q_target: Tensor,
    *,
    cache: TemporalCache | None = None,
    video_id: str = "video-a",
    trajectory_id: str = "trajectory-a",
    query_time: float | None = None,
    timestamp_values: Sequence[float] | None = None,
    timestamp_dtype: torch.dtype = torch.float64,
    detach_cache: bool = True,
) -> TemporalEncoderOutput:
    position_tuple = tuple(positions)
    if not position_tuple:
        raise ValueError("test helper requires at least one physical tubelet")
    selected = source[list(position_tuple)]
    time_count, height, width = selected.shape[:3]
    embeddings = selected.reshape(1, time_count * height * width, 4096)
    device = embeddings.device
    effective_timestamps = (
        tuple(position / 4.0 for position in position_tuple)
        if timestamp_values is None
        else tuple(timestamp_values)
    )
    if len(effective_timestamps) != time_count:
        raise ValueError("test timestamp_values must align to positions")
    timestamps = torch.tensor(
        effective_timestamps,
        dtype=timestamp_dtype,
        device=device,
    ).unsqueeze(0)
    position_ids = torch.tensor(position_tuple, dtype=torch.int64, device=device).unsqueeze(0)
    effective_query_time = float(position_tuple[-1]) / 4.0 if query_time is None else query_time
    return encoder(
        embeddings,
        torch.ones(1, embeddings.shape[1], dtype=torch.bool, device=device),
        make_metadata(time_count, height, width),
        torch.ones(1, time_count, dtype=torch.bool, device=device),
        timestamps,
        position_ids,
        torch.tensor([effective_query_time], dtype=torch.float32, device=device),
        q_target,
        (video_id,),
        (trajectory_id,),
        cache=cache,
        detach_cache=detach_cache,
    )


def encode_padded(
    encoder: TemporalEventEncoder,
    source: Tensor,
    q_target: Tensor,
    *,
    valid_count: int,
    cache: TemporalCache | None = None,
    query_time: float = 10.0,
    detach_cache: bool = True,
) -> TemporalEncoderOutput:
    time_count, height, width = source.shape[:3]
    embeddings = source.reshape(1, time_count * height * width, 4096)
    device = embeddings.device
    valid_mask = torch.zeros(1, time_count, dtype=torch.bool, device=device)
    valid_mask[:, :valid_count] = True
    timestamps = torch.full((1, time_count), -1.0, dtype=torch.float64, device=device)
    position_ids = torch.full((1, time_count), -1, dtype=torch.int64, device=device)
    if valid_count:
        timestamps[0, :valid_count] = (
            torch.arange(
                valid_count,
                dtype=torch.float64,
                device=device,
            )
            / 4.0
        )
        position_ids[0, :valid_count] = torch.arange(
            valid_count,
            dtype=torch.int64,
            device=device,
        )
    return encoder(
        embeddings,
        torch.ones(1, embeddings.shape[1], dtype=torch.bool, device=device),
        make_metadata(time_count, height, width),
        valid_mask,
        timestamps,
        position_ids,
        torch.tensor([query_time], dtype=torch.float32, device=device),
        q_target,
        ("video-a",),
        ("trajectory-a",),
        cache=cache,
        detach_cache=detach_cache,
    )


@pytest.fixture(scope="module")
def encoder() -> TemporalEventEncoder:
    torch.manual_seed(20260714)
    module = build_temporal_encoder(load_config())
    module.eval()
    return module


def test_meta_topology_and_exact_parameter_count() -> None:
    config = load_config()
    with torch.device("meta"):
        module = build_temporal_encoder(config)

    assert parameter_count(module) == EXACT_PARAMETER_COUNT
    assert parameter_count(module.spatial_pool) == 5_911_040
    assert len(module.layers) == 6
    assert [parameter_count(layer) for layer in module.layers] == [7_087_872] * 6
    assert all(layer.num_heads == 12 and layer.head_dim == 64 for layer in module.layers)
    assert all(layer.dropout == pytest.approx(0.1) for layer in module.layers)
    assert len({id(layer.q_projection.weight) for layer in module.layers}) == 6
    assert not any("position" in name for name, _ in module.named_parameters())


def test_demo_392_tokens_pool_to_eight_causal_states(
    encoder: TemporalEventEncoder,
) -> None:
    source = make_source(8, height=7, width=7, seed=1)
    q_target = make_query(2)

    with torch.inference_mode():
        output = encode_positions(encoder, source, range(8), q_target)

    assert output.hidden.shape == (1, 8, HIDDEN_DIM)
    assert output.valid_mask.all()
    assert output.position_ids.tolist() == [list(range(8))]
    assert output.cache.hidden.shape == (1, 8, HIDDEN_DIM)
    assert len(output.cache.layer_keys) == len(output.cache.layer_values) == 6
    assert all(value.shape == (1, 12, 8, 64) for value in output.cache.layer_keys)
    assert output.audit is not None
    assert output.audit.grid_shapes == ((8, 7, 7),)


def test_future_tubelets_cannot_change_past_outputs(
    encoder: TemporalEventEncoder,
) -> None:
    source = make_source(5, seed=3)
    changed = source.clone()
    changed[3:] = changed[3:] * -7.0 + 11.0
    q_target = make_query(4)

    with torch.inference_mode():
        baseline = encode_positions(encoder, source, range(5), q_target)
        perturbed = encode_positions(encoder, changed, range(5), q_target)

    torch.testing.assert_close(
        baseline.hidden[:, :3],
        perturbed.hidden[:, :3],
        atol=1.0e-6,
        rtol=1.0e-5,
    )


def test_eval_full_forward_matches_chunked_layerwise_kv(
    encoder: TemporalEventEncoder,
) -> None:
    source = make_source(8, seed=5)
    q_target = make_query(6)

    with torch.inference_mode():
        full = encode_positions(encoder, source, range(8), q_target)
        first = encode_positions(encoder, source, range(3), q_target, query_time=7.0 / 4.0)
        first_positions = first.cache.position_ids.clone()
        second = encode_positions(
            encoder,
            source,
            range(3, 8),
            q_target,
            cache=first.cache,
        )

    chunked = torch.cat((first.hidden, second.hidden), dim=1)
    torch.testing.assert_close(chunked, full.hidden, atol=1.0e-5, rtol=1.0e-4)
    assert torch.equal(first.cache.position_ids, first_positions)
    assert first.cache.layer_keys[0].untyped_storage().data_ptr() != (
        second.cache.layer_keys[0].untyped_storage().data_ptr()
    )
    assert second.cache.position_ids.tolist() == [list(range(8))]
    for full_keys, chunked_keys in zip(
        full.cache.layer_keys,
        second.cache.layer_keys,
        strict=True,
    ):
        torch.testing.assert_close(chunked_keys, full_keys, atol=1.0e-5, rtol=1.0e-4)


def test_four_tubelet_overlap_replays_and_replaces_without_duplicates(
    encoder: TemporalEventEncoder,
) -> None:
    source = make_source(12, seed=7)
    q_target = make_query(8)

    with torch.inference_mode():
        full = encode_positions(encoder, source, range(12), q_target)
        first = encode_positions(
            encoder,
            source,
            range(8),
            q_target,
            query_time=11.0 / 4.0,
        )
        replay = encode_positions(
            encoder,
            source,
            range(4, 12),
            q_target,
            cache=first.cache,
        )

    torch.testing.assert_close(replay.hidden, full.hidden[:, 4:], atol=1.0e-5, rtol=1.0e-4)
    assert replay.audit is not None
    assert replay.audit.overlap_replay_counts == (4,)
    assert replay.cache.position_ids.tolist() == [list(range(12))]
    assert replay.cache.total_seen.tolist() == [12]
    for full_keys, replayed_keys in zip(
        full.cache.layer_keys,
        replay.cache.layer_keys,
        strict=True,
    ):
        torch.testing.assert_close(replayed_keys, full_keys, atol=1.0e-5, rtol=1.0e-4)


def test_sixty_four_token_sliding_window_is_chunk_boundary_invariant_and_evicts(
    encoder: TemporalEventEncoder,
) -> None:
    source = make_source(72, seed=9)
    q_target = make_query(10)

    with torch.inference_mode():
        full = encode_positions(encoder, source, range(72), q_target)
        first = encode_positions(
            encoder,
            source,
            range(68),
            q_target,
            query_time=71.0 / 4.0,
        )
        overlap_tail = encode_positions(
            encoder,
            source,
            range(64, 72),
            q_target,
            cache=first.cache,
        )

    assert first.cache.position_ids.tolist() == [list(range(4, 68))]
    assert first.cache.cache_length == 64
    assert first.cache.replay_length == 3
    assert first.cache.replay_position_ids.tolist() == [[1, 2, 3]]
    chunked = torch.cat((first.hidden[:, :64], overlap_tail.hidden), dim=1)
    torch.testing.assert_close(chunked, full.hidden, atol=2.0e-5, rtol=2.0e-4)
    assert first.cache.cache_length == overlap_tail.cache.cache_length == 64
    assert overlap_tail.cache.position_ids.tolist() == [list(range(8, 72))]
    assert overlap_tail.cache.replay_position_ids.tolist() == [[5, 6, 7]]
    assert overlap_tail.cache.timestamps.tolist() == [[position / 4.0 for position in range(8, 72)]]
    assert overlap_tail.cache.total_seen.tolist() == [72]
    assert overlap_tail.audit is not None
    assert overlap_tail.audit.overlap_replay_counts == (4,)
    assert overlap_tail.audit.evicted_counts == (4,)


def test_cache_owner_query_and_query_time_drift_fail_closed(
    encoder: TemporalEventEncoder,
) -> None:
    source = make_source(3, seed=11)
    q_target = make_query(12)
    with torch.inference_mode():
        first = encode_positions(
            encoder,
            source,
            range(2),
            q_target,
            query_time=1.0,
        )

    with pytest.raises(ValueError, match="video owners|batch order"):
        encode_positions(
            encoder,
            source,
            (2,),
            q_target,
            cache=first.cache,
            video_id="video-b",
        )
    with pytest.raises(ValueError, match="trajectory owners|batch order"):
        encode_positions(
            encoder,
            source,
            (2,),
            q_target,
            cache=first.cache,
            trajectory_id="trajectory-b",
        )
    with pytest.raises(ValueError, match="query signature drift"):
        encode_positions(
            encoder,
            source,
            (2,),
            q_target + 1.0e-3,
            cache=first.cache,
        )
    with pytest.raises(ValueError, match="legal at query_time"):
        encode_positions(
            encoder,
            source,
            (2,),
            q_target,
            cache=first.cache,
            query_time=0.1,
        )
    with pytest.raises(ValueError, match="query_time|legal"):
        encode_positions(encoder, source, (0,), q_target, query_time=-0.1)


def test_reset_cache_is_empty_owned_and_storage_isolated(
    encoder: TemporalEventEncoder,
) -> None:
    q_target = make_query(30)[0]
    first = encoder.reset_cache("video-a", "trajectory-a", q_target)
    second = encoder.reset_cache("video-b", "trajectory-b", q_target)

    assert first.cache_length == first.replay_length == 0
    assert first.total_seen.tolist() == [0]
    assert first.video_ids == ("video-a",)
    assert first.trajectory_ids == ("trajectory-a",)
    assert torch.equal(first.query_signatures, q_target.unsqueeze(0))
    assert second.video_ids == ("video-b",)
    assert first.query_signatures.untyped_storage().data_ptr() != (
        second.query_signatures.untyped_storage().data_ptr()
    )


def test_tail_padding_and_all_invalid_rows_are_safe_and_do_not_enter_cache(
    encoder: TemporalEventEncoder,
) -> None:
    source = make_source(4, seed=13)
    poisoned = source.clone()
    poisoned[2] = torch.nan
    poisoned[3] = torch.inf
    invalid_valid_token = source.clone()
    invalid_valid_token[0] = torch.nan
    q_target = make_query(14)

    with torch.inference_mode():
        padded = encode_padded(encoder, source, q_target, valid_count=2)
        poisoned_output = encode_padded(encoder, poisoned, q_target, valid_count=2)
        empty = encode_padded(encoder, source, q_target, valid_count=0)
        empty_after_cache = encode_padded(
            encoder,
            source,
            q_target,
            valid_count=0,
            cache=padded.cache,
        )

    torch.testing.assert_close(padded.hidden, poisoned_output.hidden)
    assert torch.count_nonzero(padded.hidden[:, 2:]) == 0
    assert padded.timestamps.tolist() == [[0.0, 0.25, -1.0, -1.0]]
    assert padded.position_ids.tolist() == [[0, 1, -1, -1]]
    assert padded.cache.cache_length == 2
    assert torch.count_nonzero(empty.hidden) == 0
    assert empty.cache.cache_length == 0
    assert empty.cache.total_seen.tolist() == [0]
    assert torch.count_nonzero(empty_after_cache.hidden) == 0
    assert torch.equal(empty_after_cache.cache.hidden, padded.cache.hidden)
    assert torch.equal(empty_after_cache.cache.position_ids, padded.cache.position_ids)
    assert empty_after_cache.cache.hidden.untyped_storage().data_ptr() != (
        padded.cache.hidden.untyped_storage().data_ptr()
    )
    with pytest.raises(ValueError, match="finite"):
        encode_padded(encoder, invalid_valid_token, q_target, valid_count=2)


def test_heterogeneous_batch_pack_split_and_storage_are_isolated(
    encoder: TemporalEventEncoder,
) -> None:
    generator = torch.Generator().manual_seed(19)
    embeddings = torch.randn(2, 3, 4096, generator=generator)
    metadata = MergedVideoMetadata(
        video_grid_thw=torch.tensor([[2, 2, 2], [3, 2, 2]], dtype=torch.int64),
        merged_grid_thw=torch.tensor([[2, 1, 1], [3, 1, 1]], dtype=torch.int64),
        spatial_merge_size=2,
        token_counts=(2, 3),
        token_offsets=(0, 2, 5),
    )
    valid_mask = torch.tensor([[True, True, False], [True, True, True]])
    timestamps = torch.tensor([[0.0, 0.25, -1.0], [0.0, 0.25, 0.5]])
    position_ids = torch.tensor([[0, 1, -1], [0, 1, 2]], dtype=torch.int64)
    q_target = torch.randn(2, QUERY_DIM, generator=generator)

    with torch.inference_mode():
        output = encoder(
            embeddings,
            torch.tensor([[True, True, False], [True, True, True]]),
            metadata,
            valid_mask,
            timestamps,
            position_ids,
            torch.tensor([1.0, 1.0]),
            q_target,
            ("video-a", "video-b"),
            ("trajectory-a", "trajectory-b"),
        )

    assert output.hidden.shape == (2, 3, HIDDEN_DIM)
    assert torch.count_nonzero(output.hidden[0, 2]) == 0
    assert output.cache.valid_mask.tolist() == [[True, True, False], [True, True, True]]
    rows = output.cache.split()
    assert [row.cache_length for row in rows] == [2, 3]
    assert rows[0].hidden.untyped_storage().data_ptr() != (
        rows[1].hidden.untyped_storage().data_ptr()
    )
    assert rows[0].layer_keys[0].untyped_storage().data_ptr() != (
        rows[1].layer_keys[0].untyped_storage().data_ptr()
    )
    repacked = TemporalCache.pack(rows)
    assert torch.equal(repacked.hidden, output.cache.hidden)
    assert torch.equal(repacked.valid_mask, output.cache.valid_mask)
    assert repacked.hidden.untyped_storage().data_ptr() != (
        rows[0].hidden.untyped_storage().data_ptr()
    )
    round_trip = repacked.split()
    assert round_trip[0].video_ids == ("video-a",)
    assert round_trip[1].trajectory_ids == ("trajectory-b",)


def test_overlap_timestamp_mismatch_and_evicted_rewind_fail_closed(
    encoder: TemporalEventEncoder,
) -> None:
    source = make_source(12, seed=21)
    q_target = make_query(22)
    with torch.inference_mode():
        first = encode_positions(
            encoder,
            source,
            range(8),
            q_target,
            query_time=3.0,
        )

    mismatched_times = [1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
    with pytest.raises(ValueError, match="timestamps must match"):
        encode_positions(
            encoder,
            source,
            range(4, 12),
            q_target,
            cache=first.cache,
            query_time=3.0,
            timestamp_values=mismatched_times,
        )

    cached_positions = torch.arange(4, 68, dtype=torch.int64).unsqueeze(0)
    evicted_cache = TemporalCache(
        hidden=torch.zeros(1, 64, HIDDEN_DIM),
        layer_keys=tuple(torch.zeros(1, 12, 64, 64) for _ in range(6)),
        layer_values=tuple(torch.zeros(1, 12, 64, 64) for _ in range(6)),
        replay_layer_keys=tuple(torch.zeros(1, 12, 3, 64) for _ in range(6)),
        replay_layer_values=tuple(torch.zeros(1, 12, 3, 64) for _ in range(6)),
        timestamps=cached_positions.to(dtype=torch.float64) / 4.0,
        replay_timestamps=torch.tensor([[0.25, 0.5, 0.75]], dtype=torch.float64),
        position_ids=cached_positions,
        replay_position_ids=torch.tensor([[1, 2, 3]], dtype=torch.int64),
        valid_mask=torch.ones(1, 64, dtype=torch.bool),
        replay_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        video_ids=("video-a",),
        trajectory_ids=("trajectory-a",),
        query_signatures=q_target.detach().clone(),
        total_seen=torch.tensor([68], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="already-evicted"):
        encode_positions(
            encoder,
            source,
            (0,),
            q_target,
            cache=evicted_cache,
            query_time=20.0,
        )


def test_overlap_timestamp_identity_accepts_float32_then_float64_metadata(
    encoder: TemporalEventEncoder,
) -> None:
    source = make_source(12, seed=27)
    q_target = make_query(28)
    first_times = [position / 10.0 for position in range(8)]
    replay_times = [position / 10.0 for position in range(4, 12)]

    with torch.inference_mode():
        first = encode_positions(
            encoder,
            source,
            range(8),
            q_target,
            query_time=2.0,
            timestamp_values=first_times,
            timestamp_dtype=torch.float32,
        )
        replay = encode_positions(
            encoder,
            source,
            range(4, 12),
            q_target,
            cache=first.cache,
            query_time=2.0,
            timestamp_values=replay_times,
            timestamp_dtype=torch.float64,
        )

    assert replay.audit is not None
    assert replay.audit.overlap_replay_counts == (4,)
    assert replay.cache.position_ids.tolist() == [list(range(12))]
    assert replay.cache.timestamps.dtype == torch.float64


@pytest.mark.parametrize("grad_field", ["key", "value"])
def test_non_differentiable_cache_rejects_replay_tensors_with_grad(
    grad_field: str,
) -> None:
    main_positions = torch.arange(4, 68, dtype=torch.int64).unsqueeze(0)
    replay_positions = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    replay_keys = [torch.zeros(1, 12, 3, 64) for _ in range(6)]
    replay_values = [torch.zeros(1, 12, 3, 64) for _ in range(6)]
    target = replay_keys if grad_field == "key" else replay_values
    target[2].requires_grad_(True)

    with pytest.raises(ValueError, match="non-differentiable|detached"):
        TemporalCache(
            hidden=torch.zeros(1, 64, HIDDEN_DIM),
            layer_keys=tuple(torch.zeros(1, 12, 64, 64) for _ in range(6)),
            layer_values=tuple(torch.zeros(1, 12, 64, 64) for _ in range(6)),
            replay_layer_keys=tuple(replay_keys),
            replay_layer_values=tuple(replay_values),
            timestamps=main_positions.to(dtype=torch.float64) / 4.0,
            replay_timestamps=replay_positions.to(dtype=torch.float64) / 4.0,
            position_ids=main_positions,
            replay_position_ids=replay_positions,
            valid_mask=torch.ones(1, 64, dtype=torch.bool),
            replay_valid_mask=torch.ones(1, 3, dtype=torch.bool),
            video_ids=("video-a",),
            trajectory_ids=("trajectory-a",),
            query_signatures=torch.zeros(1, QUERY_DIM),
            total_seen=torch.tensor([68], dtype=torch.int64),
        )


def test_current_output_keeps_gradients_while_cache_detach_is_explicit(
    encoder: TemporalEventEncoder,
) -> None:
    encoder.zero_grad(set_to_none=True)
    source = make_source(2, width=2, seed=15, requires_grad=True)
    q_target = make_query(16, requires_grad=True)
    output = encode_positions(
        encoder,
        source,
        range(2),
        q_target,
        detach_cache=True,
    )
    output.hidden.square().mean().backward()

    assert source.grad is not None and bool(torch.isfinite(source.grad).all())
    assert q_target.grad is not None and bool(torch.isfinite(q_target.grad).all())
    assert float(source.grad.abs().sum()) > 0.0
    assert float(q_target.grad.abs().sum()) > 0.0
    assert encoder.spatial_pool.query_projection.weight.grad is not None
    assert encoder.layers[0].q_projection.weight.grad is not None
    assert encoder.layers[-1].ffn_out.weight.grad is not None
    assert not output.cache.hidden.requires_grad
    assert all(not value.requires_grad for value in output.cache.layer_keys)

    source_2 = make_source(1, width=2, seed=17, requires_grad=True)
    q_target_2 = make_query(18, requires_grad=True)
    differentiable = encode_positions(
        encoder,
        source_2,
        (0,),
        q_target_2,
        detach_cache=False,
    )
    assert differentiable.cache.differentiable
    assert differentiable.cache.hidden.requires_grad
    assert all(value.requires_grad for value in differentiable.cache.layer_keys)
    assert differentiable.hidden.shape == (1, 1, HIDDEN_DIM)
    assert bool(torch.isfinite(differentiable.hidden).all())


def test_float16_forward_backward_and_dtype_guards_are_finite(
    encoder: TemporalEventEncoder,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260715)
    module = build_temporal_encoder(load_config()).to(device=device, dtype=torch.float16).eval()
    source = make_source(1, width=2, seed=23).to(device=device, dtype=torch.float16)
    source.requires_grad_(True)
    q_target = make_query(24).to(device=device, dtype=torch.float16)
    q_target.requires_grad_(True)

    try:
        output = encode_positions(
            module,
            source,
            (0,),
            q_target,
            detach_cache=False,
        )
        output.hidden.float().square().mean().backward()
    except RuntimeError as error:
        unsupported_half = device.type == "cpu" and (
            "not implemented for 'Half'" in str(error) or "not implemented for Half" in str(error)
        )
        if unsupported_half:
            pytest.skip(f"CPU float16 kernel unavailable: {error}")
        raise

    assert output.hidden.dtype == torch.float16
    assert output.cache.hidden.dtype == torch.float16
    assert output.timestamps.dtype == torch.float64
    assert all(value.dtype == torch.float16 for value in output.cache.layer_keys)
    assert bool(torch.isfinite(output.hidden).all())
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())
    assert q_target.grad is not None and bool(torch.isfinite(q_target.grad).all())

    with pytest.raises(ValueError, match="dtype|share"):
        encode_positions(
            encoder,
            make_source(1, seed=25),
            (0,),
            make_query(26).double(),
        )


def test_temporal_builder_returns_registered_component() -> None:
    config = load_config()
    with torch.device("meta"):
        temporal = build_temporal_encoder(config)

    assert isinstance(temporal, TemporalEventEncoder)
    assert parameter_count(temporal) == EXACT_PARAMETER_COUNT
    with pytest.raises(ValueError, match="[Cc]onfig"):
        build_temporal_encoder()
