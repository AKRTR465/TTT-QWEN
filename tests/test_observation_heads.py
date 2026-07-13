from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.observation_heads import (
    E1PointEventDecoder,
    E1RuntimeState,
    E1SoftOutput,
    E2IntervalEventDecoder,
    E2RuntimeState,
    E2SoftOutput,
    ObservationHeads,
    build_observation_heads,
    observation_head_parameter_counts,
    observation_heads_parameter_count,
)
from ttt_svcbench_qwen.state_encoder import (
    SpatialEncoderOutput,
    TemporalCache,
    TemporalEncoderOutput,
)

EXACT_HEAD_COUNTS = {
    "o1": 2_632_710,
    "o2": 2_103_042,
    "e1": 9_584_643,
    "e2": 7_094_792,
}
EXACT_TOTAL = 21_415_187
HIDDEN_DIM = 768
QUERY_DIM = 512


@pytest.fixture(scope="module")
def heads() -> ObservationHeads:
    torch.manual_seed(20260716)
    module = build_observation_heads(load_config())
    module.eval()
    return module


def _empty_temporal_cache(
    query_signatures: Tensor,
    video_ids: tuple[str, ...],
    trajectory_ids: tuple[str, ...],
) -> TemporalCache:
    batch_size = query_signatures.shape[0]
    device = query_signatures.device
    dtype = query_signatures.dtype

    def empty_kv() -> Tensor:
        return torch.empty(batch_size, 12, 0, 64, dtype=dtype, device=device)

    return TemporalCache(
        hidden=torch.empty(batch_size, 0, HIDDEN_DIM, dtype=dtype, device=device),
        layer_keys=tuple(empty_kv() for _ in range(6)),
        layer_values=tuple(empty_kv() for _ in range(6)),
        replay_layer_keys=tuple(empty_kv() for _ in range(6)),
        replay_layer_values=tuple(empty_kv() for _ in range(6)),
        timestamps=torch.empty(batch_size, 0, dtype=torch.float64, device=device),
        replay_timestamps=torch.empty(batch_size, 0, dtype=torch.float64, device=device),
        position_ids=torch.empty(batch_size, 0, dtype=torch.int64, device=device),
        replay_position_ids=torch.empty(batch_size, 0, dtype=torch.int64, device=device),
        valid_mask=torch.empty(batch_size, 0, dtype=torch.bool, device=device),
        replay_valid_mask=torch.empty(batch_size, 0, dtype=torch.bool, device=device),
        video_ids=video_ids,
        trajectory_ids=trajectory_ids,
        query_signatures=query_signatures.detach().clone(),
        total_seen=torch.zeros(batch_size, dtype=torch.int64, device=device),
    )


def _typed_encoder_outputs(
    slots: Tensor,
    slot_mask: Tensor,
    hidden: Tensor,
    temporal_mask: Tensor,
    q_target: Tensor,
    video_ids: tuple[str, ...],
    trajectory_ids: tuple[str, ...],
) -> tuple[SpatialEncoderOutput, TemporalEncoderOutput]:
    batch_size, time_count = temporal_mask.shape
    timestamps = torch.full(
        (batch_size, time_count),
        -1.0,
        dtype=torch.float64,
        device=hidden.device,
    )
    position_ids = torch.full(
        (batch_size, time_count),
        -1,
        dtype=torch.int64,
        device=hidden.device,
    )
    for row in range(batch_size):
        count = int(temporal_mask[row].sum().item())
        timestamps[row, :count] = torch.arange(
            count, dtype=torch.float64, device=hidden.device
        ) / 4.0
        position_ids[row, :count] = torch.arange(
            count, dtype=torch.int64, device=hidden.device
        )
    spatial = SpatialEncoderOutput(
        slots=slots,
        slot_valid_mask=slot_mask,
        active_slot_overflow_count=torch.zeros(
            batch_size, dtype=torch.int64, device=slots.device
        ),
    )
    temporal = TemporalEncoderOutput(
        hidden=hidden,
        timestamps=timestamps,
        position_ids=position_ids,
        valid_mask=temporal_mask,
        cache=_empty_temporal_cache(q_target, video_ids, trajectory_ids),
    )
    return spatial, temporal


def _stream_tensors(
    source: Tensor,
    positions: Sequence[int],
    *,
    timestamp_values: Sequence[float] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    selected = source[list(positions)].unsqueeze(0)
    count = selected.shape[1]
    valid_mask = torch.ones(1, count, dtype=torch.bool, device=source.device)
    effective_times = (
        tuple(float(position) / 4.0 for position in positions)
        if timestamp_values is None
        else tuple(timestamp_values)
    )
    if len(effective_times) != count:
        raise ValueError("test timestamps must align to positions")
    timestamps = torch.tensor(
        effective_times,
        dtype=torch.float64,
        device=source.device,
    ).unsqueeze(0)
    position_ids = torch.tensor(
        tuple(positions),
        dtype=torch.int64,
        device=source.device,
    ).unsqueeze(0)
    return selected, valid_mask, timestamps, position_ids


def _run_e1(
    decoder: E1PointEventDecoder,
    source: Tensor,
    positions: Sequence[int],
    query: Tensor,
    *,
    prior: E1RuntimeState | None = None,
    video_id: str = "video-a",
    trajectory_id: str = "trajectory-a",
    timestamp_values: Sequence[float] | None = None,
) -> E1SoftOutput:
    hidden, mask, timestamps, position_ids = _stream_tensors(
        source,
        positions,
        timestamp_values=timestamp_values,
    )
    return decoder(
        hidden,
        mask,
        timestamps,
        position_ids,
        (video_id,),
        (trajectory_id,),
        query,
        prior_states=(prior,),
    )


def _run_e2(
    decoder: E2IntervalEventDecoder,
    source: Tensor,
    positions: Sequence[int],
    query: Tensor,
    *,
    prior: E2RuntimeState | None = None,
    video_id: str = "video-a",
    trajectory_id: str = "trajectory-a",
    timestamp_values: Sequence[float] | None = None,
) -> E2SoftOutput:
    hidden, mask, timestamps, position_ids = _stream_tensors(
        source,
        positions,
        timestamp_values=timestamp_values,
    )
    return decoder(
        hidden,
        mask,
        timestamps,
        position_ids,
        (video_id,),
        (trajectory_id,),
        query,
        prior_states=(prior,),
    )


def test_meta_topology_builder_and_exact_parameter_counts() -> None:
    config = load_config()
    with torch.device("meta"):
        module = build_observation_heads(config)

    assert isinstance(module, ObservationHeads)
    assert set(dict(module.named_children())) == {"o1", "o2", "e1", "e2"}
    assert observation_head_parameter_counts(module) == EXACT_HEAD_COUNTS
    assert observation_heads_parameter_count(module) == EXACT_TOTAL
    assert [block.dilation for block in module.e1.blocks] == [1, 2, 4, 8, 16]
    assert [block.left_padding for block in module.e1.blocks] == [2, 4, 8, 16, 32]
    assert all(block.filter_conv.bias is not None for block in module.e1.blocks)
    assert all(block.gate_conv.bias is not None for block in module.e1.blocks)
    assert all(block.residual_projection.bias is not None for block in module.e1.blocks)
    assert not any(isinstance(child, (nn.BatchNorm1d, nn.Dropout)) for child in module.modules())
    assert module.e2.gru.input_size == module.e2.gru.hidden_size == HIDDEN_DIM
    assert module.e2.gru.num_layers == 2
    assert module.e2.gru.batch_first is True
    assert module.e2.gru.bidirectional is False
    assert module.e2.gru.dropout == 0.0
    with pytest.raises(ValueError, match="[Cc]onfig"):
        build_observation_heads()


def test_registered_forward_shapes_masks_metadata_and_probabilities(
    heads: ObservationHeads,
) -> None:
    generator = torch.Generator().manual_seed(11)
    slots = torch.randn(2, 4, HIDDEN_DIM, generator=generator)
    slot_mask = torch.tensor(
        [[True, False, True, True], [True, True, False, True]],
        dtype=torch.bool,
    )
    hidden = torch.randn(2, 4, HIDDEN_DIM, generator=generator)
    hidden[1] = 0.0
    temporal_mask = torch.tensor(
        [[True, True, True, True], [False, False, False, False]],
        dtype=torch.bool,
    )
    q_target = torch.randn(2, QUERY_DIM, generator=generator)
    videos = ("video-a", "video-b")
    trajectories = ("trajectory-a", "trajectory-b")
    spatial, temporal = _typed_encoder_outputs(
        slots,
        slot_mask,
        hidden,
        temporal_mask,
        q_target,
        videos,
        trajectories,
    )

    with torch.no_grad():
        output = heads(spatial, temporal, q_target, videos, trajectories)

    effective_slot_mask = slot_mask & temporal_mask.any(dim=1, keepdim=True)
    assert output.o1.logits.shape == (2, 4, 6)
    assert output.o2.identity.shape == (2, 4, 256)
    assert output.o2.score_logits.shape == (2, 4, 2)
    assert output.e1.logits.shape == (2, 4, 3)
    assert output.e2.event_logits.shape == output.e2.phase_logits.shape == (2, 4, 4)
    assert torch.equal(output.o1.valid_mask, effective_slot_mask)
    assert torch.equal(output.o2.valid_mask, effective_slot_mask)
    assert torch.equal(output.e1.valid_mask, temporal_mask)
    assert torch.equal(output.e2.valid_mask, temporal_mask)
    assert torch.all(output.o1.timestamps[0, effective_slot_mask[0]] == 0.75)
    assert torch.all(output.o1.timestamps[~effective_slot_mask] == -1.0)
    assert torch.all(output.o1.position_ids[0, effective_slot_mask[0]] == 3)
    assert torch.all(output.o1.position_ids[~effective_slot_mask] == -1)
    assert torch.equal(output.o1.timestamps, output.o2.timestamps)
    assert torch.equal(output.o1.position_ids, output.o2.position_ids)
    assert torch.equal(output.e1.timestamps, temporal.timestamps)
    assert torch.equal(output.e2.timestamps, temporal.timestamps)
    assert torch.equal(output.e1.position_ids, temporal.position_ids)
    assert torch.equal(output.e2.position_ids, temporal.position_ids)
    assert torch.count_nonzero(output.o1.logits[~effective_slot_mask]) == 0
    assert torch.count_nonzero(output.o2.identity[~effective_slot_mask]) == 0
    assert torch.count_nonzero(output.e1.logits[~temporal_mask]) == 0
    assert torch.count_nonzero(output.e2.event_logits[~temporal_mask]) == 0
    assert torch.count_nonzero(output.e2.phase_probabilities[~temporal_mask]) == 0
    torch.testing.assert_close(
        output.o1.probabilities[effective_slot_mask],
        torch.sigmoid(output.o1.logits[effective_slot_mask]),
    )
    torch.testing.assert_close(
        output.o2.score_probabilities[effective_slot_mask],
        torch.sigmoid(output.o2.score_logits[effective_slot_mask]),
    )
    torch.testing.assert_close(
        output.e1.probabilities[temporal_mask],
        torch.sigmoid(output.e1.logits[temporal_mask]),
    )
    torch.testing.assert_close(
        output.e2.event_probabilities[temporal_mask],
        torch.sigmoid(output.e2.event_logits[temporal_mask]),
    )
    torch.testing.assert_close(
        output.e2.phase_probabilities[temporal_mask].sum(dim=-1),
        torch.ones(int(temporal_mask.sum().item())),
    )
    identity_norms = torch.linalg.vector_norm(
        output.o2.identity[effective_slot_mask].float(), dim=-1
    )
    torch.testing.assert_close(identity_norms, torch.ones_like(identity_norms))
    assert output.o1.LOGIT_NAMES == (
        "object",
        "target",
        "visible",
        "enter",
        "exit",
        "confidence",
    )
    assert output.o2.SCORE_NAMES == ("novelty", "match_confidence")
    assert output.e1.LOGIT_NAMES == ("eventness", "completion", "transition")
    assert output.e2.EVENT_NAMES == ("start", "active", "end", "complete")
    assert output.e2.PHASE_NAMES == ("inactive", "active", "end_candidate", "completed")
    assert output.e1.next_states[1].total_seen == 0
    assert output.e2.next_states[1].total_seen == 0
    with pytest.raises(ValueError, match="exactly match temporal cache owners"):
        heads(
            spatial,
            temporal,
            q_target,
            tuple(reversed(videos)),
            trajectories,
        )


def test_invalid_slot_and_temporal_padding_are_poison_safe(
    heads: ObservationHeads,
) -> None:
    generator = torch.Generator().manual_seed(17)
    slots = torch.randn(1, 3, HIDDEN_DIM, generator=generator)
    poisoned_slots = slots.clone()
    poisoned_slots[:, 1] = torch.nan
    poisoned_slots[:, 2] = torch.inf
    slot_mask = torch.tensor([[True, False, False]])
    query = torch.randn(1, QUERY_DIM, generator=generator)
    observation_time = torch.tensor([1.25], dtype=torch.float64)
    observation_position = torch.tensor([5], dtype=torch.int64)
    hidden = torch.randn(2, 3, HIDDEN_DIM, generator=generator)
    hidden[0, 1] = torch.nan
    hidden[0, 2] = torch.inf
    hidden[1] = torch.nan
    temporal_mask = torch.tensor([[True, False, False], [False, False, False]])
    timestamps = torch.tensor([[0.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], dtype=torch.float64)
    position_ids = torch.tensor([[0, -1, -1], [-1, -1, -1]], dtype=torch.int64)
    signatures = torch.randn(2, QUERY_DIM, generator=generator)

    with torch.no_grad():
        o1 = heads.o1(
            poisoned_slots,
            slot_mask,
            query,
            observation_time,
            observation_position,
        )
        o2 = heads.o2(
            poisoned_slots,
            slot_mask,
            observation_time,
            observation_position,
        )
        e1 = heads.e1(
            hidden,
            temporal_mask,
            timestamps,
            position_ids,
            ("video-a", "video-b"),
            ("trajectory-a", "trajectory-b"),
            signatures,
        )
        e2 = heads.e2(
            hidden,
            temporal_mask,
            timestamps,
            position_ids,
            ("video-a", "video-b"),
            ("trajectory-a", "trajectory-b"),
            signatures,
        )

    for tensor in (
        o1.logits,
        o1.probabilities,
        o2.identity,
        o2.score_logits,
        e1.logits,
        e2.event_logits,
        e2.phase_logits,
    ):
        assert bool(torch.isfinite(tensor).all())
    assert torch.count_nonzero(o1.logits[:, 1:]) == 0
    assert torch.count_nonzero(o2.identity[:, 1:]) == 0
    assert torch.count_nonzero(e1.logits[~temporal_mask]) == 0
    assert torch.count_nonzero(e2.event_logits[~temporal_mask]) == 0
    assert e1.next_states[1].total_seen == e2.next_states[1].total_seen == 0


def test_o1_film_formula_soft_count_query_isolation_and_gradients(
    heads: ObservationHeads,
) -> None:
    decoder = heads.o1
    decoder.zero_grad(set_to_none=True)
    generator = torch.Generator().manual_seed(23)
    slots = torch.randn(2, 3, HIDDEN_DIM, generator=generator, requires_grad=True)
    q_target = torch.randn(2, QUERY_DIM, generator=generator, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    timestamps = torch.tensor([1.0, 2.0], dtype=torch.float64)
    position_ids = torch.tensor([4, 8], dtype=torch.int64)
    output = decoder(slots, mask, q_target, timestamps, position_ids)

    safe_slots = torch.where(mask.unsqueeze(-1), slots, 0.0)
    scale, shift = decoder.film_projection(q_target).chunk(2, dim=-1)
    conditioned = decoder.slot_norm(safe_slots) * (1.0 + scale.unsqueeze(1))
    conditioned = conditioned + shift.unsqueeze(1)
    expected = decoder.output_projection(
        torch.nn.functional.silu(
            decoder.mlp_2(torch.nn.functional.silu(decoder.mlp_1(conditioned)))
        )
    )
    expected = torch.where(mask.unsqueeze(-1), expected, 0.0)
    torch.testing.assert_close(output.logits, expected)
    expected_count = (
        output.probabilities[..., 0]
        * output.probabilities[..., 1]
        * output.probabilities[..., 2]
        * mask
    ).sum(dim=1)
    torch.testing.assert_close(output.soft_count, expected_count)

    perturbed_query = q_target.detach().clone()
    perturbed_query[0] += 4.0
    with torch.no_grad():
        perturbed = decoder(
            slots.detach(),
            mask,
            perturbed_query,
            timestamps,
            position_ids,
        )
    assert not torch.allclose(output.logits.detach()[0], perturbed.logits[0])
    torch.testing.assert_close(output.logits.detach()[1], perturbed.logits[1])

    (output.logits.square().mean() + output.soft_count.mean()).backward()
    assert slots.grad is not None and bool(torch.isfinite(slots.grad).all())
    assert q_target.grad is not None and bool(torch.isfinite(q_target.grad).all())
    assert float(slots.grad.abs().sum()) > 0.0
    assert float(q_target.grad.abs().sum()) > 0.0
    assert decoder.film_projection.weight.grad is not None


def test_o2_zero_identity_fallback_and_raw_score_logits(heads: ObservationHeads) -> None:
    decoder = heads.o2
    saved = {
        "identity_weight": decoder.identity_projection.weight.detach().clone(),
        "identity_bias": decoder.identity_projection.bias.detach().clone(),
        "score_weight": decoder.score_projection.weight.detach().clone(),
        "score_bias": decoder.score_projection.bias.detach().clone(),
    }
    try:
        with torch.no_grad():
            decoder.identity_projection.weight.zero_()
            decoder.identity_projection.bias.zero_()
            decoder.score_projection.weight.zero_()
            decoder.score_projection.bias.copy_(torch.tensor([-2.0, 3.0]))
        slots = torch.randn(1, 3, HIDDEN_DIM)
        mask = torch.tensor([[True, False, True]])
        output = decoder(
            slots,
            mask,
            torch.tensor([2.5], dtype=torch.float64),
            torch.tensor([10], dtype=torch.int64),
        )

        expected_identity = torch.zeros(2, 256)
        expected_identity[:, 0] = 1.0
        torch.testing.assert_close(output.identity[mask], expected_identity)
        assert torch.count_nonzero(output.identity[~mask]) == 0
        torch.testing.assert_close(
            output.score_logits[mask],
            torch.tensor([[-2.0, 3.0], [-2.0, 3.0]]),
        )
        torch.testing.assert_close(
            output.score_probabilities[mask],
            torch.sigmoid(torch.tensor([[-2.0, 3.0], [-2.0, 3.0]])),
        )
        assert output.score is output.score_logits
    finally:
        with torch.no_grad():
            decoder.identity_projection.weight.copy_(saved["identity_weight"])
            decoder.identity_projection.bias.copy_(saved["identity_bias"])
            decoder.score_projection.weight.copy_(saved["score_weight"])
            decoder.score_projection.bias.copy_(saved["score_bias"])


def test_e1_causality_full_disjoint_and_four_overlap_replay(
    heads: ObservationHeads,
) -> None:
    source = torch.randn(8, HIDDEN_DIM, generator=torch.Generator().manual_seed(31))
    query = torch.randn(1, QUERY_DIM, generator=torch.Generator().manual_seed(32))
    with torch.no_grad():
        full = _run_e1(heads.e1, source, range(8), query)
        first = _run_e1(heads.e1, source, range(6), query)
        overlap = _run_e1(heads.e1, source, range(2, 8), query, prior=first.next_states[0])
        disjoint_first = _run_e1(heads.e1, source, range(4), query)
        disjoint_second = _run_e1(
            heads.e1,
            source,
            range(4, 8),
            query,
            prior=disjoint_first.next_states[0],
        )
        future_mutation = source.clone()
        future_mutation[4:] += 100.0
        mutated = _run_e1(heads.e1, future_mutation, range(8), query)

    torch.testing.assert_close(first.logits, full.logits[:, :6], atol=2.0e-6, rtol=2.0e-5)
    torch.testing.assert_close(overlap.logits, full.logits[:, 2:], atol=2.0e-6, rtol=2.0e-5)
    torch.testing.assert_close(
        disjoint_first.logits,
        full.logits[:, :4],
        atol=2.0e-6,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        disjoint_second.logits,
        full.logits[:, 4:],
        atol=2.0e-6,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(mutated.logits[:, :4], full.logits[:, :4])
    assert overlap.audit.overlap_replay_counts == (4,)
    assert disjoint_second.audit.overlap_replay_counts == (0,)
    assert overlap.next_states[0].total_seen == 8
    assert overlap.next_states[0].position_ids.tolist() == list(range(8))
    with pytest.raises(ValueError, match="configured replay window"):
        _run_e1(
            heads.e1,
            source,
            range(1, 8),
            query,
            prior=first.next_states[0],
        )

    with pytest.raises(ValueError, match="owner"):
        _run_e1(
            heads.e1,
            source,
            range(6, 8),
            query,
            prior=first.next_states[0],
            video_id="video-other",
        )
    with pytest.raises(ValueError, match="query signature drift"):
        _run_e1(
            heads.e1,
            source,
            range(6, 8),
            query + 1.0,
            prior=first.next_states[0],
        )
    bad_times = [position / 4.0 for position in range(2, 8)]
    bad_times[0] += 0.1
    with pytest.raises(ValueError, match="overlap timestamps"):
        _run_e1(
            heads.e1,
            source,
            range(2, 8),
            query,
            prior=first.next_states[0],
            timestamp_values=bad_times,
        )


def test_e2_causality_full_disjoint_and_four_overlap_checkpoint_replay(
    heads: ObservationHeads,
) -> None:
    source = torch.randn(8, HIDDEN_DIM, generator=torch.Generator().manual_seed(37))
    query = torch.randn(1, QUERY_DIM, generator=torch.Generator().manual_seed(38))
    with torch.no_grad():
        full = _run_e2(heads.e2, source, range(8), query)
        first = _run_e2(heads.e2, source, range(6), query)
        overlap = _run_e2(heads.e2, source, range(2, 8), query, prior=first.next_states[0])
        disjoint_first = _run_e2(heads.e2, source, range(4), query)
        disjoint_second = _run_e2(
            heads.e2,
            source,
            range(4, 8),
            query,
            prior=disjoint_first.next_states[0],
        )
        future_mutation = source.clone()
        future_mutation[4:] -= 100.0
        mutated = _run_e2(heads.e2, future_mutation, range(8), query)

    for field in ("event_logits", "phase_logits"):
        reference = getattr(full, field)
        torch.testing.assert_close(getattr(first, field), reference[:, :6])
        torch.testing.assert_close(getattr(overlap, field), reference[:, 2:])
        torch.testing.assert_close(getattr(disjoint_first, field), reference[:, :4])
        torch.testing.assert_close(getattr(disjoint_second, field), reference[:, 4:])
        torch.testing.assert_close(getattr(mutated, field)[:, :4], reference[:, :4])
    assert overlap.audit.overlap_replay_counts == (4,)
    assert disjoint_second.audit.overlap_replay_counts == (0,)
    assert overlap.next_states[0].total_seen == 8
    assert overlap.next_states[0].position_ids.tolist() == [3, 4, 5, 6, 7]
    torch.testing.assert_close(
        overlap.next_states[0].hidden,
        overlap.next_states[0].checkpoint_hidden[-1],
    )
    torch.testing.assert_close(overlap.next_states[0].hidden, full.next_states[0].hidden)

    with pytest.raises(ValueError, match="owner"):
        _run_e2(
            heads.e2,
            source,
            range(6, 8),
            query,
            prior=first.next_states[0],
            trajectory_id="trajectory-other",
        )
    with pytest.raises(ValueError, match="query signature drift"):
        _run_e2(
            heads.e2,
            source,
            range(6, 8),
            query + 1.0,
            prior=first.next_states[0],
        )
    bad_times = [position / 4.0 for position in range(2, 8)]
    bad_times[0] += 0.1
    with pytest.raises(ValueError, match="overlap timestamps"):
        _run_e2(
            heads.e2,
            source,
            range(2, 8),
            query,
            prior=first.next_states[0],
            timestamp_values=bad_times,
        )


def test_online_freeze_preserves_gradients_to_all_decoder_inputs(
    heads: ObservationHeads,
) -> None:
    heads.zero_grad(set_to_none=True)
    heads.set_online_frozen(True)
    generator = torch.Generator().manual_seed(43)
    slots = torch.randn(1, 3, HIDDEN_DIM, generator=generator, requires_grad=True)
    hidden = torch.randn(1, 4, HIDDEN_DIM, generator=generator, requires_grad=True)
    q_target = torch.randn(1, QUERY_DIM, generator=generator, requires_grad=True)
    videos = ("video-grad",)
    trajectories = ("trajectory-grad",)
    spatial, temporal = _typed_encoder_outputs(
        slots,
        torch.ones(1, 3, dtype=torch.bool),
        hidden,
        torch.ones(1, 4, dtype=torch.bool),
        q_target,
        videos,
        trajectories,
    )
    try:
        output = heads(spatial, temporal, q_target, videos, trajectories)
        loss = (
            output.o1.logits.square().mean()
            + output.o1.soft_count.mean()
            + output.o2.score_logits.square().mean()
            + output.e1.logits.square().mean()
            + output.e2.event_logits.square().mean()
            + output.e2.phase_logits.square().mean()
        )
        loss.backward()

        assert heads.online_frozen
        assert all(not parameter.requires_grad for parameter in heads.parameters())
        assert all(parameter.grad is None for parameter in heads.parameters())
        for gradient in (slots.grad, hidden.grad, q_target.grad):
            assert gradient is not None and bool(torch.isfinite(gradient).all())
            assert float(gradient.abs().sum()) > 0.0
        assert not output.e1.next_states[0].projected_history.requires_grad
        assert not output.e2.next_states[0].hidden.requires_grad
    finally:
        heads.set_online_frozen(False)
