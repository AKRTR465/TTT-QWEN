from __future__ import annotations

import copy
from collections.abc import Sequence

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tests.support import parameter_count
from tests.support.runtime_factories import make_temporal_cache
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.observation_heads import (
    E1RuntimeState,
    E2RuntimeState,
    ObservationHeads,
    build_observation_heads,
)
from ttt_svcbench_qwen.state_encoder import SpatialEncoderOutput, TemporalEncoderOutput

EXACT_HEAD_COUNTS = {
    "o1": 2_632_710,
    "o2": 2_631_171,
    "e1": 9_717_252,
    "e2": 7_293_449,
}
EXACT_TOTAL = 22_274_582
HIDDEN_DIM = 768
QUERY_DIM = 512


@pytest.fixture(scope="module")
def heads() -> ObservationHeads:
    torch.manual_seed(20260716)
    module = build_observation_heads(load_config())
    module.eval()
    return module


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
    timestamps = torch.full((batch_size, time_count), -1.0, dtype=torch.float64)
    position_ids = torch.full((batch_size, time_count), -1, dtype=torch.int64)
    for row in range(batch_size):
        count = int(temporal_mask[row].sum().item())
        timestamps[row, :count] = torch.arange(count, dtype=torch.float64) / 4.0
        position_ids[row, :count] = torch.arange(count, dtype=torch.int64)
    spatial = SpatialEncoderOutput(
        slots=slots,
        slot_valid_mask=slot_mask,
        active_slot_overflow_count=torch.zeros(batch_size, dtype=torch.int64),
    )
    temporal = TemporalEncoderOutput(
        hidden=hidden,
        timestamps=timestamps,
        position_ids=position_ids,
        valid_mask=temporal_mask,
        cache=make_temporal_cache(
            hidden=torch.empty(batch_size, 0, HIDDEN_DIM, dtype=q_target.dtype),
            video_ids=video_ids,
            trajectory_ids=trajectory_ids,
            query_signatures=q_target,
        ),
    )
    return spatial, temporal


def _run_stream(
    decoder: nn.Module,
    source: Tensor,
    positions: Sequence[int],
    query: Tensor,
    *,
    prior: E1RuntimeState | E2RuntimeState | None = None,
):
    """Drive one E1/E2 stream row over ``positions`` of ``source`` (identical signatures)."""

    selected = list(positions)
    hidden = source[selected].unsqueeze(0)
    mask = torch.ones(1, len(selected), dtype=torch.bool)
    timestamps = torch.tensor(
        [position / 4.0 for position in selected], dtype=torch.float64
    ).unsqueeze(0)
    position_ids = torch.tensor(selected, dtype=torch.int64).unsqueeze(0)
    return decoder(
        hidden,
        mask,
        timestamps,
        position_ids,
        ("video-a",),
        ("trajectory-a",),
        query,
        prior_states=(prior,),
    )


def test_meta_topology_builder_and_exact_parameter_counts() -> None:
    with torch.device("meta"):
        module = build_observation_heads(load_config())

    assert isinstance(module, ObservationHeads)
    assert set(dict(module.named_children())) == {"o1", "o2", "e1", "e2"}
    assert {name: parameter_count(getattr(module, name)) for name in EXACT_HEAD_COUNTS} == (
        EXACT_HEAD_COUNTS
    )
    assert parameter_count(module) == EXACT_TOTAL
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


def test_registered_forward_shapes_masks_and_metadata(heads: ObservationHeads) -> None:
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

    # A row with no valid temporal state cannot carry a valid slot observation.
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

    # Slot observations are stamped with the last valid temporal position of their row.
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

    # Padding never carries signal, and an all-padding row advances no stream state.
    assert torch.count_nonzero(output.o1.logits[~effective_slot_mask]) == 0
    assert torch.count_nonzero(output.o2.identity[~effective_slot_mask]) == 0
    assert torch.count_nonzero(output.e1.logits[~temporal_mask]) == 0
    assert torch.count_nonzero(output.e2.event_logits[~temporal_mask]) == 0
    assert torch.count_nonzero(output.e2.phase_probabilities[~temporal_mask]) == 0
    assert output.e1.next_states[1].total_seen == 0
    assert output.e2.next_states[1].total_seen == 0

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


def test_invalid_slot_and_temporal_padding_are_poison_safe(heads: ObservationHeads) -> None:
    """NaN/Inf parked in masked padding must be zeroed before it reaches any norm."""

    generator = torch.Generator().manual_seed(17)
    poisoned_slots = torch.randn(1, 3, HIDDEN_DIM, generator=generator)
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
    owners = (("video-a", "video-b"), ("trajectory-a", "trajectory-b"))

    with torch.no_grad():
        o1 = heads.o1(poisoned_slots, slot_mask, query, observation_time, observation_position)
        o2 = heads.o2(poisoned_slots, slot_mask, observation_time, observation_position)
        e1 = heads.e1(hidden, temporal_mask, timestamps, position_ids, *owners, signatures)
        e2 = heads.e2(hidden, temporal_mask, timestamps, position_ids, *owners, signatures)

    for tensor in (o1.logits, o2.identity, o2.score_logits, e1.logits, e2.event_logits):
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
    expected = decoder.output_projection(F.silu(decoder.mlp_2(F.silu(decoder.mlp_1(conditioned)))))
    expected = torch.where(mask.unsqueeze(-1), expected, 0.0)
    torch.testing.assert_close(output.logits, expected)

    # soft_count is the masked product of object x target x visible, never a threshold count.
    expected_count = (
        output.probabilities[..., 0]
        * output.probabilities[..., 1]
        * output.probabilities[..., 2]
        * mask
    ).sum(dim=1)
    torch.testing.assert_close(output.soft_count, expected_count)

    # Per-row query isolation: perturbing row 0's query leaves row 1 bitwise identical.
    perturbed_query = q_target.detach().clone()
    perturbed_query[0] += 4.0
    with torch.no_grad():
        perturbed = decoder(slots.detach(), mask, perturbed_query, timestamps, position_ids)
    assert not torch.allclose(output.logits.detach()[0], perturbed.logits[0])
    torch.testing.assert_close(output.logits.detach()[1], perturbed.logits[1])

    (output.logits.square().mean() + output.soft_count.mean()).backward()
    assert slots.grad is not None and bool(torch.isfinite(slots.grad).all())
    assert q_target.grad is not None and bool(torch.isfinite(q_target.grad).all())
    assert float(slots.grad.abs().sum()) > 0.0
    assert float(q_target.grad.abs().sum()) > 0.0
    assert decoder.film_projection.weight.grad is not None


def test_o2_relevance_is_query_conditioned_and_pooled_count_is_exact(
    heads: ObservationHeads,
) -> None:
    decoder = heads.o2
    decoder.zero_grad(set_to_none=True)
    generator = torch.Generator().manual_seed(31)
    slots = torch.randn(2, 3, HIDDEN_DIM, generator=generator)
    q_target = torch.randn(2, QUERY_DIM, generator=generator, requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    timestamps = torch.tensor([1.0, 2.0], dtype=torch.float64)
    position_ids = torch.tensor([4, 8], dtype=torch.int64)

    output = decoder(slots, mask, timestamps, position_ids, q_target=q_target)

    # Multiplicative relevance: sigma(<identity_i, W q_target>), masked to valid slots.
    # This is the weight the O2-Unique soft dedup target rides on, so the exact form
    # (and identity's gradient into it) is mechanism, not a metric.
    expected = torch.sigmoid(
        torch.einsum(
            "bnd,bd->bn",
            output.identity.float(),
            decoder.relevance_projection(q_target).float(),
        )
    ).to(dtype=output.identity.dtype)
    expected = torch.where(mask, expected, torch.zeros_like(expected))
    torch.testing.assert_close(output.relevance, expected)
    assert bool(torch.all(output.relevance[~mask] == 0.0))

    # O2-Gain pooled count regression: masked-mean trunk features concatenated with
    # q_target through the softplus count head, independent of the padded slot width.
    trunk = F.silu(decoder.trunk_2(F.silu(decoder.trunk_1(decoder.slot_norm(
        torch.where(mask.unsqueeze(-1), slots, 0.0)
    )))))
    weights = mask.unsqueeze(-1).to(dtype=trunk.dtype)
    pooled = (trunk * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    torch.testing.assert_close(
        output.count_prediction,
        decoder.count_head(torch.cat((pooled, q_target), dim=-1)),
    )
    assert output.count_prediction.shape == (2,)
    assert bool(torch.all(output.count_prediction > 0.0))

    # Same slots, different query: the multiplicative head must move the perturbed
    # row's relevance while leaving the untouched row bitwise identical.
    perturbed_query = q_target.detach().clone()
    perturbed_query[0] += 4.0
    with torch.no_grad():
        perturbed = decoder(slots, mask, timestamps, position_ids, q_target=perturbed_query)
    assert not torch.allclose(
        output.relevance.detach()[0][mask[0]],
        perturbed.relevance[0][mask[0]],
    )
    torch.testing.assert_close(output.relevance.detach()[1], perturbed.relevance[1])

    (output.relevance.sum() + output.count_prediction.sum()).backward()
    assert q_target.grad is not None and float(q_target.grad.abs().sum()) > 0.0
    assert decoder.relevance_projection.weight.grad is not None
    # identity is not detached at the einsum, so the relevance-weighted dedup target
    # pushes task gradient back through o2.identity itself.
    identity_grad = decoder.identity_projection.weight.grad
    assert identity_grad is not None and float(identity_grad.abs().sum()) > 0.0


def test_o2_zero_identity_fallback_and_raw_score_logits(heads: ObservationHeads) -> None:
    decoder = copy.deepcopy(heads.o2)
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


def test_e1_causality_full_disjoint_and_four_overlap_replay(heads: ObservationHeads) -> None:
    source = torch.randn(8, HIDDEN_DIM, generator=torch.Generator().manual_seed(31))
    query = torch.randn(1, QUERY_DIM, generator=torch.Generator().manual_seed(32))
    with torch.no_grad():
        full = _run_stream(heads.e1, source, range(8), query)
        first = _run_stream(heads.e1, source, range(6), query)
        overlap = _run_stream(heads.e1, source, range(2, 8), query, prior=first.next_states[0])
        disjoint_first = _run_stream(heads.e1, source, range(4), query)
        disjoint_second = _run_stream(
            heads.e1,
            source,
            range(4, 8),
            query,
            prior=disjoint_first.next_states[0],
        )
        # Causal prefix: mutating positions >= 4 cannot move logits at positions < 4.
        future_mutation = source.clone()
        future_mutation[4:] += 100.0
        mutated = _run_stream(heads.e1, future_mutation, range(8), query)

    torch.testing.assert_close(first.logits, full.logits[:, :6], atol=2.0e-6, rtol=2.0e-5)
    torch.testing.assert_close(overlap.logits, full.logits[:, 2:], atol=2.0e-6, rtol=2.0e-5)
    torch.testing.assert_close(
        disjoint_first.logits, full.logits[:, :4], atol=2.0e-6, rtol=2.0e-5
    )
    torch.testing.assert_close(
        disjoint_second.logits, full.logits[:, 4:], atol=2.0e-6, rtol=2.0e-5
    )
    torch.testing.assert_close(mutated.logits[:, :4], full.logits[:, :4])
    assert overlap.next_states[0].total_seen == 8
    assert overlap.next_states[0].position_ids.tolist() == list(range(8))


def test_e2_causality_full_disjoint_and_four_overlap_checkpoint_replay(
    heads: ObservationHeads,
) -> None:
    source = torch.randn(8, HIDDEN_DIM, generator=torch.Generator().manual_seed(37))
    query = torch.randn(1, QUERY_DIM, generator=torch.Generator().manual_seed(38))
    with torch.no_grad():
        full = _run_stream(heads.e2, source, range(8), query)
        first = _run_stream(heads.e2, source, range(6), query)
        overlap = _run_stream(heads.e2, source, range(2, 8), query, prior=first.next_states[0])
        disjoint_first = _run_stream(heads.e2, source, range(4), query)
        disjoint_second = _run_stream(
            heads.e2,
            source,
            range(4, 8),
            query,
            prior=disjoint_first.next_states[0],
        )
        # Causal prefix: mutating positions >= 4 cannot move logits at positions < 4.
        future_mutation = source.clone()
        future_mutation[4:] -= 100.0
        mutated = _run_stream(heads.e2, future_mutation, range(8), query)

    for field in ("event_logits", "phase_logits"):
        reference = getattr(full, field)
        torch.testing.assert_close(getattr(first, field), reference[:, :6])
        torch.testing.assert_close(getattr(overlap, field), reference[:, 2:])
        torch.testing.assert_close(getattr(disjoint_first, field), reference[:, :4])
        torch.testing.assert_close(getattr(disjoint_second, field), reference[:, 4:])
        torch.testing.assert_close(getattr(mutated, field)[:, :4], reference[:, :4])

    # Four-position overlap replays from the checkpoint before the overlap start, so the
    # final GRU hidden state matches the uninterrupted run bitwise.
    assert overlap.next_states[0].total_seen == 8
    assert overlap.next_states[0].position_ids.tolist() == [3, 4, 5, 6, 7]
    torch.testing.assert_close(
        overlap.next_states[0].hidden,
        overlap.next_states[0].checkpoint_hidden[-1],
    )
    torch.testing.assert_close(overlap.next_states[0].hidden, full.next_states[0].hidden)


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
