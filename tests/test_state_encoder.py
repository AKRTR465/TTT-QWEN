from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from tests.support import parameter_count
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.qwen_adapter import MergedVideoMetadata
from ttt_svcbench_qwen.state_encoder import (
    RestoredMergedGrid,
    SpatialEncoderOutput,
    SpatialObjectEncoder,
    build_spatial_encoder,
    restore_merged_grid,
)

HIDDEN_DIM = 768
QUERY_DIM = 512
ACTIVE_SLOTS = 32
EXACT_PARAMETER_COUNT = 24_815_360
VisualInputs = tuple[
    Tensor,
    Tensor,
    MergedVideoMetadata,
    Tensor,
    Tensor,
    tuple[str, ...],
]


def storage_pointer(tensor: Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def make_metadata(
    merged_shapes: tuple[tuple[int, int, int], ...],
) -> MergedVideoMetadata:
    merged = torch.tensor(merged_shapes, dtype=torch.int64)
    raw = merged.clone()
    raw[:, 1:] *= 2
    counts = tuple(int(value) for value in torch.prod(merged, dim=1).tolist())
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return MergedVideoMetadata(
        video_grid_thw=raw,
        merged_grid_thw=merged,
        spatial_merge_size=2,
        token_counts=counts,
        token_offsets=tuple(offsets),
    )


def make_visual_inputs(
    merged_shapes: tuple[tuple[int, int, int], ...],
    *,
    seed: int = 0,
) -> VisualInputs:
    metadata = make_metadata(merged_shapes)
    batch_size = len(merged_shapes)
    width = max(metadata.token_counts)
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(batch_size, width, 4096, generator=generator)
    counts = torch.tensor(metadata.token_counts, dtype=torch.int64).unsqueeze(1)
    visual_valid_mask = torch.arange(width).unsqueeze(0) < counts
    max_t = max(shape[0] for shape in merged_shapes)
    tubelet_valid_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    for row, shape in enumerate(merged_shapes):
        tubelet_valid_mask[row, : shape[0]] = True
    q_target = torch.randn(batch_size, QUERY_DIM, generator=generator)
    video_ids = tuple(f"video-{index}" for index in range(batch_size))
    return (
        embeddings,
        visual_valid_mask,
        metadata,
        tubelet_valid_mask,
        q_target,
        video_ids,
    )


def call(
    encoder: SpatialObjectEncoder,
    inputs: VisualInputs,
    **kwargs: Any,
) -> SpatialEncoderOutput:
    """Positional forward call so tests only spell out the keyword under test."""

    return encoder(*inputs, **kwargs)


@pytest.fixture(scope="module")
def encoder() -> SpatialObjectEncoder:
    torch.manual_seed(20260714)
    module = build_spatial_encoder(load_config())
    module.eval()
    return module


def test_meta_structure_and_parameter_budget_are_exact() -> None:
    config = load_config()
    with torch.device("meta"):
        encoder = build_spatial_encoder(config)

    assert all(parameter.device.type == "meta" for parameter in encoder.parameters())
    assert parameter_count(encoder) == EXACT_PARAMETER_COUNT
    assert parameter_count(encoder.input_norm) == 2 * 4096 == 8_192
    assert parameter_count(encoder.input_projection) == 4096 * HIDDEN_DIM + HIDDEN_DIM
    assert parameter_count(encoder.query_projection) == QUERY_DIM * HIDDEN_DIM + HIDDEN_DIM
    assert encoder.shared_slot_seed.numel() == HIDDEN_DIM

    stages = (encoder.stage_1, encoder.stage_2)
    stage_parameter_ids: list[set[int]] = []
    for stage in stages:
        assert parameter_count(stage) == 10_632_960
        assert parameter_count(stage.gru) == 3_543_552
        assert parameter_count(stage.ffn_in) + parameter_count(stage.ffn_out) == 4_722_432
        stage_parameter_ids.append({id(parameter) for parameter in stage.parameters()})

    assert stages[0] is not stages[1]
    assert stage_parameter_ids[0].isdisjoint(stage_parameter_ids[1])
    assert stages[0].refinements == stages[1].refinements == 3
    assert stages[0].num_heads * stages[0].head_dim == HIDDEN_DIM == 12 * 64

    # Slots are runtime state, never parameters; codes are a non-persistent buffer.
    parameter_shapes = {tuple(parameter.shape) for parameter in encoder.parameters()}
    assert (ACTIVE_SLOTS, HIDDEN_DIM) not in parameter_shapes
    assert (config.spatial_encoder.max_active_slots, HIDDEN_DIM) not in parameter_shapes
    assert tuple(encoder.slot_codes.shape) == (
        config.spatial_encoder.max_active_slots,
        HIDDEN_DIM,
    )
    assert "slot_codes" in dict(encoder.named_buffers())
    assert not any("slot_codes" in key for key in encoder.state_dict())


def test_grid_restore_is_row_major_and_masks_padding_and_invalid_tubelets() -> None:
    token_ids = torch.arange(392, dtype=torch.float32).view(1, 392, 1)
    demo = restore_merged_grid(
        token_ids.expand(1, 392, 4096),
        torch.ones(1, 392, dtype=torch.bool),
        make_metadata(((8, 7, 7),)),
        torch.ones(1, 8, dtype=torch.bool),
    )

    assert isinstance(demo, RestoredMergedGrid)
    assert demo.tokens.shape == (1, 8, 7, 7, 4096)
    assert torch.equal(demo.tokens[..., 0], torch.arange(392).view(1, 8, 7, 7))
    assert demo.grid_shapes == ((8, 7, 7),)
    assert demo.geometry_valid_mask.shape == (1, 8, 7, 7)
    assert demo.spatial_valid_mask.shape == (1, 8, 7, 7)
    assert demo.geometry_valid_mask.all() and demo.spatial_valid_mask.all()
    assert demo.tubelet_valid_mask.all()

    embeddings, visual_mask, metadata, tubelet_mask, _, _ = make_visual_inputs(
        ((2, 2, 3), (1, 1, 2)),
        seed=3,
    )
    tubelet_mask[0, 1] = False
    poisoned = embeddings.clone()
    poisoned[1, ~visual_mask[1]] = 1.0e6

    baseline = restore_merged_grid(embeddings, visual_mask, metadata, tubelet_mask)
    restored = restore_merged_grid(poisoned, visual_mask, metadata, tubelet_mask)

    # Heterogeneous grids pad to the batch maximum; 49 spatial tokens is never assumed.
    assert restored.tokens.shape == (2, 2, 2, 3, 4096)
    assert restored.grid_shapes == ((2, 2, 3), (1, 1, 2))
    assert restored.geometry_valid_mask[0].sum().item() == 12
    assert restored.geometry_valid_mask[1].sum().item() == 2
    assert restored.spatial_valid_mask[0].sum().item() == 6
    assert restored.spatial_valid_mask[1].sum().item() == 2
    assert restored.geometry_valid_mask[0, 1].all()
    assert not restored.spatial_valid_mask[0, 1].any()
    assert torch.equal(
        baseline.tokens[1][baseline.spatial_valid_mask[1]],
        restored.tokens[1][restored.spatial_valid_mask[1]],
    )


def test_baseline_forward_shape_confidence_and_fixed_codes(encoder: SpatialObjectEncoder) -> None:
    output = call(encoder, make_visual_inputs(((1, 1, 1),)))

    assert output.slots.shape == (1, ACTIVE_SLOTS, HIDDEN_DIM)
    assert output.slot_valid_mask.shape == (1, ACTIVE_SLOTS)
    assert output.slot_valid_mask.all()
    assert output.slot_confidence is not None
    assert output.slot_confidence.shape == (1, ACTIVE_SLOTS)
    assert bool(torch.isfinite(output.slots).all())
    assert bool(torch.isfinite(output.slot_confidence).all())
    assert bool(torch.all((output.slot_confidence >= 0) & (output.slot_confidence <= 1)))
    assert output.next_states is not None and len(output.next_states) == 1
    assert not torch.equal(encoder.slot_codes[0], encoder.slot_codes[1])
    assert not torch.allclose(output.slots[:, :1], output.slots[:, 1:2])


def test_slot_stage_matches_frozen_qkvo_gru_ffn_formula(
    encoder: SpatialObjectEncoder,
) -> None:
    stage = encoder.stage_1
    generator = torch.Generator().manual_seed(13)
    tokens = torch.randn(1, 2, HIDDEN_DIM, generator=generator)
    q_target = torch.randn(1, QUERY_DIM, generator=generator)
    query_condition = encoder.query_projection(q_target)
    state = encoder.reset_slot_state("video-formula", q_target[0], differentiable=True)
    slots = state.slots.unsqueeze(0)
    token_mask = torch.tensor([[True, False]])
    slot_mask = torch.ones(1, ACTIVE_SLOTS, dtype=torch.bool)
    slot_mask[:, -1] = False

    actual, actual_confidence = stage(
        tokens,
        slots,
        query_condition,
        token_mask,
        slot_mask,
    )

    normalized = stage.token_norm(tokens)
    keys = stage.k_projection(normalized).reshape(1, 2, 12, 64).transpose(1, 2)
    values = stage.v_projection(normalized).reshape(1, 2, 12, 64).transpose(1, 2)
    valid_token_count = token_mask.sum(dim=1).clamp_min(1).to(tokens.dtype)
    expected = slots
    expected_confidence = torch.zeros(1, ACTIVE_SLOTS)
    for _ in range(3):
        conditioned = stage.slot_norm(expected) + query_condition.unsqueeze(1)
        queries = stage.q_projection(conditioned).reshape(1, ACTIVE_SLOTS, 12, 64).transpose(1, 2)
        logits = torch.einsum("bhkd,bhsd->bhks", queries, keys) / math.sqrt(64)
        logits = logits.masked_fill(
            ~slot_mask[:, None, :, None],
            torch.finfo(logits.dtype).min,
        )
        assignments = torch.softmax(logits, dim=2)
        valid_pairs = slot_mask[:, None, :, None] & token_mask[:, None, None, :]
        assignments = torch.where(valid_pairs, assignments, 0.0)
        expected_confidence = assignments.sum(dim=-1) / valid_token_count[:, None, None]
        expected_confidence = torch.where(slot_mask, expected_confidence.mean(dim=1), 0.0)
        weights = assignments / (assignments.sum(dim=-1, keepdim=True) + stage.attention_epsilon)
        updates = torch.einsum("bhks,bhsd->bhkd", weights, values)
        updates = stage.output_projection(
            updates.transpose(1, 2).reshape(1, ACTIVE_SLOTS, HIDDEN_DIM)
        )
        updated = stage.gru(
            updates.reshape(ACTIVE_SLOTS, HIDDEN_DIM),
            expected.reshape(ACTIVE_SLOTS, HIDDEN_DIM),
        ).reshape(1, ACTIVE_SLOTS, HIDDEN_DIM)
        updated = updated + stage.ffn_out(F.silu(stage.ffn_in(stage.ffn_norm(updated))))
        expected = torch.where(slot_mask.unsqueeze(-1), updated, expected)

    assert torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)
    assert torch.allclose(actual_confidence, expected_confidence, rtol=1.0e-6, atol=1.0e-7)
    assert torch.equal(actual[:, -1], slots[:, -1])
    assert actual_confidence[:, -1].item() == 0.0


def test_query_conditions_slots_but_invalid_query_ignores_padding_values(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((1, 1, 1),), seed=17)

    def forward(q_target: Tensor, *, query_valid: bool) -> SpatialEncoderOutput:
        return call(
            encoder,
            (*inputs[:4], q_target, inputs[5]),
            query_valid_mask=torch.full((1,), query_valid, dtype=torch.bool),
        )

    q_zero = torch.zeros(1, QUERY_DIM)
    q_changed = torch.linspace(-2.0, 2.0, QUERY_DIM).unsqueeze(0)
    q_poisoned = torch.full((1, QUERY_DIM), torch.nan)

    valid_zero = forward(q_zero, query_valid=True)
    valid_changed = forward(q_changed, query_valid=True)
    invalid_zero = forward(q_zero, query_valid=False)
    invalid_poisoned = forward(q_poisoned, query_valid=False)

    assert not torch.allclose(valid_zero.slots, valid_changed.slots)
    assert torch.equal(invalid_zero.slots, invalid_poisoned.slots)
    assert torch.equal(invalid_zero.slot_confidence, invalid_poisoned.slot_confidence)


def test_invalid_tubelets_are_masked_and_carry_prior_state(
    encoder: SpatialObjectEncoder,
) -> None:
    generator = torch.Generator().manual_seed(19)
    first_token = torch.randn(1, 1, 4096, generator=generator)
    two_tokens = torch.cat((first_token, torch.full((1, 1, 4096), 1.0e6)), dim=1)
    q_target = torch.randn(1, QUERY_DIM, generator=generator)
    video_ids = ("video-carry",)
    ones_1 = torch.ones(1, 1, dtype=torch.bool)
    ones_2 = torch.ones(1, 2, dtype=torch.bool)
    meta_1 = make_metadata(((1, 1, 1),))
    meta_2 = make_metadata(((2, 1, 1),))
    second_masked = torch.tensor([[True, False]])

    one = call(encoder, (first_token, ones_1, meta_1, ones_1, q_target, video_ids))
    two = call(encoder, (two_tokens, ones_2, meta_2, second_masked, q_target, video_ids))

    # A masked tubelet contributes nothing, even with poisoned token values.
    assert torch.allclose(one.slots, two.slots, rtol=1.0e-6, atol=2.0e-6)
    assert torch.allclose(one.slot_confidence, two.slot_confidence, rtol=1.0e-6, atol=1.0e-7)
    assert bool(torch.isfinite(two.slots).all())
    assert one.next_states is not None and two.next_states is not None
    assert two.next_states[0].processed_tubelets == 1

    # An all-masked chunk returns the prior state unchanged and advances nothing.
    prior = one.next_states[0]
    none_valid = torch.zeros(1, 2, dtype=torch.bool)
    all_invalid = call(
        encoder,
        (two_tokens, ones_2, meta_2, none_valid, q_target, video_ids),
        prior_states=(prior,),
    )
    assert torch.equal(all_invalid.slots[0], prior.slots)
    assert torch.equal(all_invalid.slot_confidence[0], prior.slot_confidence)
    assert bool(torch.isfinite(all_invalid.slots).all())
    assert all_invalid.next_states is not None
    assert all_invalid.next_states[0].processed_tubelets == 1


def test_recurrent_full_sequence_matches_incremental_handoff(
    encoder: SpatialObjectEncoder,
) -> None:
    generator = torch.Generator().manual_seed(29)
    embeddings = torch.randn(1, 2, 4096, generator=generator)
    q_target = torch.randn(1, QUERY_DIM, generator=generator)
    video_ids = ("video-recurrent",)
    ones_1 = torch.ones(1, 1, dtype=torch.bool)
    ones_2 = torch.ones(1, 2, dtype=torch.bool)
    tail = (make_metadata(((1, 1, 1),)), ones_1, q_target, video_ids)

    full = call(
        encoder,
        (embeddings, ones_2, make_metadata(((2, 1, 1),)), ones_2, q_target, video_ids),
    )
    first = call(encoder, (embeddings[:, :1], ones_1, *tail))
    assert first.next_states is not None
    prior = first.next_states[0]
    prior_values = prior.slots.detach().clone()
    prior_pointer = storage_pointer(prior.slots)
    second = call(encoder, (embeddings[:, 1:], ones_1, *tail), prior_states=(prior,))

    assert torch.allclose(full.slots, second.slots, rtol=1.0e-5, atol=1.0e-6)
    assert torch.allclose(full.slot_confidence, second.slot_confidence, rtol=1.0e-5, atol=1.0e-6)
    assert torch.equal(prior.slots, prior_values)
    assert second.next_states is not None
    assert storage_pointer(second.next_states[0].slots) != prior_pointer
    assert second.next_states[0].processed_tubelets == 2


def test_reset_reproduces_first_step_and_runtime_detach_is_explicit(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((1, 1, 1),), seed=31)
    embeddings, _, _, _, q_target, video_ids = inputs
    fresh = encoder.reset_slot_state(video_ids[0], q_target[0])
    reset = encoder.reset_slot_state(video_ids[0], q_target[0])
    another_fresh = encoder.reset_slot_state(video_ids[0], q_target[0])

    assert torch.equal(reset.slots, another_fresh.slots)
    assert torch.equal(fresh.slots, reset.slots)
    assert reset.active_slot_overflow_count == 0
    assert reset.overflow_event_count == 0
    assert reset.processed_tubelets == 0
    assert storage_pointer(reset.slots) != storage_pointer(another_fresh.slots)

    differentiable_embeddings = embeddings.clone().requires_grad_(True)
    differentiable_q = q_target.clone().requires_grad_(True)
    attached = call(
        encoder,
        (differentiable_embeddings, *inputs[1:4], differentiable_q, inputs[5]),
        detach_runtime_state=False,
    )
    detached = call(encoder, inputs, detach_runtime_state=True)

    assert attached.next_states is not None and detached.next_states is not None
    assert attached.next_states[0].slots.grad_fn is not None
    assert attached.next_states[0].differentiable is True
    assert detached.next_states[0].slots.grad_fn is None
    assert detached.next_states[0].slots.requires_grad is False
    assert detached.next_states[0].differentiable is False
    attached.slots.square().mean().backward()
    for gradient in (differentiable_embeddings.grad, differentiable_q.grad):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())


def test_batch_matches_independent_rows_and_runtime_storage_is_isolated(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((1, 1, 1), (1, 1, 1)), seed=37)
    embeddings, visual_mask, _, tubelet_mask, q_target, video_ids = inputs
    batched = call(encoder, inputs)
    rows = [
        call(
            encoder,
            (
                embeddings[row : row + 1],
                visual_mask[row : row + 1],
                make_metadata(((1, 1, 1),)),
                tubelet_mask[row : row + 1],
                q_target[row : row + 1],
                (video_ids[row],),
            ),
        )
        for row in range(2)
    ]

    assert torch.allclose(batched.slots, torch.cat([row.slots for row in rows]), atol=1.0e-6)
    assert batched.next_states is not None
    assert batched.next_states[0].video_id == video_ids[0]
    assert batched.next_states[1].video_id == video_ids[1]
    assert storage_pointer(batched.next_states[0].slots) != storage_pointer(
        batched.next_states[1].slots
    )

    perturbed = embeddings.clone()
    perturbed[0, :, ::2] += 100.0
    changed = call(encoder, (perturbed, *inputs[1:]))
    assert torch.equal(batched.slots[1], changed.slots[1])
    assert not torch.allclose(batched.slots[0], changed.slots[0])


def test_overflow_is_explicit_and_never_expands_or_changes_slots(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((1, 1, 1),), seed=41)
    capacity = torch.tensor([ACTIVE_SLOTS], dtype=torch.int64)
    baseline = call(encoder, inputs, required_slot_counts=capacity)
    overflow = call(
        encoder,
        inputs,
        required_slot_counts=torch.tensor([65], dtype=torch.int64),
    )

    # Overflow is reported, never satisfied: the slot tensor is untouched.
    assert torch.equal(baseline.slots, overflow.slots)
    assert torch.equal(baseline.slot_confidence, overflow.slot_confidence)
    assert baseline.active_slot_overflow_count.tolist() == [0]
    assert overflow.slots.shape[1] == ACTIVE_SLOTS
    assert overflow.active_slot_overflow_count.tolist() == [33]
    assert overflow.next_states is not None
    assert overflow.next_states[0].active_slot_overflow_count == 33
    assert overflow.next_states[0].overflow_event_count == 1

    prior = overflow.next_states[0]
    no_new_excess = call(
        encoder,
        inputs,
        prior_states=(prior,),
        required_slot_counts=capacity,
    )
    more_excess = call(
        encoder,
        inputs,
        prior_states=(prior,),
        required_slot_counts=torch.tensor([34]),
    )
    assert torch.equal(no_new_excess.slots, more_excess.slots)
    assert more_excess.next_states is not None
    assert more_excess.next_states[0].active_slot_overflow_count == 35
    assert more_excess.next_states[0].overflow_event_count == 2
    assert more_excess.active_slot_overflow_count.tolist() == [35]

    # A required count above the configured maximum is accounted, not rejected.
    legal = call(
        encoder,
        make_visual_inputs(((1, 1, 1),), seed=47),
        required_slot_counts=torch.tensor([100], dtype=torch.int64),
    )
    assert legal.slots.shape == (1, ACTIVE_SLOTS, HIDDEN_DIM)
    assert legal.active_slot_overflow_count.tolist() == [68]


def test_slot_mask_confidence_padding_gradients_dtype_and_parameter_stability(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((1, 1, 2), (1, 1, 1)), seed=43)
    embeddings, visual_mask, _, _, q_target, video_ids = inputs
    embeddings.requires_grad_(True)
    q_target.requires_grad_(True)
    state = encoder.reset_slot_state(video_ids[0], q_target[0])
    slot_mask = state.slot_valid_mask.clone()
    slot_mask[-2:] = False
    masked_state = replace(
        state,
        slot_valid_mask=slot_mask,
        slot_confidence=torch.zeros_like(state.slot_confidence),
    )
    second_state = encoder.reset_slot_state(video_ids[1], q_target[1])
    parameter_ids = tuple(id(parameter) for parameter in encoder.parameters())
    state_dict_keys = tuple(encoder.state_dict())

    output = call(encoder, inputs, prior_states=(masked_state, second_state))

    assert output.slots.dtype == embeddings.dtype
    assert output.slots.device == embeddings.device
    assert output.slot_confidence.dtype == embeddings.dtype
    assert output.slot_confidence.device == embeddings.device
    assert torch.equal(output.slot_valid_mask[0], slot_mask)
    assert torch.equal(output.slot_confidence[0, -2:], torch.zeros(2))
    assert bool(torch.isfinite(output.slots).all())
    assert bool(torch.isfinite(output.slot_confidence).all())
    output.slots.square().mean().backward()
    assert embeddings.grad is not None and q_target.grad is not None
    assert embeddings.grad[0, visual_mask[0]].abs().sum() > 0
    assert embeddings.grad[1, ~visual_mask[1]].abs().sum() == 0
    assert q_target.grad.abs().sum() > 0
    assert tuple(id(parameter) for parameter in encoder.parameters()) == parameter_ids
    assert tuple(encoder.state_dict()) == state_dict_keys


def test_float16_invalid_slot_forward_backward_has_only_finite_gradients() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(73)
    encoder = build_spatial_encoder(load_config()).to(device=device, dtype=torch.float16).eval()
    embeddings = torch.randn(1, 1, 4096, device=device, dtype=torch.float16, requires_grad=True)
    q_target = torch.randn(1, QUERY_DIM, device=device, dtype=torch.float16, requires_grad=True)
    ones = torch.ones(1, 1, dtype=torch.bool, device=device)
    slot_mask = torch.ones(ACTIVE_SLOTS, dtype=torch.bool, device=device)
    slot_mask[-1] = False
    prior = encoder.reset_slot_state(
        "video-fp16",
        q_target[0],
        slot_valid_mask=slot_mask,
        differentiable=True,
    )

    try:
        output = call(
            encoder,
            (embeddings, ones, make_metadata(((1, 1, 1),)), ones, q_target, ("video-fp16",)),
            prior_states=(prior,),
            detach_runtime_state=False,
        )
        output.slots.float().square().mean().backward()
    except RuntimeError as error:
        unsupported_half = device.type == "cpu" and (
            "not implemented for 'Half'" in str(error) or "not implemented for Half" in str(error)
        )
        if unsupported_half:
            pytest.skip(f"CPU float16 kernel unavailable: {error}")
        raise

    assert bool(torch.isfinite(output.slots).all())
    assert output.slot_confidence is not None
    assert bool(torch.isfinite(output.slot_confidence).all())
    for gradient in (embeddings.grad, q_target.grad):
        assert gradient is not None
        assert bool(torch.isfinite(gradient).all())
    for stage in (encoder.stage_1, encoder.stage_2):
        for parameter in stage.parameters():
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
