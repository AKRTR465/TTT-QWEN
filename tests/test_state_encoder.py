from __future__ import annotations

import inspect
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.qwen_adapter import MergedVideoMetadata
from ttt_svcbench_qwen.state_encoder import (
    RestoredMergedGrid,
    SpatialEncoderAudit,
    SpatialEncoderOutput,
    SpatialObjectEncoder,
    SpatialSlotRuntimeState,
    StateEncoders,
    TemporalEventEncoder,
    build_spatial_encoder,
    build_state_encoders,
    restore_merged_grid,
    spatial_encoder_parameter_count,
)

ROOT = Path(__file__).resolve().parents[1]
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


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


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
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, MergedVideoMetadata, Tensor, Tensor, tuple[str, ...]]:
    metadata = make_metadata(merged_shapes)
    batch_size = len(merged_shapes)
    width = max(metadata.token_counts)
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(batch_size, width, 4096, generator=generator, dtype=dtype)
    positions = torch.arange(width).unsqueeze(0)
    counts = torch.tensor(metadata.token_counts, dtype=torch.int64).unsqueeze(1)
    visual_valid_mask = positions < counts
    max_t = max(shape[0] for shape in merged_shapes)
    tubelet_valid_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    for row, shape in enumerate(merged_shapes):
        tubelet_valid_mask[row, : shape[0]] = True
    q_target = torch.randn(batch_size, QUERY_DIM, generator=generator, dtype=dtype)
    video_ids = tuple(f"video-{index}" for index in range(batch_size))
    return (
        embeddings,
        visual_valid_mask,
        metadata,
        tubelet_valid_mask,
        q_target,
        video_ids,
    )


def run_encoder(
    encoder: SpatialObjectEncoder,
    merged_shapes: tuple[tuple[int, int, int], ...] = ((1, 1, 1),),
    *,
    seed: int = 0,
    prior_states: tuple[SpatialSlotRuntimeState, ...] | None = None,
    query_valid_mask: Tensor | None = None,
    required_slot_counts: Tensor | None = None,
    detach_runtime_state: bool = False,
) -> tuple[SpatialEncoderOutput, VisualInputs]:
    inputs = make_visual_inputs(merged_shapes, seed=seed)
    output = encoder(
        *inputs[:4],
        inputs[4],
        inputs[5],
        prior_states=prior_states,
        query_valid_mask=query_valid_mask,
        required_slot_counts=required_slot_counts,
        detach_runtime_state=detach_runtime_state,
    )
    return output, inputs


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
    assert spatial_encoder_parameter_count(encoder) == EXACT_PARAMETER_COUNT
    assert parameter_count(encoder.input_norm) == 2 * 4096 == 8_192
    assert parameter_count(encoder.input_projection) == 4096 * HIDDEN_DIM + HIDDEN_DIM
    assert parameter_count(encoder.query_projection) == QUERY_DIM * HIDDEN_DIM + HIDDEN_DIM
    assert encoder.shared_slot_seed.numel() == HIDDEN_DIM
    stages = (encoder.stage_1, encoder.stage_2)

    expected_stage_count = 10_632_960
    stage_parameter_ids: list[set[int]] = []
    for stage in stages:
        assert parameter_count(stage) == expected_stage_count
        assert parameter_count(stage.q_projection) == HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM
        assert parameter_count(stage.k_projection) == HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM
        assert parameter_count(stage.v_projection) == HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM
        assert parameter_count(stage.output_projection) == HIDDEN_DIM * HIDDEN_DIM + HIDDEN_DIM
        assert parameter_count(stage.gru) == 3_543_552
        assert parameter_count(stage.ffn_in) + parameter_count(stage.ffn_out) == 4_722_432
        assert (
            sum(
                parameter_count(norm)
                for norm in (stage.token_norm, stage.slot_norm, stage.ffn_norm)
            )
            == 4_608
        )
        stage_parameter_ids.append({id(parameter) for parameter in stage.parameters()})

    assert stages[0] is not stages[1]
    assert stage_parameter_ids[0].isdisjoint(stage_parameter_ids[1])
    assert stages[0].refinements == stages[1].refinements == 3
    assert stages[0].num_heads * stages[0].head_dim == HIDDEN_DIM == 12 * 64

    parameter_shapes = {tuple(parameter.shape) for parameter in encoder.parameters()}
    assert (ACTIVE_SLOTS, HIDDEN_DIM) not in parameter_shapes
    assert (config.spatial_encoder.max_active_slots, HIDDEN_DIM) not in parameter_shapes
    assert tuple(encoder.slot_codes.shape) == (
        config.spatial_encoder.max_active_slots,
        HIDDEN_DIM,
    )
    assert "slot_codes" in dict(encoder.named_buffers())
    assert not any("slot_codes" in key for key in encoder.state_dict())
    assert not any("confidence" in name for name, _ in encoder.named_parameters())


def test_demo_grid_restore_is_row_major_and_propagates_masks() -> None:
    metadata = make_metadata(((8, 7, 7),))
    token_ids = torch.arange(392, dtype=torch.float32).view(1, 392, 1)
    embeddings = token_ids.expand(1, 392, 4096)
    visual_valid_mask = torch.ones(1, 392, dtype=torch.bool)
    tubelet_valid_mask = torch.ones(1, 8, dtype=torch.bool)

    restored = restore_merged_grid(
        embeddings,
        visual_valid_mask,
        metadata,
        tubelet_valid_mask,
    )

    assert isinstance(restored, RestoredMergedGrid)
    assert restored.tokens.shape == (1, 8, 7, 7, 4096)
    assert restored.tokens[..., 0].shape == (1, 8, 7, 7)
    assert torch.equal(restored.tokens[..., 0], torch.arange(392).view(1, 8, 7, 7))
    assert restored.geometry_valid_mask.shape == (1, 8, 7, 7)
    assert restored.spatial_valid_mask.shape == (1, 8, 7, 7)
    assert restored.geometry_valid_mask.all()
    assert restored.spatial_valid_mask.all()
    assert restored.tubelet_valid_mask.all()
    assert restored.grid_shapes == ((8, 7, 7),)


def test_heterogeneous_grid_does_not_assume_49_and_invalid_tubelet_is_masked() -> None:
    inputs = make_visual_inputs(((2, 2, 3), (1, 1, 2)), seed=3)
    embeddings, visual_mask, metadata, tubelet_mask, _, _ = inputs
    tubelet_mask[0, 1] = False
    poisoned = embeddings.clone()
    poisoned[1, ~visual_mask[1]] = 1.0e6

    baseline = restore_merged_grid(embeddings, visual_mask, metadata, tubelet_mask)
    restored = restore_merged_grid(poisoned, visual_mask, metadata, tubelet_mask)

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
    with pytest.raises(ValueError, match="combine geometry and tubelet"):
        replace(restored, spatial_valid_mask=restored.geometry_valid_mask.clone())


def test_baseline_forward_shape_confidence_and_fixed_codes(encoder: SpatialObjectEncoder) -> None:
    output, _ = run_encoder(encoder)

    assert output.slots.shape == (1, ACTIVE_SLOTS, HIDDEN_DIM)
    assert output.slot_valid_mask.shape == (1, ACTIVE_SLOTS)
    assert output.slot_valid_mask.all()
    assert output.slot_confidence is not None
    assert output.slot_confidence.shape == (1, ACTIVE_SLOTS)
    assert bool(torch.isfinite(output.slots).all())
    assert bool(torch.isfinite(output.slot_confidence).all())
    assert bool(torch.all((output.slot_confidence >= 0) & (output.slot_confidence <= 1)))
    assert output.next_states is not None and len(output.next_states) == 1
    assert output.audit is not None
    assert not torch.equal(encoder.slot_codes[0], encoder.slot_codes[1])
    assert not torch.allclose(output.slots[:, :1], output.slots[:, 1:2])


def test_refinements_call_the_same_stage_parameters_three_times(
    encoder: SpatialObjectEncoder,
) -> None:
    calls = [0, 0]
    module_ids: list[list[int]] = [[], []]
    handles = []
    stages = (encoder.stage_1, encoder.stage_2)
    for index, stage in enumerate(stages):

        def record(
            module: nn.Module,
            _args: tuple[Any, ...],
            _output: Any,
            *,
            row: int = index,
        ) -> None:
            calls[row] += 1
            module_ids[row].append(id(module))

        handles.append(stage.q_projection.register_forward_hook(record))
    try:
        run_encoder(encoder, ((2, 1, 1),), seed=11)
    finally:
        for handle in handles:
            handle.remove()

    assert calls == [6, 6]
    assert module_ids == [
        [id(encoder.stage_1.q_projection)] * 6,
        [id(encoder.stage_2.q_projection)] * 6,
    ]


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
        expected_confidence = assignments.sum(dim=-1).mean(dim=1)
        expected_confidence = torch.where(slot_mask, expected_confidence, 0.0)
        weights = assignments / (assignments.sum(dim=-1, keepdim=True) + stage.attention_epsilon)
        updates = torch.einsum("bhks,bhsd->bhkd", weights, values)
        updates = updates.transpose(1, 2).reshape(1, ACTIVE_SLOTS, HIDDEN_DIM)
        updates = stage.output_projection(updates)
        updated = stage.gru(
            updates.reshape(ACTIVE_SLOTS, HIDDEN_DIM),
            expected.reshape(ACTIVE_SLOTS, HIDDEN_DIM),
        ).reshape(1, ACTIVE_SLOTS, HIDDEN_DIM)
        updated = updated + stage.ffn_out(F.silu(stage.ffn_in(stage.ffn_norm(updated))))
        expected = torch.where(slot_mask.unsqueeze(-1), updated, expected)

    assert torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)
    assert torch.allclose(
        actual_confidence,
        expected_confidence,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert torch.equal(actual[:, -1], slots[:, -1])
    assert actual_confidence[:, -1].item() == 0.0


def test_query_conditions_slots_but_invalid_query_ignores_padding_values(
    encoder: SpatialObjectEncoder,
) -> None:
    embeddings, visual_mask, metadata, tubelet_mask, _, video_ids = make_visual_inputs(
        ((1, 1, 1),),
        seed=17,
    )
    q_zero = torch.zeros(1, QUERY_DIM)
    q_changed = torch.linspace(-2.0, 2.0, QUERY_DIM).unsqueeze(0)
    q_padding_poisoned = torch.full((1, QUERY_DIM), torch.nan)

    valid_zero = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_zero,
        video_ids,
        query_valid_mask=torch.ones(1, dtype=torch.bool),
    )
    valid_changed = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_changed,
        video_ids,
        query_valid_mask=torch.ones(1, dtype=torch.bool),
    )
    invalid_zero = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_zero,
        video_ids,
        query_valid_mask=torch.zeros(1, dtype=torch.bool),
    )
    invalid_changed = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_padding_poisoned,
        video_ids,
        query_valid_mask=torch.zeros(1, dtype=torch.bool),
    )

    assert not torch.allclose(valid_zero.slots, valid_changed.slots)
    assert torch.equal(invalid_zero.slots, invalid_changed.slots)
    assert torch.equal(invalid_zero.slot_confidence, invalid_changed.slot_confidence)


def test_invalid_tubelet_carries_state_without_nan(encoder: SpatialObjectEncoder) -> None:
    generator = torch.Generator().manual_seed(19)
    first_token = torch.randn(1, 1, 4096, generator=generator)
    invalid_token = torch.full((1, 1, 4096), 1.0e6)
    two_tokens = torch.cat((first_token, invalid_token), dim=1)
    q_target = torch.randn(1, QUERY_DIM, generator=generator)
    video_ids = ("video-carry",)

    one = encoder(
        first_token,
        torch.ones(1, 1, dtype=torch.bool),
        make_metadata(((1, 1, 1),)),
        torch.ones(1, 1, dtype=torch.bool),
        q_target,
        video_ids,
    )
    two = encoder(
        two_tokens,
        torch.ones(1, 2, dtype=torch.bool),
        make_metadata(((2, 1, 1),)),
        torch.tensor([[True, False]]),
        q_target,
        video_ids,
    )

    assert torch.allclose(one.slots, two.slots, rtol=1.0e-6, atol=2.0e-6)
    assert torch.allclose(
        one.slot_confidence,
        two.slot_confidence,
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert bool(torch.isfinite(two.slots).all())
    assert two.audit is not None
    assert two.audit.valid_tubelet_counts == (1,)


def test_all_invalid_tubelets_return_safe_initialized_state(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((2, 1, 1),), seed=23)
    embeddings, visual_mask, metadata, _, q_target, video_ids = inputs
    initialized = encoder.reset_slot_state(video_ids[0], q_target[0])
    output = encoder(
        embeddings,
        visual_mask,
        metadata,
        torch.zeros(1, 2, dtype=torch.bool),
        q_target,
        video_ids,
        prior_states=(initialized,),
    )

    assert torch.equal(output.slots[0], initialized.slots)
    assert torch.equal(output.slot_confidence[0], initialized.slot_confidence)
    assert bool(torch.isfinite(output.slots).all())
    assert output.audit is not None
    assert output.audit.valid_tubelet_counts == (0,)
    with pytest.raises(ValueError, match="fresh spatial runtime"):
        encoder(
            embeddings,
            visual_mask,
            metadata,
            torch.zeros(1, 2, dtype=torch.bool),
            q_target,
            video_ids,
        )


def test_recurrent_full_sequence_matches_incremental_handoff(
    encoder: SpatialObjectEncoder,
) -> None:
    generator = torch.Generator().manual_seed(29)
    embeddings = torch.randn(1, 2, 4096, generator=generator)
    q_target = torch.randn(1, QUERY_DIM, generator=generator)
    video_ids = ("video-recurrent",)
    full = encoder(
        embeddings,
        torch.ones(1, 2, dtype=torch.bool),
        make_metadata(((2, 1, 1),)),
        torch.ones(1, 2, dtype=torch.bool),
        q_target,
        video_ids,
    )
    first = encoder(
        embeddings[:, :1],
        torch.ones(1, 1, dtype=torch.bool),
        make_metadata(((1, 1, 1),)),
        torch.ones(1, 1, dtype=torch.bool),
        q_target,
        video_ids,
    )
    assert first.next_states is not None
    prior = first.next_states[0]
    prior_values = prior.slots.detach().clone()
    prior_pointer = storage_pointer(prior.slots)
    second = encoder(
        embeddings[:, 1:],
        torch.ones(1, 1, dtype=torch.bool),
        make_metadata(((1, 1, 1),)),
        torch.ones(1, 1, dtype=torch.bool),
        q_target,
        video_ids,
        prior_states=(prior,),
    )

    assert torch.allclose(full.slots, second.slots, rtol=1.0e-5, atol=1.0e-6)
    assert torch.allclose(
        full.slot_confidence,
        second.slot_confidence,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    assert torch.equal(prior.slots, prior_values)
    assert second.next_states is not None
    assert storage_pointer(second.next_states[0].slots) != prior_pointer
    assert second.next_states[0].processed_tubelets == 2


def test_reset_reproduces_first_step_and_runtime_detach_is_explicit(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((1, 1, 1),), seed=31)
    embeddings, visual_mask, metadata, tubelet_mask, q_target, video_ids = inputs
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
    attached = encoder(
        differentiable_embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        differentiable_q,
        video_ids,
        detach_runtime_state=False,
    )
    detached = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_target,
        video_ids,
        detach_runtime_state=True,
    )

    assert attached.next_states is not None and detached.next_states is not None
    assert attached.next_states[0].slots.grad_fn is not None
    assert attached.next_states[0].differentiable is True
    assert detached.next_states[0].slots.grad_fn is None
    assert detached.next_states[0].slots.requires_grad is False
    assert detached.next_states[0].differentiable is False
    attached.slots.square().mean().backward()
    assert differentiable_embeddings.grad is not None
    assert differentiable_q.grad is not None
    assert bool(torch.isfinite(differentiable_embeddings.grad).all())
    assert bool(torch.isfinite(differentiable_q.grad).all())


def test_batch_matches_independent_rows_and_runtime_storage_is_isolated(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((1, 1, 1), (1, 1, 1)), seed=37)
    embeddings, visual_mask, metadata, tubelet_mask, q_target, video_ids = inputs
    batched = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_target,
        video_ids,
    )
    rows = []
    for row in range(2):
        rows.append(
            encoder(
                embeddings[row : row + 1],
                visual_mask[row : row + 1],
                make_metadata(((1, 1, 1),)),
                tubelet_mask[row : row + 1],
                q_target[row : row + 1],
                (video_ids[row],),
            )
        )

    assert torch.allclose(batched.slots, torch.cat([row.slots for row in rows]), atol=1.0e-6)
    assert batched.next_states is not None
    assert batched.next_states[0].video_id == video_ids[0]
    assert batched.next_states[1].video_id == video_ids[1]
    assert storage_pointer(batched.next_states[0].slots) != storage_pointer(
        batched.next_states[1].slots
    )

    perturbed = embeddings.clone()
    perturbed[0, :, ::2] += 100.0
    changed = encoder(
        perturbed,
        visual_mask,
        metadata,
        tubelet_mask,
        q_target,
        video_ids,
    )
    assert torch.equal(batched.slots[1], changed.slots[1])
    assert not torch.allclose(batched.slots[0], changed.slots[0])


def test_overflow_is_explicit_audited_and_never_expands_or_changes_slots(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((1, 1, 1),), seed=41)
    embeddings, visual_mask, metadata, tubelet_mask, q_target, video_ids = inputs
    capacity = torch.tensor([ACTIVE_SLOTS], dtype=torch.int64)
    above_configured_max = torch.tensor([65], dtype=torch.int64)
    baseline = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_target,
        video_ids,
        required_slot_counts=capacity,
    )
    overflow = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_target,
        video_ids,
        required_slot_counts=above_configured_max,
    )

    assert torch.equal(baseline.slots, overflow.slots)
    assert torch.equal(baseline.slot_confidence, overflow.slot_confidence)
    assert overflow.slots.shape[1] == ACTIVE_SLOTS
    assert overflow.active_slot_overflow_count.tolist() == [33]
    assert overflow.next_states is not None and overflow.audit is not None
    assert overflow.next_states[0].active_slot_overflow_count == 33
    assert overflow.next_states[0].overflow_event_count == 1
    assert overflow.audit.excess_slot_counts == (33,)
    assert overflow.audit.overflow_policy == "preserve_existing_reject_excess"
    assert overflow.audit.overflow_events == (True,)
    assert overflow.audit.required_slot_counts == (65,)

    prior = overflow.next_states[0]
    no_new_excess = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_target,
        video_ids,
        prior_states=(prior,),
        required_slot_counts=capacity,
    )
    more_excess = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_target,
        video_ids,
        prior_states=(prior,),
        required_slot_counts=torch.tensor([34]),
    )
    assert torch.equal(no_new_excess.slots, more_excess.slots)
    assert more_excess.next_states is not None
    assert more_excess.next_states[0].active_slot_overflow_count == 35
    assert more_excess.next_states[0].overflow_event_count == 2
    assert more_excess.active_slot_overflow_count.tolist() == [35]


def test_slot_mask_confidence_padding_gradients_dtype_and_parameter_stability(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((1, 1, 2), (1, 1, 1)), seed=43)
    embeddings, visual_mask, metadata, tubelet_mask, q_target, video_ids = inputs
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

    output = encoder(
        embeddings,
        visual_mask,
        metadata,
        tubelet_mask,
        q_target,
        video_ids,
        prior_states=(masked_state, second_state),
    )

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


@pytest.mark.parametrize(
    "invalid_required",
    [
        torch.tensor([-1], dtype=torch.int64),
        torch.tensor([True]),
        torch.tensor([32.0]),
        torch.tensor([[32]], dtype=torch.int64),
    ],
)
def test_required_slot_counts_reject_invalid_values_but_allow_above_max(
    encoder: SpatialObjectEncoder,
    invalid_required: Tensor,
) -> None:
    inputs = make_visual_inputs(((1, 1, 1),), seed=47)
    with pytest.raises((TypeError, ValueError), match="required_slot_counts"):
        encoder(*inputs, required_slot_counts=invalid_required)

    legal = encoder(*inputs, required_slot_counts=torch.tensor([100], dtype=torch.int64))
    assert legal.slots.shape == (1, ACTIVE_SLOTS, HIDDEN_DIM)
    assert legal.active_slot_overflow_count.tolist() == [68]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("embeddings", torch.zeros(1, 4096)),
        ("embeddings", torch.zeros(1, 1, 4095)),
        ("embeddings", torch.zeros(1, 1, 4096, dtype=torch.int64)),
        ("embeddings", torch.full((1, 1, 4096), torch.nan)),
        ("embeddings", torch.full((1, 1, 4096), torch.inf)),
        ("visual_mask", torch.ones(1, 2, dtype=torch.bool)),
        ("visual_mask", torch.ones(1, 1)),
        ("tubelet_mask", torch.ones(1, 2, dtype=torch.bool)),
        ("tubelet_mask", torch.ones(1, 1)),
        ("q_target", torch.zeros(1, QUERY_DIM - 1)),
        ("q_target", torch.zeros(1, QUERY_DIM, dtype=torch.int64)),
        ("q_target", torch.full((1, QUERY_DIM), torch.nan)),
        ("q_target", torch.full((1, QUERY_DIM), torch.inf)),
    ],
)
def test_forward_rejects_bad_shapes_dtypes_nonfinite_and_masks(
    encoder: SpatialObjectEncoder,
    field: str,
    replacement: Tensor,
) -> None:
    inputs = list(make_visual_inputs(((1, 1, 1),), seed=53))
    positions = {
        "embeddings": 0,
        "visual_mask": 1,
        "tubelet_mask": 3,
        "q_target": 4,
    }
    inputs[positions[field]] = replacement
    with pytest.raises((TypeError, ValueError)):
        encoder(*inputs)


def test_query_video_and_runtime_flags_fail_closed(encoder: SpatialObjectEncoder) -> None:
    inputs = make_visual_inputs(((1, 1, 1),), seed=57)
    with pytest.raises(ValueError, match="query_valid_mask"):
        encoder(*inputs, query_valid_mask=torch.ones(1))
    with pytest.raises(ValueError, match="query_valid_mask"):
        encoder(*inputs, query_valid_mask=torch.ones(1, 1, dtype=torch.bool))
    with pytest.raises(TypeError, match="detach_runtime_state"):
        encoder(*inputs, detach_runtime_state=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="prior_states"):
        encoder(*inputs, prior_states=(object(),))  # type: ignore[arg-type]

    duplicate_inputs = list(make_visual_inputs(((1, 1, 1), (1, 1, 1)), seed=58))
    duplicate_inputs[5] = ("same-video", "same-video")
    with pytest.raises(ValueError, match="duplicate video_ids"):
        encoder(*duplicate_inputs)
    duplicate_inputs[5] = ("video-a", "")
    with pytest.raises(ValueError, match="non-empty video_id"):
        encoder(*duplicate_inputs)


def test_spatial_output_runtime_and_audit_types_reject_invalid_values(
    encoder: SpatialObjectEncoder,
) -> None:
    slots = torch.zeros(1, ACTIVE_SLOTS, HIDDEN_DIM)
    slot_mask = torch.ones(1, ACTIVE_SLOTS, dtype=torch.bool)
    overflow = torch.zeros(1, dtype=torch.int64)
    with pytest.raises(ValueError, match="finite"):
        SpatialEncoderOutput(
            slots=torch.full_like(slots, torch.nan),
            slot_valid_mask=slot_mask,
            active_slot_overflow_count=overflow,
        )
    with pytest.raises(ValueError, match="non-negative"):
        SpatialEncoderOutput(
            slots=slots,
            slot_valid_mask=slot_mask,
            active_slot_overflow_count=torch.tensor([-1]),
        )

    state = encoder.reset_slot_state("video-contract", torch.zeros(QUERY_DIM))
    with pytest.raises(ValueError, match="finite"):
        replace(state, slots=torch.full_like(state.slots, torch.inf))
    with pytest.raises(TypeError, match="exact integers"):
        replace(state, processed_tubelets=True)
    with pytest.raises(ValueError, match="non-negative"):
        replace(state, active_slot_overflow_count=-1)

    audit = SpatialEncoderAudit(
        grid_shapes=((1, 1, 1),),
        visual_token_counts=(1,),
        valid_tubelet_counts=(1,),
        required_slot_counts=(ACTIVE_SLOTS,),
        excess_slot_counts=(0,),
        overflow_events=(False,),
        overflow_policy="preserve_existing_reject_excess",
        stage_refinement_calls=(3, 3),
    )
    with pytest.raises(TypeError, match="overflow flags"):
        replace(audit, overflow_events=(1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        replace(audit, excess_slot_counts=(-1,))


@pytest.mark.parametrize(
    ("slot_count", "slot_dim"),
    [(ACTIVE_SLOTS, HIDDEN_DIM - 1), (65, HIDDEN_DIM)],
)
def test_public_output_and_runtime_reject_wrong_dim_or_more_than_64_slots(
    encoder: SpatialObjectEncoder,
    slot_count: int,
    slot_dim: int,
) -> None:
    slots = torch.zeros(1, slot_count, slot_dim)
    slot_mask = torch.ones(1, slot_count, dtype=torch.bool)
    confidence = torch.zeros(1, slot_count)
    with pytest.raises(ValueError, match=r"\[B, K_a, 768\].*64"):
        SpatialEncoderOutput(
            slots=slots,
            slot_valid_mask=slot_mask,
            active_slot_overflow_count=torch.zeros(1, dtype=torch.int64),
            slot_confidence=confidence,
        )

    state = encoder.reset_slot_state("video-public-contract", torch.zeros(QUERY_DIM))
    invalid_runtime_slots = torch.zeros(slot_count, slot_dim)
    with pytest.raises(ValueError, match=r"\[K_a, 768\].*64"):
        replace(
            state,
            slots=invalid_runtime_slots,
            slot_valid_mask=torch.ones(slot_count, dtype=torch.bool),
            slot_confidence=torch.zeros(slot_count),
        )


def test_float16_invalid_slot_forward_backward_has_only_finite_gradients() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(73)
    encoder = build_spatial_encoder(load_config()).to(device=device, dtype=torch.float16).eval()
    embeddings = torch.randn(
        1,
        1,
        4096,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    q_target = torch.randn(
        1,
        QUERY_DIM,
        device=device,
        dtype=torch.float16,
        requires_grad=True,
    )
    visual_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    tubelet_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    slot_mask = torch.ones(ACTIVE_SLOTS, dtype=torch.bool, device=device)
    slot_mask[-1] = False
    prior = encoder.reset_slot_state(
        "video-fp16",
        q_target[0],
        slot_valid_mask=slot_mask,
        differentiable=True,
    )

    try:
        output = encoder(
            embeddings,
            visual_mask,
            make_metadata(((1, 1, 1),)),
            tubelet_mask,
            q_target,
            ("video-fp16",),
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


def test_metadata_count_grid_and_mask_mismatches_fail_explicitly(
    encoder: SpatialObjectEncoder,
) -> None:
    with pytest.raises(ValueError, match="token_counts"):
        MergedVideoMetadata(
            video_grid_thw=torch.tensor([[1, 2, 2]]),
            merged_grid_thw=torch.tensor([[1, 1, 1]]),
            spatial_merge_size=2,
            token_counts=(2,),
            token_offsets=(0, 2),
        )
    with pytest.raises(ValueError, match="divide only H/W"):
        MergedVideoMetadata(
            video_grid_thw=torch.tensor([[1, 4, 4]]),
            merged_grid_thw=torch.tensor([[1, 1, 2]]),
            spatial_merge_size=2,
            token_counts=(2,),
            token_offsets=(0, 2),
        )

    inputs = list(make_visual_inputs(((1, 1, 1),), seed=59))
    inputs[0] = torch.cat((inputs[0], inputs[0]), dim=0)
    inputs[1] = torch.cat((inputs[1], inputs[1]), dim=0)
    inputs[3] = torch.cat((inputs[3], inputs[3]), dim=0)
    inputs[4] = torch.cat((inputs[4], inputs[4]), dim=0)
    inputs[5] = ("video-a", "video-b")
    with pytest.raises(ValueError, match="metadata|batch"):
        encoder(*inputs)

    inputs = list(make_visual_inputs(((1, 1, 1),), seed=61))
    inputs[1] = torch.zeros_like(inputs[1])
    with pytest.raises(ValueError, match="visual_valid_mask|token count"):
        encoder(*inputs)


def test_runtime_rejects_stale_shape_dtype_alias_and_video_order(
    encoder: SpatialObjectEncoder,
) -> None:
    inputs = make_visual_inputs(((1, 1, 1), (1, 1, 1)), seed=67)
    embeddings, visual_mask, metadata, tubelet_mask, q_target, video_ids = inputs
    first = encoder.reset_slot_state(video_ids[0], q_target[0])
    second = encoder.reset_slot_state(video_ids[1], q_target[1])

    with pytest.raises((TypeError, ValueError), match="one prior|batch"):
        encoder(*inputs, prior_states=(first,))
    with pytest.raises(ValueError, match="video_id|order"):
        encoder(*inputs, prior_states=(second, first))

    shared = replace(first, video_id=video_ids[1])
    with pytest.raises(ValueError, match="share|storage|distinct"):
        encoder(*inputs, prior_states=(first, shared))

    stale_dtype = replace(
        first,
        slots=first.slots.double(),
        slot_confidence=first.slot_confidence.double(),
    )
    with pytest.raises(ValueError, match="dtype|device"):
        encoder(
            embeddings[:1],
            visual_mask[:1],
            make_metadata(((1, 1, 1),)),
            tubelet_mask[:1],
            q_target[:1],
            (video_ids[0],),
            prior_states=(stale_dtype,),
        )

    stale_shape = replace(
        first,
        slots=first.slots[:-1],
        slot_valid_mask=first.slot_valid_mask[:-1],
        slot_confidence=first.slot_confidence[:-1],
    )
    with pytest.raises(ValueError, match="slot shape"):
        encoder(
            embeddings[:1],
            visual_mask[:1],
            make_metadata(((1, 1, 1),)),
            tubelet_mask[:1],
            q_target[:1],
            (video_ids[0],),
            prior_states=(stale_shape,),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mask_device_mismatch_fails_on_cuda() -> None:
    embeddings, visual_mask, metadata, tubelet_mask, q_target, video_ids = make_visual_inputs(
        ((1, 1, 1),),
        seed=71,
    )
    encoder = build_spatial_encoder(load_config()).cuda().eval()
    with pytest.raises(ValueError, match="device"):
        encoder(
            embeddings.cuda(),
            visual_mask,
            metadata,
            tubelet_mask.cuda(),
            q_target.cuda(),
            video_ids,
        )


def test_spatial_and_joint_builders_require_config_and_p7_builder_is_implemented() -> None:
    with pytest.raises((TypeError, ValueError), match="config|ProjectConfig"):
        build_spatial_encoder()  # type: ignore[call-arg]
    with pytest.raises((TypeError, ValueError), match="config|ProjectConfig"):
        build_state_encoders()  # type: ignore[call-arg]
    with torch.device("meta"):
        encoders = build_state_encoders(load_config())
    assert isinstance(encoders, StateEncoders)
    assert isinstance(encoders.spatial, SpatialObjectEncoder)
    assert isinstance(encoders.temporal, TemporalEventEncoder)

    source = (ROOT / "src" / "ttt_svcbench_qwen" / "state_encoder.py").read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "from ttt_svcbench_qwen.state_bank" not in source
    assert "from ttt_svcbench_qwen.observation_heads" not in source
    assert "optimizer" not in inspect.signature(SpatialObjectEncoder.forward).parameters
