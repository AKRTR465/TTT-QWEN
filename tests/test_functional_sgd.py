from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from ttt_svcbench_qwen.config import InnerSGDConfig, load_config
from ttt_svcbench_qwen.fast_ttt import (
    FastWeightsState,
    OptimizerRuntimeState,
    build_fast_ttt_adapter,
)
from ttt_svcbench_qwen.functional_sgd import (
    FunctionalSGDResult,
    GradientMode,
    UpdateSkipReason,
    functional_sgd_step,
    functional_sgd_step_from_ttt_row,
    functional_sgd_steps_from_ttt,
    initialize_optimizer_state,
    reset_optimizer_state,
)
from ttt_svcbench_qwen.losses import (
    IdentityConsistencyAudit,
    LossSkipReason,
    LossTerm,
    TTTLossOutput,
)

MATRIX_SIZE = 768
MATRIX_SHAPE = (MATRIX_SIZE, MATRIX_SIZE)
MATRIX_ELEMENTS = MATRIX_SIZE * MATRIX_SIZE


@pytest.fixture(scope="module")
def optimizer_config() -> InnerSGDConfig:
    return load_config().fast_ttt.optimizer


def make_state(
    *,
    value: float = 0.0,
    dtype: torch.dtype = torch.float32,
    differentiable: bool = False,
) -> FastWeightsState:
    w0_1 = torch.full(MATRIX_SHAPE, value, dtype=dtype)
    w0_2 = torch.full(MATRIX_SHAPE, value, dtype=dtype)
    if differentiable:
        w0_1.requires_grad_(True)
        w0_2.requires_grad_(True)
        w_t_1 = w0_1.clone()
        w_t_2 = w0_2.clone()
    else:
        w_t_1 = w0_1.clone().requires_grad_(True)
        w_t_2 = w0_2.clone().requires_grad_(True)
    return FastWeightsState(
        w0_1=w0_1,
        w0_2=w0_2,
        w_t_1=w_t_1,
        w_t_2=w_t_2,
        fast_version=0,
        update_count=0,
        skip_count=0,
        differentiable=differentiable,
    )


def storage_pointer(tensor: Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def make_ttt_output(
    states: tuple[FastWeightsState, ...],
    valid: tuple[bool, ...],
) -> TTTLossOutput:
    row_losses = torch.stack([state.w_t_1.sum() + state.w_t_2.sum() for state in states]).float()
    valid_mask = torch.tensor(valid, dtype=torch.bool)
    counts = valid_mask.to(torch.int64)
    per_row = torch.where(valid_mask, row_losses, torch.zeros_like(row_losses))
    pred = LossTerm(
        value=per_row[valid_mask].mean() if any(valid) else per_row.sum() * 0.0,
        per_row=per_row,
        row_valid_mask=valid_mask,
        valid_counts=counts,
        mask_counts=counts.clone(),
        skip_reasons=tuple(
            None if is_valid else LossSkipReason.INSUFFICIENT_TIME for is_valid in valid
        ),
    )
    zero_rows = row_losses * 0.0
    zero_counts = torch.zeros_like(counts)

    def zero_term(reason: LossSkipReason) -> LossTerm:
        return LossTerm(
            value=zero_rows.sum() * 0.0,
            per_row=zero_rows,
            row_valid_mask=torch.zeros_like(valid_mask),
            valid_counts=zero_counts,
            mask_counts=zero_counts.clone(),
            skip_reasons=(reason,) * len(states),
        )

    identity = zero_term(LossSkipReason.NO_RELIABLE_MATCH)
    e1 = zero_term(LossSkipReason.NO_ALIGNED_EVENT)
    e2 = zero_term(LossSkipReason.NO_ALIGNED_EVENT)
    event = zero_term(LossSkipReason.NO_ALIGNED_EVENT)
    return TTTLossOutput(
        pred=pred,
        identity=identity,
        e1_event=e1,
        e2_event=e2,
        event=event,
        total=per_row[valid_mask].mean() if any(valid) else per_row.sum() * 0.0,
        per_row_total=per_row,
        update_valid_mask=valid_mask,
        identity_audit=IdentityConsistencyAudit(
            matched_counts=zero_counts,
            mismatch_counts=zero_counts.clone(),
            duplicate_counts=zero_counts.clone(),
            low_confidence_counts=zero_counts.clone(),
            invalid_source_counts=zero_counts.clone(),
            padding_counts=zero_counts.clone(),
        ),
    )


def step(
    state: FastWeightsState,
    loss: Tensor | None,
    config: InnerSGDConfig,
    optimizer: OptimizerRuntimeState | None = None,
    *,
    valid_term_count: int = 1,
    invalid_reason: UpdateSkipReason | None = None,
) -> FunctionalSGDResult:
    return functional_sgd_step(
        loss=loss,
        fast_state=state,
        optimizer_config=config,
        optimizer_state=optimizer or initialize_optimizer_state(config),
        valid_term_count=valid_term_count,
        invalid_reason=invalid_reason,
    )


def test_exact_two_matrix_step_matches_manual_sgd_without_mutating_old_state(
    optimizer_config: InnerSGDConfig,
) -> None:
    state = make_state()
    old_values = tuple(value.detach().clone() for value in state.fast_parameters)
    old_storage = tuple(storage_pointer(value) for value in state.fast_parameters)
    slow_sentinel = torch.tensor(3.0, requires_grad=True)
    loss = state.w_t_1.mean() + 2.0 * state.w_t_2.mean() + slow_sentinel.square()

    result = step(state, loss, optimizer_config)

    expected_1 = old_values[0] - optimizer_config.learning_rate / MATRIX_ELEMENTS
    expected_2 = old_values[1] - 2.0 * optimizer_config.learning_rate / MATRIX_ELEMENTS
    assert result.did_update is True
    assert result.gradient_mode is GradientMode.ONLINE_LEAF
    assert result.skip_reason is None
    assert result.fast_state.fast_version == result.fast_state.update_count == 1
    assert result.fast_state.skip_count == 0
    assert result.optimizer_state.attempted_update_count == 1
    assert result.optimizer_state.last_skip_reason is None
    assert torch.equal(result.fast_state.w_t_1, expected_1)
    assert torch.equal(result.fast_state.w_t_2, expected_2)
    assert result.fast_state.w_t_1.is_leaf and result.fast_state.w_t_1.requires_grad
    assert result.fast_state.w_t_2.is_leaf and result.fast_state.w_t_2.requires_grad
    assert slow_sentinel.grad is None
    assert tuple(storage_pointer(value) for value in state.fast_parameters) == old_storage
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(state.fast_parameters, old_values, strict=True)
    )
    assert all(
        storage_pointer(new) != storage_pointer(old)
        for new, old in zip(result.fast_state.fast_parameters, state.fast_parameters, strict=True)
    )


def test_joint_fp32_global_norm_clips_both_matrices_to_one(
    optimizer_config: InnerSGDConfig,
) -> None:
    state = make_state()
    loss = state.w_t_1.sum() + state.w_t_2.sum()

    result = step(state, loss, optimizer_config)

    expected_preclip = math.sqrt(2.0 * MATRIX_ELEMENTS)
    assert result.did_update
    assert result.gradient_norm == pytest.approx(expected_preclip, rel=1.0e-6)
    assert result.per_matrix_gradient_norms == pytest.approx(
        (float(MATRIX_SIZE), float(MATRIX_SIZE)), rel=1.0e-6
    )
    assert result.clipped_gradient_norm == pytest.approx(1.0, abs=2.0e-6)
    assert result.per_matrix_clipped_norms is not None
    assert math.sqrt(sum(value * value for value in result.per_matrix_clipped_norms)) == (
        pytest.approx(1.0, abs=2.0e-6)
    )
    assert result.update_norm > 0.0


@pytest.mark.parametrize(
    "reason",
    (UpdateSkipReason.NO_VALID_TERM, UpdateSkipReason.INSUFFICIENT_TIME),
)
def test_invalid_terms_skip_without_autograd_and_advance_audit_only(
    reason: UpdateSkipReason,
    optimizer_config: InnerSGDConfig,
) -> None:
    state = make_state(value=0.25)
    original_storage = tuple(storage_pointer(value) for value in state.fast_parameters)

    result = step(
        state,
        None,
        optimizer_config,
        valid_term_count=0,
        invalid_reason=reason,
    )

    assert result.did_update is False
    assert result.skip_reason is reason
    assert result.fast_state.fast_version == result.fast_state.update_count == 0
    assert result.fast_state.skip_count == 1
    assert result.optimizer_state.attempted_update_count == 1
    assert result.optimizer_state.last_skip_reason == reason.value
    assert result.gradient_norm is None
    assert result.update_norm == 0.0
    assert all(
        torch.equal(new, old)
        for new, old in zip(result.fast_state.fast_parameters, state.fast_parameters, strict=True)
    )
    assert all(
        storage_pointer(new) != pointer
        for new, pointer in zip(result.fast_state.fast_parameters, original_storage, strict=True)
    )


def test_nonfinite_loss_skips_before_gradient_computation(
    optimizer_config: InnerSGDConfig,
) -> None:
    state = make_state()
    loss = (state.w_t_1.sum() + state.w_t_2.sum()) * torch.tensor(float("nan"))

    result = step(state, loss, optimizer_config)

    assert result.skip_reason is UpdateSkipReason.NONFINITE_LOSS
    assert result.skip_detail == "loss_is_not_finite"
    assert result.gradient_norm is None
    assert state.w_t_1.grad is None and state.w_t_2.grad is None


class _FiniteForwardNaNGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, value: Tensor) -> Tensor:
        del ctx
        return value.sum() * 0.0 + 1.0

    @staticmethod
    def backward(ctx: object, output_gradient: Tensor) -> tuple[Tensor]:
        del ctx, output_gradient
        return (torch.full(MATRIX_SHAPE, torch.nan),)


def test_nonfinite_gradient_fault_injection_skips_without_touching_grad_fields(
    optimizer_config: InnerSGDConfig,
) -> None:
    state = make_state()
    loss = _FiniteForwardNaNGradient.apply(state.w_t_1) + state.w_t_2.sum() * 0.0

    result = step(state, loss, optimizer_config)

    assert result.skip_reason is UpdateSkipReason.NONFINITE_GRADIENT
    assert result.skip_detail == "one_or_more_fast_gradients_are_not_finite"
    assert state.w_t_1.grad is None and state.w_t_2.grad is None


def test_zero_gradient_and_bfloat16_unrepresentable_delta_are_explicit_skips(
    optimizer_config: InnerSGDConfig,
) -> None:
    zero_state = make_state()
    zero_loss = (zero_state.w_t_1.sum() + zero_state.w_t_2.sum()) * 0.0
    zero = step(zero_state, zero_loss, optimizer_config)

    bf16_state = make_state(value=1.0, dtype=torch.bfloat16)
    bf16_loss = bf16_state.w_t_1.sum() + bf16_state.w_t_2.sum()
    bf16 = step(bf16_state, bf16_loss, optimizer_config)

    assert zero.skip_reason is UpdateSkipReason.ZERO_GRADIENT
    assert zero.skip_detail == "clipped_gradient_has_zero_global_norm"
    assert bf16.skip_reason is UpdateSkipReason.UNREPRESENTABLE_UPDATE
    assert bf16.skip_detail == "update_not_representable_in_fast_dtype"
    assert torch.equal(bf16.fast_state.w_t_1, bf16_state.w_t_1)
    assert torch.equal(bf16.fast_state.w_t_2, bf16_state.w_t_2)


def test_meta_update_keeps_full_outer_gradient_to_w0(
    optimizer_config: InnerSGDConfig,
) -> None:
    state = make_state(value=1.0e-4, differentiable=True)
    support_loss = state.w_t_1.square().sum() + state.w_t_2.square().sum()

    result = step(state, support_loss, optimizer_config)
    query_loss = result.fast_state.w_t_1.sum() + result.fast_state.w_t_2.sum()
    outer_gradients = torch.autograd.grad(query_loss, (state.w0_1, state.w0_2))

    assert result.did_update
    assert result.gradient_mode is GradientMode.META_FULL_SECOND_ORDER
    assert result.differentiable_update is True
    assert result.fast_state.w_t_1.is_leaf is False
    assert result.fast_state.w_t_2.is_leaf is False
    expected = 1.0 - 2.0 * optimizer_config.learning_rate
    for gradient in outer_gradients:
        assert torch.isfinite(gradient).all()
        assert torch.allclose(gradient, torch.full_like(gradient, expected), atol=1.0e-6)
    assert state.w0_1.grad is None and state.w0_2.grad is None


def test_update_is_transactional_and_only_next_generation_changes_output(
    optimizer_config: InnerSGDConfig,
) -> None:
    state = make_state(value=0.25)
    current_output = (state.w_t_1[0, 0] + state.w_t_2[0, 0]).detach().clone()
    loss = state.w_t_1.sum() + state.w_t_2.sum()

    result = step(state, loss, optimizer_config)

    assert state.w_t_1[0, 0] + state.w_t_2[0, 0] == current_output
    next_output = result.fast_state.w_t_1[0, 0] + result.fast_state.w_t_2[0, 0]
    assert next_output != current_output
    assert torch.equal(state.w_t_1, torch.full_like(state.w_t_1, 0.25))
    assert torch.equal(state.w_t_2, torch.full_like(state.w_t_2, 0.25))


def test_attempted_update_accounting_and_optimizer_reset(
    optimizer_config: InnerSGDConfig,
) -> None:
    state = make_state()
    first = step(state, state.w_t_1.mean() + state.w_t_2.mean(), optimizer_config)
    second = step(
        first.fast_state,
        None,
        optimizer_config,
        first.optimizer_state,
        valid_term_count=0,
        invalid_reason=UpdateSkipReason.NO_VALID_TERM,
    )
    reset = reset_optimizer_state(optimizer_config)

    assert second.fast_state.update_count == 1
    assert second.fast_state.skip_count == 1
    assert second.optimizer_state.attempted_update_count == 2
    assert second.optimizer_state.last_skip_reason == UpdateSkipReason.NO_VALID_TERM.value
    assert reset.attempted_update_count == 0
    assert reset.last_skip_reason is None


def test_p5_fast_and_optimizer_reset_jointly_restore_w0_and_all_counters() -> None:
    project = load_config()
    adapter = build_fast_ttt_adapter(project)
    optimizer_config = project.fast_ttt.optimizer
    initial = adapter.initialize_fast_state()
    updated = step(
        initial,
        initial.w_t_1.sum() + initial.w_t_2.sum(),
        optimizer_config,
    )
    skipped = step(
        updated.fast_state,
        None,
        optimizer_config,
        updated.optimizer_state,
        valid_term_count=0,
        invalid_reason=UpdateSkipReason.NO_VALID_TERM,
    )
    pre_reset_storage = tuple(
        storage_pointer(value) for value in skipped.fast_state.fast_parameters
    )

    reset_fast = adapter.reset_fast_state(skipped.fast_state)
    reset_optimizer = reset_optimizer_state(optimizer_config)

    assert reset_fast.fast_version == 0
    assert reset_fast.update_count == 0
    assert reset_fast.skip_count == 0
    assert torch.equal(reset_fast.w_t_1, adapter.w0_1)
    assert torch.equal(reset_fast.w_t_2, adapter.w0_2)
    assert all(
        storage_pointer(value) not in pre_reset_storage for value in reset_fast.fast_parameters
    )
    assert reset_optimizer.attempted_update_count == 0
    assert reset_optimizer.last_skip_reason is None


def test_two_video_update_and_skip_are_storage_and_counter_isolated(
    optimizer_config: InnerSGDConfig,
) -> None:
    first = make_state(value=0.0)
    second = make_state(value=0.5)

    updated = step(
        first,
        first.w_t_1.sum() + first.w_t_2.sum(),
        optimizer_config,
    )
    skipped = step(
        second,
        None,
        optimizer_config,
        valid_term_count=0,
        invalid_reason=UpdateSkipReason.NO_VALID_TERM,
    )

    assert updated.fast_state.update_count == 1
    assert updated.fast_state.skip_count == 0
    assert skipped.fast_state.update_count == 0
    assert skipped.fast_state.skip_count == 1
    assert updated.optimizer_state.attempted_update_count == 1
    assert skipped.optimizer_state.attempted_update_count == 1
    all_fast = updated.fast_state.fast_parameters + skipped.fast_state.fast_parameters
    assert len({storage_pointer(value) for value in all_fast}) == 4
    assert not torch.equal(updated.fast_state.w_t_1, first.w_t_1)
    assert torch.equal(skipped.fast_state.w_t_1, second.w_t_1)


def test_typed_ttt_batch_bridge_derives_mixed_row_update_and_skip(
    optimizer_config: InnerSGDConfig,
) -> None:
    states = (make_state(value=0.0), make_state(value=0.5))
    ttt_output = make_ttt_output(states, (True, False))
    optimizers = tuple(initialize_optimizer_state(optimizer_config) for _ in states)

    results = functional_sgd_steps_from_ttt(
        ttt_output=ttt_output,
        fast_states=states,
        optimizer_config=optimizer_config,
        optimizer_states=optimizers,
    )

    updated, skipped = results
    assert updated.did_update is True
    assert updated.valid_term_count == 1
    assert updated.fast_state.update_count == 1
    assert updated.fast_state.skip_count == 0
    assert skipped.did_update is False
    assert skipped.skip_reason is UpdateSkipReason.INSUFFICIENT_TIME
    assert skipped.valid_term_count == 0
    assert skipped.fast_state.update_count == 0
    assert skipped.fast_state.skip_count == 1
    assert torch.equal(skipped.fast_state.w_t_1, states[1].w_t_1)
    all_fast = updated.fast_state.fast_parameters + skipped.fast_state.fast_parameters
    assert len({storage_pointer(value) for value in all_fast}) == 4


def test_typed_ttt_batch_bridge_retains_one_shared_graph_between_valid_rows(
    optimizer_config: InnerSGDConfig,
) -> None:
    states = (make_state(), make_state())
    ttt_output = make_ttt_output(states, (True, True))

    results = functional_sgd_steps_from_ttt(
        ttt_output=ttt_output,
        fast_states=states,
        optimizer_config=optimizer_config,
        optimizer_states=tuple(initialize_optimizer_state(optimizer_config) for _ in states),
    )

    assert all(result.did_update for result in results)
    assert all(result.valid_term_count == 1 for result in results)
    assert all(result.gradient_mode is GradientMode.ONLINE_LEAF for result in results)


def test_typed_ttt_bridge_rejects_row_and_batch_state_mismatches(
    optimizer_config: InnerSGDConfig,
) -> None:
    states = (make_state(), make_state(value=0.5))
    ttt_output = make_ttt_output(states, (True, False))
    optimizer = initialize_optimizer_state(optimizer_config)

    with pytest.raises(IndexError, match="in-range"):
        functional_sgd_step_from_ttt_row(
            ttt_output=ttt_output,
            row=2,
            fast_state=states[0],
            optimizer_config=optimizer_config,
            optimizer_state=optimizer,
        )
    with pytest.raises(ValueError, match="identical B"):
        functional_sgd_steps_from_ttt(
            ttt_output=ttt_output,
            fast_states=states[:1],
            optimizer_config=optimizer_config,
            optimizer_states=(optimizer,),
        )
    aliased = (states[0], states[0])
    with pytest.raises(ValueError, match="storage-isolated"):
        functional_sgd_steps_from_ttt(
            ttt_output=ttt_output,
            fast_states=aliased,
            optimizer_config=optimizer_config,
            optimizer_states=(optimizer, optimizer),
        )


def test_disconnected_loss_stale_grad_and_counter_drift_fail_closed(
    optimizer_config: InnerSGDConfig,
) -> None:
    disconnected = make_state()
    with pytest.raises(ValueError, match="connect to both"):
        step(disconnected, disconnected.w_t_1.sum(), optimizer_config)

    stale = make_state()
    stale.w_t_1.grad = torch.ones_like(stale.w_t_1)
    with pytest.raises(ValueError, match="stale"):
        step(stale, stale.w_t_1.sum() + stale.w_t_2.sum(), optimizer_config)

    drifted_runtime = replace(
        initialize_optimizer_state(optimizer_config),
        attempted_update_count=1,
    )
    fresh = make_state()
    with pytest.raises(ValueError, match="attempts"):
        step(
            fresh,
            fresh.w_t_1.sum() + fresh.w_t_2.sum(),
            optimizer_config,
            drifted_runtime,
        )


def test_non_scalar_loss_is_rejected_before_autograd(
    optimizer_config: InnerSGDConfig,
) -> None:
    state = make_state()
    vector_loss = state.w_t_1[0] + state.w_t_2[0]
    with pytest.raises(ValueError, match="scalar"):
        step(state, vector_loss, optimizer_config)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("learning_rate", 3.0e-4),
        ("momentum", 0.9),
        ("weight_decay", 0.1),
        ("steps_per_chunk", 2),
        ("grad_clip_norm", 2.0),
        ("reset_per_video", False),
        ("meta_gradient_mode", "first_order"),
    ),
)
def test_frozen_optimizer_config_drift_is_rejected(
    field: str,
    value: object,
    optimizer_config: InnerSGDConfig,
) -> None:
    drifted = optimizer_config.model_copy(update={field: value})
    with pytest.raises(ValueError, match=field):
        initialize_optimizer_state(drifted)


def test_result_rejects_nonfinite_audit_values(
    optimizer_config: InnerSGDConfig,
) -> None:
    state = make_state()
    skipped = step(
        state,
        None,
        optimizer_config,
        valid_term_count=0,
        invalid_reason=UpdateSkipReason.NO_VALID_TERM,
    )
    with pytest.raises(ValueError, match="finite"):
        FunctionalSGDResult(
            fast_state=skipped.fast_state,
            optimizer_state=skipped.optimizer_state,
            did_update=False,
            valid_term_count=0,
            gradient_norm=float("nan"),
            clipped_gradient_norm=None,
            per_matrix_gradient_norms=None,
            per_matrix_clipped_norms=None,
            update_norm=0.0,
            skip_reason=UpdateSkipReason.NO_VALID_TERM,
            skip_detail="fault",
            gradient_mode=GradientMode.ONLINE_LEAF,
        )
