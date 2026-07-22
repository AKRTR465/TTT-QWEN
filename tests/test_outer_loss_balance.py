from __future__ import annotations

import inspect

import pytest
import torch
from torch import Tensor

from ttt_svcbench_qwen.config import OfficialWeakBalanceConfig
from ttt_svcbench_qwen.losses import AnswerLossOutput, LossSkipReason, LossTerm
from ttt_svcbench_qwen.outer_loss_balance import (
    OfficialWeakGradientAnchors,
    OfficialWeakOuterLossComposer,
)
from ttt_svcbench_qwen.stage_a_targets import (
    OfficialWeakLossAudit,
    OfficialWeakLossTerm,
    OfficialWeakStateLossOutput,
)


def _answer(value: Tensor, *, valid: bool = True) -> AnswerLossOutput:
    scalar = value.float()
    per_row = scalar.reshape(1) if valid else (scalar * 0.0).reshape(1)
    row_valid = torch.tensor([valid], dtype=torch.bool, device=scalar.device)
    counts = row_valid.to(torch.int64)
    term = LossTerm(
        value=per_row.mean() if valid else per_row.sum() * 0.0,
        per_row=per_row,
        row_valid_mask=row_valid,
        valid_counts=counts,
        mask_counts=torch.ones_like(counts),
        skip_reasons=(None if valid else LossSkipReason.NO_ANSWER_TOKEN,),
    )
    return AnswerLossOutput(term, term, term, term, term)


def _weak(value: Tensor, *, valid: bool = True) -> OfficialWeakLossTerm:
    return OfficialWeakLossTerm(
        value=value.float() if valid else value.float() * 0.0,
        valid_rows=int(valid),
    )


def _state(
    task: Tensor,
    operator: Tensor,
    retrieval: Tensor,
    time: Tensor,
    *,
    valid: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> OfficialWeakStateLossOutput:
    terms = tuple(
        _weak(value, valid=is_valid)
        for value, is_valid in zip((task, operator, retrieval, time), valid, strict=True)
    )
    total = torch.stack(tuple(term.value for term in terms)).sum()
    return OfficialWeakStateLossOutput(
        task=terms[0],
        operator=terms[1],
        retrieval=terms[2],
        time=terms[3],
        total=total,
        audit=OfficialWeakLossAudit(
            labels_joined_after_forward=True,
            runtime_payload_reused_for_labels=False,
            identity_target_fabricated=False,
            unique_retrieval_id_fabricated=False,
            future_occurrences_ignored=0,
            retrieval_bag_sizes=(1,),
        ),
    )


def _gradient_connected_state(
    anchors: OfficialWeakGradientAnchors,
    factors: tuple[float, float, float, float],
    *,
    value: float = 1.0,
    valid: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> OfficialWeakStateLossOutput:
    source = (
        anchors.q_target,
        anchors.q_operator,
        anchors.q_target,
        anchors.q_time,
    )
    values = tuple(
        factor * anchor.sum() + (value - factor)
        for factor, anchor in zip(factors, source, strict=True)
    )
    return _state(*values, valid=valid)


def _anchors() -> OfficialWeakGradientAnchors:
    return OfficialWeakGradientAnchors(
        q_target=torch.ones((1, 1), requires_grad=True),
        q_operator=torch.ones((1, 1), requires_grad=True),
        q_time=torch.ones((1, 1), requires_grad=True),
    )


def test_config_rejects_inverted_scale_bounds() -> None:
    with pytest.raises(ValueError, match="scale_min"):
        OfficialWeakBalanceConfig(
            group_weight=0.3,
            scale_min=10.0,
            scale_max=0.1,
            epsilon=1.0e-8,
        )


def test_ema_answer_reference_persists_and_missing_term_keeps_history() -> None:
    config = OfficialWeakBalanceConfig(
        group_weight=0.3,
        scale_min=0.001,
        scale_max=20.0,
        epsilon=1.0e-8,
        ema_beta=0.5,
        grad_ema_beta=0.5,
    )
    composer = OfficialWeakOuterLossComposer(config)
    first_anchors = _anchors()
    first = composer.compose(
        (_answer(torch.tensor(4.0, requires_grad=True)),),
        (
            _gradient_connected_state(
                first_anchors,
                (2.0, 2.0, 2.0, 2.0),
                value=2.0,
            ),
        ),
        gradient_anchors=(first_anchors,),
    )
    assert first.audit is not None
    assert tuple(term.scale for term in first.audit.terms) == pytest.approx((1.0,) * 4)
    assert first.audit.ema_means == pytest.approx((4.0, 2.0, 2.0, 2.0, 2.0))
    assert first.audit.ema_update_counts == (1, 1, 1, 1, 1)
    assert first.audit.gradient_ema_rms == pytest.approx((2.0, 2.0, 2.0, 2.0))
    assert all(
        anchor.grad is None
        for anchor in (
            first_anchors.q_target,
            first_anchors.q_operator,
            first_anchors.q_time,
        )
    )

    second_anchors = _anchors()
    second = composer.compose(
        (_answer(torch.tensor(2.0, requires_grad=True)),),
        (
            _gradient_connected_state(
                second_anchors,
                (8.0, 8.0, 8.0, 8.0),
                value=8.0,
                valid=(True, True, False, True),
            ),
        ),
        gradient_anchors=(second_anchors,),
    )
    assert second.audit is not None
    assert second.audit.ema_means == pytest.approx((3.0, 5.0, 5.0, 2.0, 5.0))
    assert second.audit.ema_update_counts == (2, 2, 2, 1, 2)
    assert second.audit.gradient_ema_rms == pytest.approx((9.0, 9.0, 2.0, 9.0))
    assert second.audit.gradient_ema_update_counts == (2, 2, 1, 2)
    assert second.audit.terms[2].global_valid_count == 0
    assert torch.isnan(second.audit.terms[2].scale)
    assert torch.isnan(second.audit.terms[2].raw_gradient_rms)
    assert dict(second.audit.metrics())["loss/scale/retrieval"] is None
    assert second.audit.terms[0].loss_scale == pytest.approx(2.0)

    restored = OfficialWeakOuterLossComposer(config)
    restored.load_state_dict(composer.state_dict(), strict=True)
    assert torch.equal(restored.ema_values, composer.ema_values)
    assert torch.equal(restored.ema_update_counts, composer.ema_update_counts)
    assert torch.equal(restored.gradient_ema_values, composer.gradient_ema_values)
    assert int(restored.balance_schema_version.item()) == 7


def test_ema_buffers_remain_float64_across_parent_dtype_conversions() -> None:
    composer = OfficialWeakOuterLossComposer(
        OfficialWeakBalanceConfig(
            group_weight=0.3,
            scale_min=0.001,
            scale_max=20.0,
            epsilon=1.0e-8,
        )
    )
    composer.ema_values.copy_(torch.linspace(0.123456789, 0.523456789, 5))
    composer.gradient_ema_values.copy_(torch.linspace(0.234567891, 0.534567891, 4))
    expected_loss = composer.ema_values.clone()
    expected_gradient = composer.gradient_ema_values.clone()
    owner = torch.nn.Module()
    owner.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    owner.composer = composer

    owner.to(dtype=torch.bfloat16)
    assert owner.weight.dtype == torch.bfloat16
    assert composer.ema_values.dtype == torch.float64
    assert composer.gradient_ema_values.dtype == torch.float64
    assert torch.equal(composer.ema_values, expected_loss)
    assert torch.equal(composer.gradient_ema_values, expected_gradient)
    if torch.cuda.is_available():
        owner.cuda()
        assert composer.ema_values.device.type == "cuda"
        assert composer.ema_values.dtype == torch.float64
        owner.cpu()
        assert torch.equal(composer.ema_values, expected_loss)
    owner.to(dtype=torch.float32)

    assert owner.weight.dtype == torch.float32
    assert composer.ema_values.dtype == torch.float64
    assert composer.gradient_ema_values.dtype == torch.float64
    assert torch.equal(composer.ema_values, expected_loss)
    assert torch.equal(composer.gradient_ema_values, expected_gradient)


def test_float64_ema_updates_below_bfloat16_resolution_do_not_plateau() -> None:
    composer = OfficialWeakOuterLossComposer(
        OfficialWeakBalanceConfig(
            group_weight=0.3,
            scale_min=0.001,
            scale_max=20.0,
            epsilon=1.0e-8,
            ema_beta=0.99,
        )
    ).to(dtype=torch.bfloat16)
    valid = torch.ones(5, dtype=torch.bool)
    composer._update_ema(torch.ones(5, dtype=torch.float64), valid)
    first = composer.ema_values.clone()
    composer._update_ema(torch.full((5,), 1.001, dtype=torch.float64), valid)

    assert composer.ema_values.dtype == torch.float64
    assert torch.all(composer.ema_values > first)
    assert torch.all(composer.ema_values < 1.001)


def test_checkpoint_load_rejects_quantized_or_stale_balance_state() -> None:
    composer = OfficialWeakOuterLossComposer(
        OfficialWeakBalanceConfig(
            group_weight=0.3,
            scale_min=0.001,
            scale_max=20.0,
            epsilon=1.0e-8,
        )
    )
    quantized = composer.state_dict()
    quantized["ema_values"] = quantized["ema_values"].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="expected torch.float64"):
        composer.load_state_dict(quantized, strict=True)

    stale = composer.state_dict()
    stale["balance_schema_version"] = torch.tensor(6, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="expected 7"):
        composer.load_state_dict(stale, strict=True)

    wrong_shape = composer.state_dict()
    wrong_shape["gradient_ema_values"] = torch.zeros(5, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="expected torch.float64.*found"):
        composer.load_state_dict(wrong_shape, strict=True)


def test_group_guard_uses_previous_answer_ema_and_reports_unclamped_current_ratio() -> None:
    composer = OfficialWeakOuterLossComposer(
        OfficialWeakBalanceConfig(
            group_weight=0.4,
            answer_reference_floor=0.1,
            scale_min=0.001,
            scale_max=20.0,
            epsilon=1.0e-8,
        )
    )
    first_anchors = _anchors()
    composer.compose(
        (_answer(torch.tensor(1.0, requires_grad=True)),),
        (_gradient_connected_state(first_anchors, (1.0, 1.0, 1.0, 1.0)),),
        gradient_anchors=(first_anchors,),
    )
    second_anchors = _anchors()
    second = composer.compose(
        (_answer(torch.tensor(1.0e-6, requires_grad=True)),),
        (_gradient_connected_state(second_anchors, (1.0, 1.0, 1.0, 1.0)),),
        gradient_anchors=(second_anchors,),
    )

    assert second.audit is not None
    assert second.audit.group_guard_reference == pytest.approx(1.0)
    assert not bool(second.audit.group_guard_reference_floored)
    assert second.audit.group_guard == pytest.approx(1.0)
    assert second.audit.state_global_mean == pytest.approx(0.4)
    assert float(second.audit.state_to_reference_ratio) <= 0.4 + 1.0e-6
    assert second.audit.state_to_current_answer_ratio > 100_000
    assert second.audit.auxiliary_to_answer_ratio > 100_000


def test_group_guard_reference_applies_configured_floor_on_cold_start() -> None:
    composer = OfficialWeakOuterLossComposer(
        OfficialWeakBalanceConfig(
            group_weight=0.4,
            answer_reference_floor=0.1,
            scale_min=0.001,
            scale_max=20.0,
            epsilon=1.0e-8,
        )
    )
    anchors = _anchors()
    output = composer.compose(
        (_answer(torch.tensor(0.01, requires_grad=True)),),
        (_gradient_connected_state(anchors, (1.0, 1.0, 1.0, 1.0)),),
        gradient_anchors=(anchors,),
    )

    assert output.audit is not None
    assert output.audit.group_guard_reference == pytest.approx(0.1)
    assert bool(output.audit.group_guard_reference_floored)
    assert output.audit.state_to_reference_ratio <= 0.4 + 1.0e-6


def test_gradient_ema_uses_previous_step_and_balances_activation_rms() -> None:
    composer = OfficialWeakOuterLossComposer(
        OfficialWeakBalanceConfig(
            group_weight=0.3,
            scale_min=0.001,
            scale_max=20.0,
            epsilon=1.0e-8,
            ema_beta=0.99,
            grad_ema_beta=0.99,
            grad_scale_min=0.1,
            grad_scale_max=10.0,
        )
    )
    factors = (1.0, 10.0, 100.0, 1000.0)
    first_anchors = _anchors()
    first = composer.compose(
        (_answer(torch.tensor(1.0, requires_grad=True)),),
        (_gradient_connected_state(first_anchors, factors),),
        gradient_anchors=(first_anchors,),
    )
    assert first.audit is not None
    assert tuple(term.scale for term in first.audit.terms) == pytest.approx((1.0,) * 4)

    second_anchors = _anchors()
    second = composer.compose(
        (_answer(torch.tensor(1.0, requires_grad=True)),),
        (_gradient_connected_state(second_anchors, factors),),
        gradient_anchors=(second_anchors,),
    )
    assert second.audit is not None
    assert tuple(term.gradient_scale for term in second.audit.terms) == pytest.approx(
        (10.0, 10.0**0.5, 10.0**-0.5, 0.1),
        rel=1.0e-5,
    )
    assert tuple(term.scale for term in second.audit.terms) == pytest.approx(
        (10.0, 10.0**0.5, 10.0**-0.5, 0.1),
        rel=1.0e-5,
    )
    assert all(
        anchor.grad is None
        for anchor in (
            second_anchors.q_target,
            second_anchors.q_operator,
            second_anchors.q_time,
        )
    )
    second.mean_total.backward()
    assert all(
        anchor.grad is not None
        for anchor in (
            second_anchors.q_target,
            second_anchors.q_operator,
            second_anchors.q_time,
        )
    )


def test_formal_collectives_have_fixed_loss_then_streamed_gradient_lengths() -> None:
    calls: list[int] = []

    def reduce(values: Tensor) -> Tensor:
        calls.append(values.numel())
        return values * 4.0

    composer = OfficialWeakOuterLossComposer(
        OfficialWeakBalanceConfig(
            group_weight=0.3,
            scale_min=0.001,
            scale_max=20.0,
            epsilon=1.0e-8,
        ),
        reduce_sum=reduce,
        world_size=4,
    )
    calibration_anchors = _anchors()
    state = _gradient_connected_state(
        calibration_anchors,
        (1.0, 2.0, 3.0, 4.0),
        valid=(True, True, False, True),
    )
    calibrated = composer.calibrate(
        (_answer(torch.tensor(1.0, requires_grad=True)),),
        (state,),
    )
    assert calibrated.audit is not None
    statistics = composer.measure_streamed_gradients(
        state,
        calibration_anchors,
        calibrated.audit,
    )
    committed = composer.commit_streamed_gradients((statistics,), calibrated.audit)

    assert calls == [18, 8]
    assert tuple(term.global_valid_count for term in committed.terms) == (4, 4, 0, 4)
    assert torch.isnan(committed.terms[2].raw_gradient_rms)
    assert dict(committed.metrics())["grad_balance/raw_rms/retrieval"] is None


def test_old_ema_checkpoint_cannot_silently_load_into_schema_six() -> None:
    composer = OfficialWeakOuterLossComposer(
        OfficialWeakBalanceConfig(
            group_weight=0.3,
            scale_min=0.001,
            scale_max=20.0,
            epsilon=1.0e-8,
        )
    )
    old_state = {
        "ema_values": torch.zeros(5, dtype=torch.float64),
        "ema_valid": torch.zeros(5, dtype=torch.bool),
        "ema_update_counts": torch.zeros(5, dtype=torch.int64),
    }

    with pytest.raises(RuntimeError, match="Missing required balance-state key"):
        composer.load_state_dict(old_state, strict=True)


def test_compose_hot_path_has_no_host_item_control_flow() -> None:
    source = inspect.getsource(OfficialWeakOuterLossComposer.compose)

    assert ".item(" not in source
