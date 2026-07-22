from __future__ import annotations

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
    assert second.audit.terms[2].scale is None
    assert second.audit.terms[2].raw_gradient_rms is None
    assert second.audit.terms[0].loss_scale == pytest.approx(2.0)

    restored = OfficialWeakOuterLossComposer(config)
    restored.load_state_dict(composer.state_dict(), strict=True)
    assert torch.equal(restored.ema_values, composer.ema_values)
    assert torch.equal(restored.ema_update_counts, composer.ema_update_counts)
    assert torch.equal(restored.gradient_ema_values, composer.gradient_ema_values)
    assert int(restored.balance_schema_version.item()) == 6


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
    assert committed.terms[2].raw_gradient_rms is None


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

    with pytest.raises(RuntimeError, match="Missing key"):
        composer.load_state_dict(old_state, strict=True)
