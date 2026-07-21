from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from ttt_svcbench_qwen.config import (
    OfficialWeakBalanceConfig,
    OfficialWeakBalanceMode,
)
from ttt_svcbench_qwen.losses import AnswerLossOutput, LossSkipReason, LossTerm
from ttt_svcbench_qwen.outer_loss_balance import OfficialWeakOuterLossComposer
from ttt_svcbench_qwen.stage_a_targets import (
    OfficialWeakLossAudit,
    OfficialWeakLossTerm,
    OfficialWeakStateLossOutput,
)


def _config(mode: OfficialWeakBalanceMode) -> OfficialWeakBalanceConfig:
    return OfficialWeakBalanceConfig(
        mode=mode,
        group_weight=0.3,
        scale_min=0.1,
        scale_max=10.0,
        epsilon=1.0e-8,
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


def _remote_sum(values: tuple[float, ...]) -> Callable[[Tensor], Tensor]:
    def reduce(local: Tensor) -> Tensor:
        return local + torch.tensor(values, dtype=local.dtype, device=local.device)

    return reduce


def test_legacy_sum_is_exact_and_collective_free() -> None:
    answer_value = torch.tensor(2.0, requires_grad=True)
    auxiliaries = tuple(torch.tensor(value, requires_grad=True) for value in (1.0, 2.0, 3.0, 4.0))

    def forbidden_collective(_local: Tensor) -> Tensor:
        raise AssertionError("legacy_sum must not enter a collective")

    composer = OfficialWeakOuterLossComposer(
        _config(OfficialWeakBalanceMode.LEGACY_SUM),
        reduce_sum=forbidden_collective,
        world_size=2,
    )
    output = composer.compose((_answer(answer_value),), (_state(*auxiliaries),))

    assert output.audit is None
    assert output.mean_total.item() == pytest.approx(12.0)
    output.mean_total.backward()
    assert answer_value.grad is not None and answer_value.grad.item() == pytest.approx(1.0)
    assert all(
        value.grad is not None and value.grad.item() == pytest.approx(1.0) for value in auxiliaries
    )


def test_instant_equal_assigns_four_fixed_slots_and_keeps_scales_detached() -> None:
    answer_value = torch.tensor(4.0, requires_grad=True)
    auxiliaries = tuple(torch.tensor(2.0, requires_grad=True) for _ in range(4))
    composer = OfficialWeakOuterLossComposer(_config(OfficialWeakBalanceMode.INSTANT_EQUAL))
    assert composer.state_dict() == {}

    output = composer.compose((_answer(answer_value),), (_state(*auxiliaries),))

    assert output.audit is not None
    assert output.mean_total.item() == pytest.approx(5.2)
    assert output.audit.auxiliary_to_answer_ratio == pytest.approx(0.3)
    assert all(term.scale == pytest.approx(2.0) for term in output.audit.terms)
    assert all(term.weighted_global_mean == pytest.approx(0.3) for term in output.audit.terms)
    output.mean_total.backward()
    assert answer_value.grad is not None and answer_value.grad.item() == pytest.approx(1.0)
    assert all(
        value.grad is not None and value.grad.item() == pytest.approx(0.15) for value in auxiliaries
    )


def test_group_guard_caps_extreme_auxiliary_at_thirty_percent_of_answer() -> None:
    answer_value = torch.tensor(1.0, requires_grad=True)
    task = torch.tensor(1000.0, requires_grad=True)
    ordinary = tuple(torch.tensor(1.0, requires_grad=True) for _ in range(3))
    composer = OfficialWeakOuterLossComposer(_config(OfficialWeakBalanceMode.INSTANT_EQUAL))

    output = composer.compose((_answer(answer_value),), (_state(task, *ordinary),))

    assert output.audit is not None
    assert output.audit.group_guard_active
    assert output.audit.state_global_mean == pytest.approx(0.3)
    assert output.audit.auxiliary_to_answer_ratio == pytest.approx(0.3)
    assert output.mean_total.item() == pytest.approx(1.3)


def test_missing_terms_keep_empty_slots_instead_of_redistributing_budget() -> None:
    answer_value = torch.tensor(4.0, requires_grad=True)
    values = tuple(torch.tensor(4.0, requires_grad=True) for _ in range(4))
    composer = OfficialWeakOuterLossComposer(_config(OfficialWeakBalanceMode.INSTANT_EQUAL))

    one_term = composer.compose(
        (_answer(answer_value),),
        (_state(*values, valid=(True, False, False, False)),),
    )
    no_terms = composer.compose(
        (_answer(answer_value),),
        (_state(*values, valid=(False, False, False, False)),),
    )

    assert one_term.mean_total.item() == pytest.approx(4.3)
    assert no_terms.mean_total.item() == pytest.approx(4.0)
    assert one_term.audit is not None
    assert tuple(term.global_valid_count for term in one_term.audit.terms) == (1, 0, 0, 0)


def test_sparse_rank_term_uses_global_valid_count_for_ddp_gradient() -> None:
    answer_value = torch.tensor(2.0, requires_grad=True)
    retrieval = torch.tensor(4.0, requires_grad=True)
    anchor = torch.tensor(0.0, requires_grad=True)
    # Remote rank contributes one Answer row with sum 6 and no official-weak rows.
    remote = (6.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    composer = OfficialWeakOuterLossComposer(
        _config(OfficialWeakBalanceMode.INSTANT_EQUAL),
        reduce_sum=_remote_sum(remote),
        world_size=2,
    )

    output = composer.compose(
        (_answer(answer_value),),
        (_state(anchor, anchor, retrieval, anchor, valid=(False, False, True, False)),),
    )
    output.mean_total.backward()

    assert output.audit is not None
    assert output.audit.answer_global_mean == pytest.approx(4.0)
    assert output.audit.terms[2].global_valid_count == 1
    # Local coefficient is W/N=2; fixed slot and group weight make it 2*0.3/4.
    assert retrieval.grad is not None and retrieval.grad.item() == pytest.approx(0.15)


def test_multiple_local_queries_mean_to_one_global_rank_objective() -> None:
    answers = tuple(torch.tensor(value, requires_grad=True) for value in (2.0, 4.0))
    task_values = tuple(torch.tensor(value, requires_grad=True) for value in (1.0, 3.0))
    anchor = torch.tensor(0.0, requires_grad=True)
    # Remote rank contributes one Answer=6 and one Task=2.
    remote = (6.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    collective_calls = 0

    def reduce(local: Tensor) -> Tensor:
        nonlocal collective_calls
        collective_calls += 1
        return _remote_sum(remote)(local)

    composer = OfficialWeakOuterLossComposer(
        _config(OfficialWeakBalanceMode.INSTANT_EQUAL),
        reduce_sum=reduce,
        world_size=2,
    )
    states = tuple(
        _state(task, anchor, anchor, anchor, valid=(True, False, False, False))
        for task in task_values
    )

    output = composer.compose(tuple(_answer(value) for value in answers), states)

    assert len(output.objectives) == 2
    assert collective_calls == 1
    # This rank owns two of three global rows, so its DDP-correct local objective is 4.4;
    # averaging it with the remote rank's 4.2 yields the global 4.3 objective.
    assert output.mean_total.item() == pytest.approx(4.4)
    assert output.audit is not None
    assert output.audit.answer_global_mean == pytest.approx(4.0)
    assert output.audit.terms[0].raw_global_mean == pytest.approx(2.0)


def test_config_rejects_inverted_scale_bounds() -> None:
    with pytest.raises(ValueError, match="scale_min"):
        OfficialWeakBalanceConfig(
            mode=OfficialWeakBalanceMode.INSTANT_EQUAL,
            group_weight=0.3,
            scale_min=10.0,
            scale_max=0.1,
            epsilon=1.0e-8,
        )


def test_ema_answer_reference_persists_and_missing_term_keeps_history() -> None:
    config = OfficialWeakBalanceConfig(
        mode=OfficialWeakBalanceMode.EMA_ANSWER_REF,
        group_weight=0.3,
        scale_min=0.001,
        scale_max=20.0,
        epsilon=1.0e-8,
        ema_beta=0.5,
    )
    composer = OfficialWeakOuterLossComposer(config)
    values = tuple(torch.tensor(2.0, requires_grad=True) for _ in range(4))
    first = composer.compose((_answer(torch.tensor(4.0, requires_grad=True)),), (_state(*values),))
    assert first.audit is not None
    assert first.audit.ema_means == pytest.approx((4.0, 2.0, 2.0, 2.0, 2.0))
    assert first.audit.ema_update_counts == (1, 1, 1, 1, 1)

    second_values = tuple(torch.tensor(8.0, requires_grad=True) for _ in range(4))
    second = composer.compose(
        (_answer(torch.tensor(2.0, requires_grad=True)),),
        (_state(*second_values, valid=(True, True, False, True)),),
    )
    assert second.audit is not None
    assert second.audit.ema_means == pytest.approx((3.0, 5.0, 5.0, 2.0, 5.0))
    assert second.audit.ema_update_counts == (2, 2, 2, 1, 2)
    assert second.audit.terms[2].global_valid_count == 0
    assert second.audit.terms[2].scale == pytest.approx(1.5)

    restored = OfficialWeakOuterLossComposer(config)
    restored.load_state_dict(composer.state_dict(), strict=True)
    assert torch.equal(restored.ema_values, composer.ema_values)
    assert torch.equal(restored.ema_update_counts, composer.ema_update_counts)
