from __future__ import annotations

import pytest

from ttt_svcbench_qwen.llamafactory_trainer import (
    _counterfactual_all_ranks_eligible,
    _counterfactual_query_selector,
)


def test_counterfactual_query_selector_is_rank_invariant() -> None:
    selectors = tuple(_counterfactual_query_selector(8) for _rank in range(4))

    assert selectors == (8, 8, 8, 8)


@pytest.mark.parametrize("optimizer_step", (0, -1, 1.5, True))
def test_counterfactual_query_selector_rejects_invalid_steps(
    optimizer_step: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _counterfactual_query_selector(optimizer_step)  # type: ignore[arg-type]


def test_counterfactual_audit_defers_when_any_rank_is_padding() -> None:
    assert _counterfactual_all_ranks_eligible((1.0, 1.0, 1.0, 1.0)) is True
    assert _counterfactual_all_ranks_eligible((1.0, 0.0, 1.0, 1.0)) is False


@pytest.mark.parametrize("loss_weights", ((), (0.5,), (1.0, 2.0)))
def test_counterfactual_rank_eligibility_rejects_invalid_weights(
    loss_weights: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="binary loss weights"):
        _counterfactual_all_ranks_eligible(loss_weights)
