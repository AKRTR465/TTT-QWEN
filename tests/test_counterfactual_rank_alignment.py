from __future__ import annotations

import pytest

from ttt_svcbench_qwen.llamafactory_trainer import _counterfactual_query_selector


def test_counterfactual_query_selector_is_rank_invariant() -> None:
    selectors = tuple(_counterfactual_query_selector(8) for _rank in range(4))

    assert selectors == (8, 8, 8, 8)


@pytest.mark.parametrize("optimizer_step", (0, -1, 1.5, True))
def test_counterfactual_query_selector_rejects_invalid_steps(
    optimizer_step: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _counterfactual_query_selector(optimizer_step)  # type: ignore[arg-type]
