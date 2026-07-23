from __future__ import annotations

import pytest

from ttt_svcbench_qwen.batch_tuning import VisualBatchTrial, select_visual_batch_size


def _trial(size: int, p50: float, *, free: float = 20.0) -> VisualBatchTrial:
    return VisualBatchTrial(size, False, free, p50, True, True)


def test_a2_selects_largest_sequential_batch_with_five_percent_gain() -> None:
    selected = select_visual_batch_size(
        "a2",
        (_trial(1, 10.0), _trial(2, 9.0), _trial(4, 8.4), _trial(8, 8.2)),
    )

    assert selected == 4


def test_a5_requires_batch_two_to_pass_before_batch_four() -> None:
    assert (
        select_visual_batch_size(
            "a5",
            (_trial(1, 10.0), _trial(2, 9.8), _trial(4, 8.0)),
        )
        == 1
    )
    with pytest.raises(ValueError, match="baseline"):
        select_visual_batch_size("a5", (_trial(1, 10.0, free=8.0),))
