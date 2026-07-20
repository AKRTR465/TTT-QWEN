from __future__ import annotations

import pytest

from ttt_svcbench_qwen.loader_tuning import LoaderTrial, select_loader_profile


def _trial(
    workers: int,
    prefetch_factor: int,
    *,
    wait: float,
    support_read: float = 10.0,
    memory: float = 32.0,
) -> LoaderTrial:
    return LoaderTrial(
        workers=workers,
        prefetch_factor=prefetch_factor,
        ga_group_wait_p95_seconds=wait,
        support_read_p95_seconds=support_read,
        host_memory_gib=memory,
        host_memory_budget_gib=64.0,
    )


def test_loader_selector_accepts_four_by_one_only_after_every_gate() -> None:
    assert select_loader_profile(
        (_trial(2, 2, wait=100.0), _trial(4, 1, wait=80.0, support_read=11.0))
    ) == (4, 1)


@pytest.mark.parametrize(
    "candidate",
    (
        _trial(4, 1, wait=80.1),
        _trial(4, 1, wait=79.0, support_read=11.1),
        _trial(4, 1, wait=79.0, memory=64.1),
    ),
)
def test_loader_selector_retains_two_by_two_when_any_gate_fails(
    candidate: LoaderTrial,
) -> None:
    assert select_loader_profile((_trial(2, 2, wait=100.0), candidate)) == (2, 2)


def test_loader_selector_requires_a_safe_baseline() -> None:
    with pytest.raises(ValueError, match="2x2 baseline"):
        select_loader_profile((_trial(4, 1, wait=20.0),))
