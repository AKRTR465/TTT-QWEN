from __future__ import annotations

import pytest
import torch

from ttt_svcbench_qwen.training_context import (
    QueryActivationOffloadBudget,
    _OffloadedActivation,
    _query_offload_budget_bytes,
)


def test_query_offload_budget_defaults_to_eight_gib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TTT_QUERY_ACTIVATION_OFFLOAD_MAX_GB", raising=False)

    assert _query_offload_budget_bytes() == 8 * (1 << 30)


@pytest.mark.parametrize("raw", ("0", "-1", "1.5", "invalid"))
def test_query_offload_budget_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv("TTT_QUERY_ACTIVATION_OFFLOAD_MAX_GB", raw)

    with pytest.raises(ValueError, match="positive integer"):
        _query_offload_budget_bytes()


def test_query_offload_budget_accepts_positive_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TTT_QUERY_ACTIVATION_OFFLOAD_MAX_GB", "3")

    assert _query_offload_budget_bytes() == 3 * (1 << 30)


def test_shared_query_offload_budget_never_exceeds_episode_limit() -> None:
    budget = QueryActivationOffloadBudget(maximum_bytes=10)

    assert budget.claim(6)
    assert not budget.claim(5)
    budget.release(6)
    assert budget.claim(4)
    assert budget.claimed_bytes == 4
    assert budget.peak_claimed_bytes == 6
    assert budget.total_claimed_bytes == 10


def test_offloaded_activation_reuses_one_device_restore() -> None:
    source = torch.arange(4, dtype=torch.float32)
    budget = QueryActivationOffloadBudget(maximum_bytes=source.nbytes)
    assert budget.claim(source.nbytes)
    packed = _OffloadedActivation(
        device=torch.device("cpu"),
        tensor=source,
        budget=budget,
        nbytes=source.nbytes,
    )

    first = packed.restore()
    second = packed.restore()

    assert first is second
    assert torch.equal(first, source)
    assert budget.claimed_bytes == 0


def test_offloaded_activation_explicit_release_frees_unrestored_claim() -> None:
    source = torch.arange(4, dtype=torch.float32)
    budget = QueryActivationOffloadBudget(maximum_bytes=source.nbytes)
    assert budget.claim(source.nbytes)
    packed = _OffloadedActivation(
        device=torch.device("cpu"),
        tensor=source,
        budget=budget,
        nbytes=source.nbytes,
    )

    packed.release()
    packed.release()

    assert packed.tensor is None
    assert packed.restored_tensor is None
    assert budget.claimed_bytes == 0


@pytest.mark.parametrize("nbytes", (0, -1))
def test_shared_query_offload_budget_rejects_nonpositive_claim(nbytes: int) -> None:
    budget = QueryActivationOffloadBudget(maximum_bytes=10)

    with pytest.raises(ValueError, match="positive"):
        budget.claim(nbytes)


@pytest.mark.parametrize("nbytes", (0, -1, 11))
def test_shared_query_offload_budget_rejects_invalid_release(nbytes: int) -> None:
    budget = QueryActivationOffloadBudget(maximum_bytes=10)
    budget.claim(10)

    with pytest.raises(ValueError, match="release"):
        budget.release(nbytes)
