from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.fast_ttt import (
    FastMemoryState,
    MemoryWriteBatch,
    build_fast_ttt_adapter,
)
from ttt_svcbench_qwen.memory_write import (
    GradientMode,
    MemoryWriteSkipReason,
    apply_memory_write,
    apply_memory_writes,
    truncate_memory_state,
)


def _zero_state(*, differentiable: bool = False) -> FastMemoryState:
    return FastMemoryState(
        m=torch.zeros((768, 768), dtype=torch.float32, requires_grad=True),
        write_version=0,
        write_count=0,
        skip_count=0,
        differentiable=differentiable,
    )


def _batch(
    *,
    keys: Tensor,
    values: Tensor,
    etas: Tensor,
    slot_mask: Tensor | None = None,
    beta: float = 0.01,
) -> MemoryWriteBatch:
    mask = (
        torch.ones(etas.shape, dtype=torch.bool) if slot_mask is None else slot_mask
    )
    return MemoryWriteBatch(
        keys=keys * mask.unsqueeze(-1),
        values=values * mask.unsqueeze(-1),
        etas=etas * mask,
        slot_mask=mask,
        beta=torch.tensor(beta),
        eta_renormalized=(False,) * keys.shape[0],
    )


def _unit(vector: Tensor) -> Tensor:
    return F.normalize(vector, dim=-1)


def test_single_pair_write_recalls_exact_direction_with_eta_magnitude() -> None:
    torch.manual_seed(3)
    key = _unit(torch.randn(768))
    value = _unit(torch.randn(768))
    eta = 0.25
    beta = 0.01
    batch = _batch(
        keys=key.reshape(1, 1, 768),
        values=value.reshape(1, 1, 768),
        etas=torch.tensor([[eta]]),
        beta=beta,
    )

    result = apply_memory_write(fast_state=_zero_state(), batch=batch, row=0)

    assert result.did_write and result.slots_written == 1
    assert result.gradient_mode is GradientMode.ONLINE_LEAF
    recall = result.fast_state.m.detach() @ key
    cosine = float(F.cosine_similarity(recall, value, dim=0))
    assert cosine == pytest.approx(1.0, abs=1.0e-6)
    assert float(recall.norm()) == pytest.approx(eta, rel=1.0e-5)
    assert result.pre_write_cosine_mean == pytest.approx(0.0)
    assert result.post_write_cosine_mean == pytest.approx(1.0, abs=1.0e-5)
    assert result.eta_sum == pytest.approx(eta)
    assert result.fast_state.write_version == result.fast_state.write_count == 1
    assert result.fast_state.m.is_leaf and result.fast_state.m.requires_grad


def test_parallel_write_is_slot_order_invariant() -> None:
    torch.manual_seed(5)
    keys = _unit(torch.randn(1, 8, 768))
    values = _unit(torch.randn(1, 8, 768))
    etas = torch.full((1, 8), 0.1)
    permutation = torch.randperm(8)

    forward = apply_memory_write(
        fast_state=_zero_state(),
        batch=_batch(keys=keys, values=values, etas=etas),
        row=0,
    )
    permuted = apply_memory_write(
        fast_state=_zero_state(),
        batch=_batch(
            keys=keys[:, permutation],
            values=values[:, permutation],
            etas=etas[:, permutation],
        ),
        row=0,
    )

    torch.testing.assert_close(
        forward.fast_state.m.detach(),
        permuted.fast_state.m.detach(),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_eta_budget_renormalization_is_enforced_by_the_batch_contract() -> None:
    keys = _unit(torch.randn(1, 8, 768))
    values = _unit(torch.randn(1, 8, 768))
    with pytest.raises(ValueError, match="unit chunk budget"):
        _batch(keys=keys, values=values, etas=torch.full((1, 8), 0.25))

    config = load_config()
    adapter = build_fast_ttt_adapter(config)
    visual = torch.randn(1, 6, 4096)
    adapter(visual, fast_state=None)
    intermediates = adapter.consume_associative_intermediates()

    class _Slots:
        slots = torch.randn(1, 32, 768)
        slot_valid_mask = torch.ones(1, 32, dtype=torch.bool)
        slot_confidence = torch.rand(1, 32)

    batch = adapter.prepare_write(intermediates, _Slots())
    assert batch.eta_renormalized == (True,)
    assert float(batch.etas.detach().sum()) == pytest.approx(
        config.fast_memory.eta_chunk_budget, rel=1.0e-5
    )


def test_contraction_bounds_memory_norm_under_two_hundred_writes() -> None:
    torch.manual_seed(7)
    state = _zero_state()
    beta = 0.05
    for step in range(200):
        torch.manual_seed(step)
        keys = _unit(torch.randn(1, 4, 768))
        values = _unit(torch.randn(1, 4, 768))
        etas = torch.full((1, 4), 0.25)
        result = apply_memory_write(
            fast_state=state,
            batch=_batch(keys=keys, values=values, etas=etas, beta=beta),
            row=0,
        )
        assert result.did_write
        state = result.fast_state
        operator_norm = float(torch.linalg.matrix_norm(state.m.detach(), ord=2))
        assert operator_norm <= min(float(step + 1), 1.0 / beta) + 1.0e-4
    assert state.write_count == 200


def test_masks_and_skip_reasons_are_fail_closed() -> None:
    keys = _unit(torch.randn(2, 3, 768))
    values = _unit(torch.randn(2, 3, 768))
    etas = torch.full((2, 3), 0.1)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    batch = _batch(keys=keys, values=values, etas=etas, slot_mask=mask)
    states = (_zero_state(), _zero_state())

    results = apply_memory_writes(fast_states=states, batch=batch)

    assert results[0].did_write and results[0].slots_written == 2
    assert not results[1].did_write
    assert results[1].skip_reason is MemoryWriteSkipReason.NO_VALID_SLOT
    assert results[1].fast_state.skip_count == 1
    assert results[1].fast_state.write_version == 0
    assert torch.equal(
        results[1].fast_state.m.detach(),
        torch.zeros((768, 768)),
    )
    with pytest.raises(ValueError, match="zero-pad invalid slots"):
        MemoryWriteBatch(
            keys=keys,
            values=values * mask.unsqueeze(-1),
            etas=etas * mask,
            slot_mask=mask,
            beta=torch.tensor(0.01),
            eta_renormalized=(False, False),
        )
    with pytest.raises(ValueError, match="exactly zero eta"):
        MemoryWriteBatch(
            keys=keys * mask.unsqueeze(-1),
            values=values * mask.unsqueeze(-1),
            etas=etas,
            slot_mask=mask,
            beta=torch.tensor(0.01),
            eta_renormalized=(False, False),
        )
    with pytest.raises(ValueError, match="strictly inside"):
        _batch(keys=keys, values=values, etas=etas, slot_mask=mask, beta=0.0)


def test_fp32_master_and_storage_isolation_are_mandatory() -> None:
    with pytest.raises(ValueError, match="float32"):
        FastMemoryState(
            m=torch.zeros((768, 768), dtype=torch.float64, requires_grad=True),
            write_version=0,
            write_count=0,
            skip_count=0,
        )
    state = _zero_state()
    shared = FastMemoryState(
        m=state.m,
        write_version=0,
        write_count=0,
        skip_count=0,
    )
    batch = _batch(
        keys=_unit(torch.randn(2, 1, 768)),
        values=_unit(torch.randn(2, 1, 768)),
        etas=torch.full((2, 1), 0.1),
    )
    with pytest.raises(ValueError, match="storage-isolated"):
        apply_memory_writes(fast_states=(state, shared), batch=batch)


def test_meta_mode_gradients_reach_write_inputs_but_never_slot_states() -> None:
    torch.manual_seed(11)
    adapter = build_fast_ttt_adapter(load_config())
    state = adapter.initialize_fast_state(differentiable=True)
    visual = torch.randn(1, 5, 4096, requires_grad=True)
    adapter(visual, fast_state=(state,))
    intermediates = adapter.consume_associative_intermediates()

    slots = torch.randn(1, 32, 768, requires_grad=True)

    class _Slots:
        slot_valid_mask = torch.ones(1, 32, dtype=torch.bool)
        slot_confidence = torch.rand(1, 32)

    view = _Slots()
    view.slots = slots  # type: ignore[attr-defined]
    batch = adapter.prepare_write(intermediates, view)
    result = apply_memory_write(fast_state=state, batch=batch, row=0)
    assert result.gradient_mode is GradientMode.META_LINEAR_RECURRENCE

    loss = result.fast_state.m.square().sum()
    gradients = torch.autograd.grad(
        loss,
        (
            adapter.memory_eta_gate_output.bias,
            adapter.memory_key_probe.weight,
            adapter.memory_value_projection.weight,
            adapter.memory_beta_raw,
            adapter.p_in.weight,
            state.m,
            slots,
        ),
        retain_graph=True,
        allow_unused=True,
    )
    for name, gradient in zip(
        ("gate", "probe", "value", "beta", "p_in", "previous_memory"),
        gradients[:6],
        strict=True,
    ):
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
    assert gradients[6] is None  # slot states stay detached in probe and value paths


def test_truncation_cuts_gradient_and_preserves_values_bitwise() -> None:
    torch.manual_seed(13)
    adapter = build_fast_ttt_adapter(load_config())
    state = adapter.initialize_fast_state(differentiable=True)
    visual = torch.randn(1, 4, 4096)
    adapter(visual, fast_state=(state,))
    intermediates = adapter.consume_associative_intermediates()

    class _Slots:
        slots = torch.randn(1, 8, 768)
        slot_valid_mask = torch.ones(1, 8, dtype=torch.bool)
        slot_confidence = torch.rand(1, 8)

    batch = adapter.prepare_write(intermediates, _Slots())
    written = apply_memory_write(fast_state=state, batch=batch, row=0).fast_state
    assert written.m.grad_fn is not None

    truncated, audit = truncate_memory_state(written)

    assert torch.equal(truncated.m.detach(), written.m.detach())
    assert truncated.m.is_leaf and truncated.m.requires_grad
    assert truncated.m.grad_fn is None
    assert audit.max_abs_value_drift == 0.0
    assert audit.old_graph_truncated and audit.storage_isolated
    assert (truncated.write_version, truncated.write_count, truncated.skip_count) == (
        written.write_version,
        written.write_count,
        written.skip_count,
    )
    gate_gradient = torch.autograd.grad(
        truncated.m.sum(),
        adapter.memory_eta_gate_output.bias,
        allow_unused=True,
    )[0]
    assert gate_gradient is None
    with pytest.raises(ValueError, match="differentiable"):
        truncate_memory_state(_zero_state())


def test_nonfinite_payload_rows_skip_without_poisoning_the_batch() -> None:
    keys = _unit(torch.randn(1, 2, 768))
    values = _unit(torch.randn(1, 2, 768))
    batch = _batch(keys=keys, values=values, etas=torch.full((1, 2), 0.1))
    poisoned = MemoryWriteBatch.__new__(MemoryWriteBatch)
    object.__setattr__(poisoned, "keys", batch.keys.clone())
    object.__setattr__(poisoned, "values", batch.values.clone())
    object.__setattr__(poisoned, "etas", batch.etas.clone())
    object.__setattr__(poisoned, "slot_mask", batch.slot_mask.clone())
    object.__setattr__(poisoned, "beta", batch.beta.clone())
    object.__setattr__(poisoned, "eta_renormalized", batch.eta_renormalized)
    with torch.no_grad():
        poisoned.keys[0, 0, 0] = math.nan

    result = apply_memory_write(fast_state=_zero_state(), batch=poisoned, row=0)

    assert not result.did_write
    assert result.skip_reason is MemoryWriteSkipReason.NONFINITE_KEY_VALUE
    assert result.fast_state.skip_count == 1
