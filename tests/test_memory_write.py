from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.fast_ttt import (
    FastMemoryState,
    GradientMode,
    MemoryWriteBatch,
    MemoryWriteSkipReason,
    apply_memory_write,
    apply_memory_writes,
    build_fast_ttt_adapter,
    deferred_fast_vjp_loss,
    make_query_proxy_fast_state,
    truncate_memory_state,
    truncate_memory_states,
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
    mask = torch.ones(etas.shape, dtype=torch.bool) if slot_mask is None else slot_mask
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


class _Slots:
    def __init__(self, slots: Tensor) -> None:
        count = slots.shape[1]
        self.slots = slots
        self.slot_valid_mask = torch.ones(1, count, dtype=torch.bool)
        self.slot_confidence = torch.rand(1, count)


def test_single_pair_write_recalls_exact_direction_with_eta_magnitude() -> None:
    torch.manual_seed(3)
    key = _unit(torch.randn(768))
    value = _unit(torch.randn(768))
    eta = 0.25
    batch = _batch(
        keys=key.reshape(1, 1, 768),
        values=value.reshape(1, 1, 768),
        etas=torch.tensor([[eta]]),
        beta=0.01,
    )

    result = apply_memory_write(fast_state=_zero_state(), batch=batch, row=0)

    assert result.did_write and result.slots_written == 1
    assert result.gradient_mode is GradientMode.ONLINE_LEAF
    recall = result.fast_state.m.detach() @ key
    cosine = float(F.cosine_similarity(recall, value, dim=0))
    assert cosine == pytest.approx(1.0, abs=1.0e-6)
    assert float(recall.norm()) == pytest.approx(eta, rel=1.0e-5)
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


def test_eta_budget_is_respected_and_renormalizes_when_saturated() -> None:
    """Sum(eta) <= eta_chunk_budget, the delta-rule contraction precondition."""

    config = load_config()
    adapter = build_fast_ttt_adapter(config)
    adapter(torch.randn(1, 6, 4096), fast_state=None)
    intermediates = adapter.consume_associative_intermediates()
    assert intermediates is not None

    view = _Slots(torch.randn(1, 32, 768))
    batch = adapter.prepare_write(intermediates, view)
    # The shipped configuration must start with budget headroom, otherwise the
    # renormalizer fires on every write from step zero and the gate can only set
    # the relative distribution of eta, never the total.
    assert batch.eta_renormalized == (False,)
    expected_total = (
        float(config.spatial_encoder.active_slots) * config.fast_memory.eta_gate_init
    )
    assert expected_total < config.fast_memory.eta_chunk_budget
    # Exact, not approximate: the gate's output projection is zero-initialized, so
    # the data term contributes nothing and the total is the product alone.
    assert float(batch.etas.detach().sum()) == pytest.approx(expected_total, rel=1.0e-6)

    with torch.no_grad():
        adapter.memory_eta_gate_output.bias.fill_(20.0)
    saturated = adapter.prepare_write(intermediates, view)
    assert saturated.eta_renormalized == (True,)
    assert float(saturated.etas.detach().sum()) == pytest.approx(
        config.fast_memory.eta_chunk_budget, rel=1.0e-5
    )


@pytest.mark.parametrize("seed", (0, 3, 11))
@pytest.mark.parametrize("slot_norm", (9.0, 70.0))
def test_initial_eta_total_is_seed_and_scale_independent(
    seed: int, slot_norm: float
) -> None:
    """Pin the initial write budget against near-collinear production slots.

    ``eta_gate_init`` only bounds the chunk total if the gate's data term is zero
    at construction.  Production slots are near-collinear, so that term is one DC
    offset shared by all 32 slots: it does not average away and it grows with the
    slot norm.  Random near-orthogonal slots hide this.
    """
    config = load_config()
    slots = config.spatial_encoder.active_slots
    torch.manual_seed(seed)
    adapter = build_fast_ttt_adapter(config)
    adapter(torch.randn(1, 6, 4096), fast_state=None)
    intermediates = adapter.consume_associative_intermediates()
    assert intermediates is not None

    generator = torch.Generator().manual_seed(seed + 9_000)
    shared = torch.randn(1, 1, 768, generator=generator)
    shared = shared / shared.norm() * slot_norm
    spread = torch.randn(1, slots, 768, generator=generator)
    spread = spread / spread.norm(dim=-1, keepdim=True) * (slot_norm * 0.04)

    batch = adapter.prepare_write(intermediates, _Slots(shared + spread))
    assert batch.eta_renormalized == (False,)
    assert float(batch.etas.detach().sum()) == pytest.approx(
        float(slots) * config.fast_memory.eta_gate_init, rel=1.0e-6
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
    """The silent-skip path: a row with no valid slot advances nothing but skip_count."""

    keys = _unit(torch.randn(2, 3, 768))
    values = _unit(torch.randn(2, 3, 768))
    etas = torch.full((2, 3), 0.1)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    batch = _batch(keys=keys, values=values, etas=etas, slot_mask=mask)
    states = (_zero_state(), _zero_state())

    results = apply_memory_writes(fast_states=states, batch=batch)

    assert results[0].did_write and results[0].slots_written == 2
    assert not results[1].did_write
    assert results[1].slots_written == 0
    assert results[1].skip_reason is MemoryWriteSkipReason.NO_VALID_SLOT
    assert results[1].fast_state.skip_count == 1
    assert results[1].fast_state.write_version == 0
    assert results[1].fast_state.write_count == 0
    assert torch.equal(results[1].fast_state.m.detach(), torch.zeros((768, 768)))


def test_nonfinite_payload_rows_skip_without_poisoning_the_batch() -> None:
    keys = _unit(torch.randn(1, 2, 768))
    keys[0, 0, 0] = math.nan
    values = _unit(torch.randn(1, 2, 768))
    batch = _batch(keys=keys, values=values, etas=torch.full((1, 2), 0.1))

    result = apply_memory_write(fast_state=_zero_state(), batch=batch, row=0)

    assert not result.did_write
    assert result.skip_reason is MemoryWriteSkipReason.NONFINITE_KEY_VALUE
    assert result.fast_state.skip_count == 1
    assert torch.equal(result.fast_state.m.detach(), torch.zeros((768, 768)))


def test_slot_centering_removes_the_shared_component_and_guards_degeneracy() -> None:
    from ttt_svcbench_qwen.fast_ttt import _centered_over_valid_slots

    torch.manual_seed(0)
    shared = torch.randn(768) * 5.0
    slots = torch.stack(
        [shared + 0.05 * torch.randn(768) for _ in range(6)]
    ).unsqueeze(0)
    full = torch.ones(1, 6, dtype=torch.bool)

    centered = _centered_over_valid_slots(slots, full)
    # Each slot's norm must be restored.  Centering alone shrinks the probe input
    # by ~150x, which flattens the attention softmax and puts every slot back on
    # the same pooled token mean -- the mirror image of the failure being fixed.
    assert torch.allclose(centered[0].norm(dim=-1), slots[0].norm(dim=-1), rtol=1.0e-4)

    # The mean must ignore invalid slots, or their garbage pollutes every row.
    # Compare directions only: the magnitude is deliberately rescaled back.
    partial = torch.tensor([[True, True, True, False, False, False]])
    expected = slots[0][:3] - slots[0][:3].mean(dim=0, keepdim=True)
    got = _centered_over_valid_slots(slots, partial)[0][:3]
    assert torch.allclose(
        got / got.norm(dim=-1, keepdim=True),
        expected / expected.norm(dim=-1, keepdim=True),
        atol=1.0e-5,
    )

    # Fewer than two valid slots must pass through unchanged: centering one slot
    # would yield exactly zero.
    for degenerate in (
        torch.tensor([[True, False, False, False, False, False]]),
        torch.zeros(1, 6, dtype=torch.bool),
    ):
        assert torch.equal(_centered_over_valid_slots(slots, degenerate), slots)


def test_meta_mode_gradients_reach_write_inputs_but_never_slot_states() -> None:
    torch.manual_seed(11)
    adapter = build_fast_ttt_adapter(load_config())
    state = adapter.initialize_fast_state(differentiable=True)
    visual = torch.randn(1, 5, 4096, requires_grad=True)
    adapter(visual, fast_state=(state,))
    intermediates = adapter.consume_associative_intermediates()
    assert intermediates is not None

    slots = torch.randn(1, 32, 768, requires_grad=True)
    batch = adapter.prepare_write(intermediates, _Slots(slots))
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
    adapter(torch.randn(1, 4, 4096), fast_state=(state,))
    intermediates = adapter.consume_associative_intermediates()
    assert intermediates is not None

    batch = adapter.prepare_write(intermediates, _Slots(torch.randn(1, 8, 768)))
    written = apply_memory_write(fast_state=state, batch=batch, row=0).fast_state
    assert written.m.grad_fn is not None

    truncated = truncate_memory_state(written)

    assert torch.equal(truncated.m.detach(), written.m.detach())
    assert truncated.m.is_leaf and truncated.m.requires_grad
    assert truncated.m.grad_fn is None
    assert truncated.differentiable
    assert (truncated.write_version, truncated.write_count, truncated.skip_count) == (
        written.write_version,
        written.write_count,
        written.skip_count,
    )
    # The detach at the K=8 boundary is what bounds the meta-gradient window: no
    # write parameter from before the cut may be reachable from the new leaf.
    gate_gradient = torch.autograd.grad(
        truncated.m.sum(),
        adapter.memory_eta_gate_output.bias,
        allow_unused=True,
    )[0]
    assert gate_gradient is None

    (plural,) = truncate_memory_states((written,))
    assert torch.equal(plural.m.detach(), written.m.detach())
    assert plural.m.is_leaf and plural.m.grad_fn is None


def test_real_k8_query_gradient_reaches_all_token_keys_then_cuts() -> None:
    torch.manual_seed(101)
    adapter = build_fast_ttt_adapter(load_config()).eval()
    initial = adapter.initialize_fast_state(differentiable=True)
    state = initial
    token_keys: list[Tensor] = []

    for step in range(8):
        visual = torch.randn(1, 3, 4096, requires_grad=True)
        slots = torch.randn(1, 1, 768, requires_grad=True)
        adapter(visual, fast_state=(state,))
        intermediates = adapter.consume_associative_intermediates()
        assert intermediates is not None
        token_keys.append(intermediates.keys)
        result = apply_memory_write(
            fast_state=state,
            batch=adapter.prepare_write(intermediates, _Slots(slots)),
            row=0,
        )
        assert result.did_write
        assert result.gradient_mode is GradientMode.META_LINEAR_RECURRENCE
        state = result.fast_state
        assert state.write_count == state.write_version == step + 1

    proxy = make_query_proxy_fast_state(state)
    query_loss = adapter(torch.randn(1, 3, 4096), fast_state=(proxy,)).square().mean()
    proxy_gradient, direct_gradient = torch.autograd.grad(
        query_loss,
        (proxy.m, state.m),
        allow_unused=True,
    )
    assert proxy_gradient is not None and torch.count_nonzero(proxy_gradient) > 0
    assert direct_gradient is None

    write_parameters = adapter.collect_memory_write_parameters()
    targets = (*write_parameters, adapter.p_in.weight, *token_keys, initial.m)
    gradients = torch.autograd.grad(
        deferred_fast_vjp_loss((state,), (proxy_gradient.detach(),)),
        targets,
        allow_unused=True,
    )
    write_gradients = gradients[: len(write_parameters)]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    for index in (0, 1, 2, 3, 6, 7, 8):
        assert torch.count_nonzero(write_gradients[index]) > 0  # type: ignore[arg-type]
    key_start = len(write_parameters) + 1
    key_gradients = gradients[key_start : key_start + len(token_keys)]
    assert all(torch.count_nonzero(gradient) > 0 for gradient in key_gradients)  # type: ignore[arg-type]
    assert torch.count_nonzero(gradients[-1]) > 0  # type: ignore[arg-type]

    before = state.m.detach().clone()
    truncated = truncate_memory_state(state)
    assert torch.equal(truncated.m.detach(), before)
    assert truncated.m.is_leaf and truncated.m.grad_fn is None
    cut = torch.autograd.grad(
        truncated.m.sum(),
        (state.m, *write_parameters, adapter.p_in.weight, *token_keys, initial.m),
        allow_unused=True,
    )
    assert all(gradient is None for gradient in cut)
