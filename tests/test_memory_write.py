from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F

from ttt_svcbench_qwen.associative_ttt import AssociativeTTTIntermediates
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.fast_ttt import (
    PROBE_FIELDS,
    FastMemoryState,
    MemoryWriteBatch,
    SlotGeometryProbe,
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
    # The shipped configuration must start with budget headroom.  G4 requires the
    # eta renormalization rate to stay under 20%, so a default where
    # active_slots * eta_gate_init already exceeds eta_chunk_budget makes the
    # renormalizer fire on every write from step zero and the gate can only ever
    # set the relative distribution of eta, never the total.  This assertion used
    # to read `(True,)` -- it passed only *because* of that misconfiguration.
    assert batch.eta_renormalized == (False,)
    expected_total = (
        float(config.spatial_encoder.active_slots) * config.fast_memory.eta_gate_init
    )
    assert expected_total < config.fast_memory.eta_chunk_budget
    # Exact, not approximate: the gate's output projection is zero-initialized, so
    # the data term contributes nothing and the total is the product alone.  This
    # used to carry `rel=0.2` to absorb a seed-dependent DC offset -- the slack that
    # let the shipped default renormalize 100% of H200 chunks while passing here.
    assert float(batch.etas.detach().sum()) == pytest.approx(expected_total, rel=1.0e-6)

    # Forcing the gate wide open must still clamp to exactly the budget, so the
    # renormalizer itself stays covered now that the default no longer trips it.
    with torch.no_grad():
        adapter.memory_eta_gate_output.bias.fill_(20.0)
    saturated = adapter.prepare_write(intermediates, _Slots())
    assert saturated.eta_renormalized == (True,)
    assert float(saturated.etas.detach().sum()) == pytest.approx(
        config.fast_memory.eta_chunk_budget, rel=1.0e-5
    )


def test_prepare_write_attention_is_invariant_to_a_shared_token_shift() -> None:
    """The property token-key centering leans on, pinned as a regression.

    Adding one constant vector to every token key shifts each slot's score row
    uniformly, and softmax over the token axis is shift-invariant, so token
    *selection* must not change -- centering the keys may change what is
    stored, never what is attended.  Etas read slots, not keys, so they must be
    bitwise identical.
    """

    torch.manual_seed(5)
    adapter = build_fast_ttt_adapter(load_config())
    adapter(torch.randn(1, 6, 4096), fast_state=None)
    inter = adapter.consume_associative_intermediates()
    shift = torch.randn(1, 1, 768) * 7.0
    shifted = AssociativeTTTIntermediates(
        keys=inter.keys + shift,
        predictions=inter.predictions,
        valid_mask=inter.valid_mask,
        bank_record_counts=inter.bank_record_counts,
        bank_versions=inter.bank_versions,
    )

    class _Slots:
        slots = torch.randn(1, 32, 768)
        slot_valid_mask = torch.ones(1, 32, dtype=torch.bool)
        slot_confidence = torch.rand(1, 32)

    view = _Slots()
    plain = adapter.prepare_write(inter, view)
    moved = adapter.prepare_write(shifted, view)

    assert torch.equal(plain.etas.detach(), moved.etas.detach())
    for before, after in zip(
        plain.slot_geometry_probes, moved.slot_geometry_probes, strict=True
    ):
        assert after.attention_entropy_ratio == pytest.approx(
            before.attention_entropy_ratio, abs=1.0e-5
        )


@pytest.mark.parametrize("seed", (0, 3, 11))
@pytest.mark.parametrize("slot_norm", (9.0, 70.0))
def test_initial_eta_total_is_seed_and_scale_independent(seed: int, slot_norm: float) -> None:
    """Pin the initial write budget against the production slot geometry.

    ``eta_gate_init`` only bounds the chunk total if the gate's data term is zero
    at construction.  Production slots are near-collinear, so that term is one DC
    offset shared by all 32 slots -- it does not average away, it is drawn once per
    seed, and it grows linearly with the slot norm.  Random near-orthogonal slots
    hide this (the per-slot draws cancel), which is why every existing test passed
    on this box while H200 renormalized every chunk.  Sweeping seed x scale against
    collinear slots is the assertion that reproduces the failure locally.
    """

    config = load_config()
    slots = config.spatial_encoder.active_slots
    torch.manual_seed(seed)
    adapter = build_fast_ttt_adapter(config)
    adapter(torch.randn(1, 6, 4096), fast_state=None)
    intermediates = adapter.consume_associative_intermediates()

    generator = torch.Generator().manual_seed(seed + 9_000)
    shared = torch.randn(1, 1, 768, generator=generator)
    shared = shared / shared.norm() * slot_norm
    spread = torch.randn(1, slots, 768, generator=generator)
    spread = spread / spread.norm(dim=-1, keepdim=True) * (slot_norm * 0.04)

    class _Collinear:
        slot_valid_mask = torch.ones(1, slots, dtype=torch.bool)
        slot_confidence = torch.rand(1, slots, generator=generator)

    view = _Collinear()
    view.slots = shared + spread  # type: ignore[attr-defined]

    batch = adapter.prepare_write(intermediates, view)
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


def test_write_probe_reports_pairwise_payload_cosines() -> None:
    shared_key = _unit(torch.ones(768)).reshape(1, 1, 768).repeat(1, 4, 1)
    shared_value = (
        _unit(torch.arange(768, dtype=torch.float32) + 1.0).reshape(1, 1, 768).repeat(1, 4, 1)
    )
    collinear = apply_memory_write(
        fast_state=_zero_state(),
        batch=_batch(keys=shared_key, values=shared_value, etas=torch.full((1, 4), 0.1)),
        row=0,
    )
    assert collinear.did_write
    assert collinear.key_pairwise_cosine_mean == pytest.approx(1.0, abs=1.0e-5)
    assert collinear.value_pairwise_cosine_mean == pytest.approx(1.0, abs=1.0e-5)
    assert collinear.delta_pairwise_cosine_mean == pytest.approx(1.0, abs=1.0e-5)

    basis = torch.eye(768)
    orthogonal = apply_memory_write(
        fast_state=_zero_state(),
        batch=_batch(
            keys=basis[:4].reshape(1, 4, 768),
            values=basis[4:8].reshape(1, 4, 768),
            etas=torch.full((1, 4), 0.1),
        ),
        row=0,
    )
    assert orthogonal.key_pairwise_cosine_mean == pytest.approx(0.0, abs=1.0e-6)
    assert orthogonal.value_pairwise_cosine_mean == pytest.approx(0.0, abs=1.0e-6)
    assert orthogonal.delta_pairwise_cosine_mean == pytest.approx(0.0, abs=1.0e-6)

    antiparallel = apply_memory_write(
        fast_state=_zero_state(),
        batch=_batch(
            keys=torch.stack((basis[0], -basis[0])).reshape(1, 2, 768),
            values=torch.stack((basis[1], -basis[1])).reshape(1, 2, 768),
            etas=torch.full((1, 2), 0.1),
        ),
        row=0,
    )
    assert antiparallel.key_pairwise_cosine_mean == pytest.approx(-1.0, abs=1.0e-6)
    assert antiparallel.value_pairwise_cosine_mean == pytest.approx(-1.0, abs=1.0e-6)

    single = apply_memory_write(
        fast_state=_zero_state(),
        batch=_batch(
            keys=basis[0].reshape(1, 1, 768),
            values=basis[1].reshape(1, 1, 768),
            etas=torch.full((1, 1), 0.1),
        ),
        row=0,
    )
    assert single.did_write
    assert single.key_pairwise_cosine_mean == 0.0
    assert single.value_pairwise_cosine_mean == 0.0
    assert single.delta_pairwise_cosine_mean == 0.0

    skipped = apply_memory_write(
        fast_state=_zero_state(),
        batch=_batch(
            keys=basis[:2].reshape(1, 2, 768),
            values=basis[2:4].reshape(1, 2, 768),
            etas=torch.full((1, 2), 0.1),
            slot_mask=torch.zeros((1, 2), dtype=torch.bool),
        ),
        row=0,
    )
    assert skipped.skip_reason is MemoryWriteSkipReason.NO_VALID_SLOT
    assert skipped.key_pairwise_cosine_mean == 0.0
    assert skipped.value_pairwise_cosine_mean == 0.0
    assert skipped.delta_pairwise_cosine_mean == 0.0


def test_slot_geometry_probe_algebra_separates_shared_and_residual() -> None:
    from ttt_svcbench_qwen.fast_ttt import (
        _centered_offdiag_cosine_mean,
        _offdiag_cosine_mean,
    )

    # A large shared component with tiny orthogonal residuals is the observed
    # regime: the raw cosine pins near 1 while the residual is still healthy.
    basis = torch.eye(768)
    shared = basis[0] * 1.0
    collapsed = torch.stack([shared + 0.01 * basis[index + 1] for index in range(8)])
    raw = _offdiag_cosine_mean(collapsed)
    assert raw > 0.999
    # Centering removes the shared direction exactly, so an equal-norm centered
    # set lands on its algebraic floor -1/(n-1).
    centered = _centered_offdiag_cosine_mean(collapsed)
    assert centered == pytest.approx(-1.0 / 7.0, abs=1.0e-4)

    # A genuinely collapsed set stays collapsed after centering: the residual
    # itself carries nothing, which is the branch where centering cannot help.
    identical = shared.reshape(1, 768).repeat(8, 1)
    assert _offdiag_cosine_mean(identical) == pytest.approx(1.0, abs=1.0e-5)
    assert _centered_offdiag_cosine_mean(identical) == 0.0

    orthogonal = basis[:8]
    assert _offdiag_cosine_mean(orthogonal) == pytest.approx(0.0, abs=1.0e-6)
    assert _offdiag_cosine_mean(basis[:1]) == 0.0


def test_prepare_write_probes_are_plain_floats_and_never_reach_the_graph() -> None:
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

    assert len(batch.slot_geometry_probes) == 1
    probe = batch.slot_geometry_probes[0]
    assert isinstance(probe, SlotGeometryProbe)
    # The structural grad-leak guard: prepare_write runs with grad ENABLED, so a
    # probe that forgot no_grad/detach would carry a Tensor here and attach audit
    # ops to the meta graph.  Plain floats prove it did not.
    for name in PROBE_FIELDS:
        value = getattr(probe, name)
        assert type(value) is float, name
        assert math.isfinite(value), name
        assert -1.0 <= value <= 1.0, name

    result = apply_memory_write(fast_state=state, batch=batch, row=0)
    assert result.slot_geometry is probe
    # Probing must not open a path to the slot states.
    (slot_grad,) = torch.autograd.grad(
        result.fast_state.m.square().sum(),
        (slots,),
        retain_graph=True,
        allow_unused=True,
    )
    assert slot_grad is None


def test_skipped_writes_carry_no_slot_geometry() -> None:
    basis = torch.eye(768)
    skipped = apply_memory_write(
        fast_state=_zero_state(),
        batch=_batch(
            keys=basis[:2].reshape(1, 2, 768),
            values=basis[2:4].reshape(1, 2, 768),
            etas=torch.full((1, 2), 0.1),
            slot_mask=torch.zeros((1, 2), dtype=torch.bool),
        ),
        row=0,
    )
    assert skipped.skip_reason is MemoryWriteSkipReason.NO_VALID_SLOT
    assert skipped.slot_geometry is None


def test_slot_geometry_probe_rejects_out_of_range_and_non_float() -> None:
    with pytest.raises(ValueError, match=r"finite in \[-1, 1\]"):
        SlotGeometryProbe(slot_pairwise=1.5)
    with pytest.raises(ValueError, match=r"finite in \[-1, 1\]"):
        SlotGeometryProbe(token_key_pairwise=math.nan)
    with pytest.raises(TypeError, match="plain floats"):
        SlotGeometryProbe(value_raw_pairwise=0)  # type: ignore[arg-type]


def test_slot_centering_removes_the_shared_component_and_guards_degeneracy() -> None:
    from ttt_svcbench_qwen.fast_ttt import _centered_over_valid_slots, _offdiag_cosine_mean

    torch.manual_seed(0)
    shared = torch.randn(768) * 5.0
    slots = torch.stack([shared + 0.05 * torch.randn(768) for _ in range(6)]).unsqueeze(0)
    full = torch.ones(1, 6, dtype=torch.bool)

    assert _offdiag_cosine_mean(slots[0]) > 0.999
    centered = _centered_over_valid_slots(slots, full)
    # An equal-norm centered set sits on its algebraic floor -1/(n-1).
    assert _offdiag_cosine_mean(centered[0]) == pytest.approx(-1.0 / 5.0, abs=1.0e-3)

    # Each slot's norm must be restored.  Centering alone shrinks the probe input
    # by ~150x, which flattens the attention softmax and puts every slot back on
    # the same pooled token mean -- the mirror image of the failure being fixed.
    assert torch.allclose(
        centered[0].norm(dim=-1), slots[0].norm(dim=-1), rtol=1.0e-4
    )

    # The mean must ignore invalid slots, or their garbage pollutes every row.
    # Compare directions: the magnitude is deliberately rescaled back to the
    # original slot norm, so only the direction carries the centering.
    partial = torch.tensor([[True, True, True, False, False, False]])
    expected = slots[0][:3] - slots[0][:3].mean(dim=0, keepdim=True)
    got = _centered_over_valid_slots(slots, partial)[0][:3]
    assert torch.allclose(
        got / got.norm(dim=-1, keepdim=True),
        expected / expected.norm(dim=-1, keepdim=True),
        atol=1.0e-5,
    )

    # Fewer than two valid slots must pass through: centering one slot yields
    # exactly zero, and a zero-norm valid payload row is rejected outright.
    for degenerate in (
        torch.tensor([[True, False, False, False, False, False]]),
        torch.zeros(1, 6, dtype=torch.bool),
    ):
        assert torch.equal(_centered_over_valid_slots(slots, degenerate), slots)


def test_degenerate_valid_slot_row_fails_closed() -> None:
    """A valid slot whose row is zero must be rejected, not silently absorbed.

    Nothing downstream notices one: the delta is zero, the update is zero, the
    write still reports ``did_write`` with ``write_norm == 0.0``, and the cosine
    audits return ``0.0`` rather than a NaN.  It is invisible capacity loss, and
    subtracting a slot mean is one way to produce it.
    """

    basis = torch.eye(768)
    for name in ("keys", "values"):
        rows = torch.stack((basis[0], basis[1], torch.zeros(768))).reshape(1, 3, 768)
        other = torch.stack((basis[2], basis[3], basis[4])).reshape(1, 3, 768)
        payload = {"keys": rows, "values": other} if name == "keys" else {
            "keys": other,
            "values": rows,
        }
        with pytest.raises(ValueError, match="zero-norm valid slot"):
            _batch(etas=torch.full((1, 3), 0.1), **payload)

    # Marking the degenerate slot invalid is the legitimate way to express it.
    ok = _batch(
        keys=torch.stack((basis[0], basis[1], torch.zeros(768))).reshape(1, 3, 768),
        values=torch.stack((basis[2], basis[3], basis[4])).reshape(1, 3, 768),
        etas=torch.full((1, 3), 0.1),
        slot_mask=torch.tensor([[True, True, False]]),
    )
    assert ok.slot_mask.tolist() == [[True, True, False]]


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
