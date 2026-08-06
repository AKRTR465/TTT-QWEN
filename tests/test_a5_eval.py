from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ttt_svcbench_qwen.a5_eval import A5MemoryGenerationAudit, _write_chunk
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.fast_ttt import build_fast_ttt_adapter


def _audit(**overrides: object) -> A5MemoryGenerationAudit:
    payload: dict[str, object] = {
        "mode": "a5_memory",
        "support_count": 8,
        "observed_chunk_count": 9,
        "memory_writes_attempted": 8,
        "memory_writes_applied": 8,
        "memory_writes_skipped": 0,
        "fast_state_row_count": 1,
        "final_write_version": 8,
        "final_memory_norm": 0.34,
        "reader_status": "disabled",
        "reader_exact_count": None,
        "reader_selected_record_count": 0,
        "lifecycle_observation_count": 9,
        "lifecycle_prefill_count": 1,
        "lifecycle_decode_count": 0,
    }
    payload.update(overrides)
    return A5MemoryGenerationAudit(**payload)  # type: ignore[arg-type]


def test_healthy_a5_episode_audit_is_accepted() -> None:
    audit = _audit()
    assert audit.memory_writes_applied == audit.support_count
    assert audit.final_write_version == audit.memory_writes_applied
    # A skipped write is legitimate (a chunk can yield no valid slot), as long as the write was
    # attempted for every Support chunk and at least one landed.
    partial = _audit(memory_writes_applied=7, memory_writes_skipped=1, final_write_version=7)
    assert partial.memory_writes_applied + partial.memory_writes_skipped == partial.support_count


def test_a5_audit_rejects_an_episode_that_never_wrote() -> None:
    """The one failure that the accuracy number alone cannot reveal.

    If the writes silently never happen, the run produces exactly the A2-static answers while
    being reported as A5, so an A2 number would be published as an A5 result.  Fail closed.
    """

    with pytest.raises(ValueError, match="no applied write"):
        _audit(memory_writes_applied=0, memory_writes_skipped=8, final_write_version=0)


def test_a5_audit_rejects_an_all_zero_final_memory() -> None:
    with pytest.raises(ValueError, match="all-zero final memory"):
        _audit(final_memory_norm=0.0)


def test_a5_audit_rejects_write_accounting_that_does_not_add_up() -> None:
    with pytest.raises(ValueError, match="one write per Support chunk"):
        _audit(memory_writes_attempted=7)
    with pytest.raises(ValueError, match="cover every Support chunk"):
        _audit(memory_writes_applied=6, memory_writes_skipped=0, final_write_version=6)
    with pytest.raises(ValueError, match="final write version"):
        _audit(final_write_version=5)


def test_a5_audit_rejects_a2_static_shaped_payloads() -> None:
    with pytest.raises(ValueError, match="wrong mode"):
        _audit(mode="a2_static")
    with pytest.raises(ValueError, match="one per-video memory row"):
        _audit(fast_state_row_count=0)
    with pytest.raises(ValueError, match="at least one Support chunk"):
        _audit(
            support_count=0,
            memory_writes_attempted=0,
            memory_writes_applied=0,
            final_write_version=0,
        )


def _observation(intermediates: object, spatial: object) -> SimpleNamespace:
    return SimpleNamespace(
        soft_intermediates=SimpleNamespace(fast_associative=intermediates, spatial=spatial)
    )


def test_write_chunk_moves_a_zero_memory_and_advances_its_version() -> None:
    """The core of the evaluator: one Support chunk must actually change M.

    Built against a real FastTTTAdapter rather than a stub, because the whole point is that the
    production write path runs -- a synthetic double could report success while writing nothing.
    """

    torch.manual_seed(11)
    adapter = build_fast_ttt_adapter(load_config())
    slots = load_config().spatial_encoder.active_slots
    state = adapter.initialize_fast_state(differentiable=False)
    assert state.write_version == 0
    assert not state.m.any()

    adapter(torch.randn(1, 6, 4096), fast_state=(state,))
    intermediates = adapter.consume_associative_intermediates()
    spatial = SimpleNamespace(
        slots=torch.randn(1, slots, 768),
        slot_valid_mask=torch.ones(1, slots, dtype=torch.bool),
        slot_confidence=torch.rand(1, slots),
    )

    written, applied = _write_chunk(_observation(intermediates, spatial), state, adapter)

    assert applied is True
    assert written.write_version == 1
    assert written.m.any(), "a successful write must leave a non-zero memory"
    assert float(torch.linalg.matrix_norm(written.m.detach())) > 0.0
    # The evaluator threads the returned state forward; the input must not be mutated in place.
    assert not state.m.any()
    assert state.write_version == 0


def test_write_chunk_fails_closed_on_a_missing_or_differentiable_payload() -> None:
    torch.manual_seed(11)
    adapter = build_fast_ttt_adapter(load_config())
    slots = load_config().spatial_encoder.active_slots
    state = adapter.initialize_fast_state(differentiable=False)
    adapter(torch.randn(1, 6, 4096), fast_state=(state,))
    intermediates = adapter.consume_associative_intermediates()
    spatial = SimpleNamespace(
        slots=torch.randn(1, slots, 768),
        slot_valid_mask=torch.ones(1, slots, dtype=torch.bool),
        slot_confidence=torch.rand(1, slots),
    )

    with pytest.raises(ValueError, match="associative write tensors"):
        _write_chunk(_observation(None, spatial), state, adapter)
    with pytest.raises(ValueError, match="spatial slot state"):
        _write_chunk(_observation(intermediates, None), state, adapter)
    # A differentiable state would retain the meta graph across the whole evaluation.
    meta_state = adapter.initialize_fast_state(differentiable=True)
    with pytest.raises(ValueError, match="non-differentiable online leaf"):
        _write_chunk(_observation(intermediates, spatial), meta_state, adapter)
