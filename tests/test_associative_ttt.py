from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ttt_svcbench_qwen.associative_ttt import (
    ASSOCIATIVE_CONTRACT,
    ASSOCIATIVE_CONTRACT_VERSION,
    AssociativeTTTIntermediates,
    build_fast_associative_context,
)
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.fast_ttt import build_fast_ttt_adapter


def _view(embeddings: torch.Tensor, present: torch.Tensor, valid: torch.Tensor) -> object:
    return SimpleNamespace(
        embeddings=embeddings,
        present_mask=present,
        record_valid_mask=valid,
        bank_versions=tuple(range(embeddings.shape[0])),
        hard_payload=("ignored", 99, 123.5),
    )


def test_contract_identity_is_the_slot_memory_v3() -> None:
    config = load_config()
    assert ASSOCIATIVE_CONTRACT == "bank_conditioned_slot_memory_v3"
    assert ASSOCIATIVE_CONTRACT_VERSION == 3
    assert config.associative_ttt.contract == ASSOCIATIVE_CONTRACT
    assert config.associative_ttt.key_dim == 768
    assert config.associative_ttt.bank_embedding_dim == 512


def test_parameter_free_bank_pooling_uses_only_present_valid_semantics() -> None:
    query = torch.zeros(2, 512)
    records = torch.zeros(2, 3, 512)
    records[0, 0, 0] = 1.0
    records[0, 1, 1] = 1.0
    records[0, 2, 2] = 100.0
    present = torch.tensor([[True, True, True], [False, False, False]])
    valid = torch.tensor([[True, True, False], [False, False, False]])

    context = build_fast_associative_context(query, _view(records, present, valid))

    assert context.bank_record_counts.tolist() == [2, 0]
    assert context.combined_query[0, :3].tolist() == pytest.approx([0.5, 0.5, 0.0])
    assert torch.equal(context.combined_query[1], query[1])
    changed = records.clone()
    changed[0, 0, 0] = 0.0
    changed[0, 0, 3] = 1.0
    changed_context = build_fast_associative_context(query, _view(changed, present, valid))
    assert not torch.equal(context.combined_query, changed_context.combined_query)


def test_adapter_key_is_bank_conditioned_and_write_targets_are_detached() -> None:
    config = load_config()
    adapter = build_fast_ttt_adapter(config)
    with torch.no_grad():
        adapter.p_context.weight.fill_(0.01)
    visual = torch.randn(1, 2, 4096, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    first = build_fast_associative_context(
        torch.zeros(1, 512),
        _view(
            torch.zeros(1, 0, 512),
            torch.zeros(1, 0, dtype=torch.bool),
            torch.zeros(1, 0, dtype=torch.bool),
        ),
    )
    second_query = torch.arange(512, dtype=torch.float32).unsqueeze(0)
    second = build_fast_associative_context(
        second_query,
        _view(
            torch.zeros(1, 0, 512),
            torch.zeros(1, 0, dtype=torch.bool),
            torch.zeros(1, 0, dtype=torch.bool),
        ),
    )
    with adapter.use_associative_context(first):
        adapter(visual, mask)
    first_items = adapter.consume_associative_intermediates()
    with adapter.use_associative_context(second):
        adapter(visual, mask)
    second_items = adapter.consume_associative_intermediates()
    assert not torch.equal(first_items.keys, second_items.keys)

    slots = torch.randn(1, 4, 768, requires_grad=True)

    class _Slots:
        slot_valid_mask = torch.ones(1, 4, dtype=torch.bool)
        slot_confidence = torch.rand(1, 4)

    view = _Slots()
    view.slots = slots  # type: ignore[attr-defined]
    batch = adapter.prepare_write(second_items, view)
    # Write keys stay differentiable through the bank-conditioned token keys...
    key_grads = torch.autograd.grad(
        batch.keys.sum(),
        (adapter.p_context.weight, adapter.p_in.weight),
        retain_graph=True,
    )
    assert all(
        torch.isfinite(value).all() and torch.count_nonzero(value) > 0
        for value in key_grads
    )
    # ...while slot states feed the write only through stop-gradient.
    slot_grad = torch.autograd.grad(
        batch.keys.sum() + batch.values.sum() + batch.etas.sum(),
        slots,
        allow_unused=True,
    )[0]
    assert slot_grad is None


def test_intermediates_contract_rejects_shape_and_alignment_drift() -> None:
    keys = torch.zeros(1, 2, 768)
    with pytest.raises(ValueError, match="must align"):
        AssociativeTTTIntermediates(
            keys=keys,
            predictions=torch.zeros(1, 3, 768),
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            bank_record_counts=torch.zeros(1, dtype=torch.int64),
            bank_versions=(0,),
        )
    with pytest.raises(ValueError, match="bank_versions"):
        AssociativeTTTIntermediates(
            keys=keys,
            predictions=keys.clone(),
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            bank_record_counts=torch.zeros(1, dtype=torch.int64),
            bank_versions=(0, 1),
        )
