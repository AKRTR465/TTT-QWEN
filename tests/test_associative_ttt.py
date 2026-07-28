from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ttt_svcbench_qwen.associative_ttt import (
    ASSOCIATIVE_CONTRACT_VERSION,
    AssociativeTTTIntermediates,
    build_fast_associative_context,
    compute_associative_ttt_loss,
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


def test_adapter_key_is_bank_conditioned_and_value_target_stops_raw_visual_gradient() -> None:
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

    loss = compute_associative_ttt_loss(second_items)
    fast_grads = torch.autograd.grad(
        loss.total,
        (adapter.w0_1, adapter.w0_2, adapter.p_value.weight),
        retain_graph=True,
    )
    assert all(
        torch.isfinite(value).all() and torch.count_nonzero(value) > 0
        for value in fast_grads
    )
    value_only = second_items.values.square().mean()
    value_visual_grad = torch.autograd.grad(value_only, visual, allow_unused=True)[0]
    assert value_visual_grad is None


def test_masked_fp32_mse_and_zero_token_skip_are_exact() -> None:
    keys = torch.zeros(2, 3, 768, dtype=torch.float16)
    values = torch.zeros_like(keys)
    predictions = torch.ones_like(keys)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    output = compute_associative_ttt_loss(
        AssociativeTTTIntermediates(
            keys=keys,
            values=values,
            predictions=predictions,
            valid_mask=mask,
            bank_record_counts=torch.tensor([1, 0]),
            bank_versions=(3, 0),
        )
    )
    assert output.total.dtype == torch.float32
    assert output.total.item() == pytest.approx(1.0)
    assert output.per_row_total.tolist() == pytest.approx([1.0, 0.0])
    assert output.valid_token_counts.tolist() == [2, 0]
    assert output.update_valid_mask.tolist() == [True, False]


def test_associative_contract_version_is_persistent_and_strict() -> None:
    adapter = build_fast_ttt_adapter(load_config())
    state = adapter.state_dict()
    assert int(state["associative_contract_version"].item()) == ASSOCIATIVE_CONTRACT_VERSION
    legacy = dict(state)
    legacy.pop("associative_contract_version")
    with pytest.raises(RuntimeError, match="Missing key"):
        adapter.load_state_dict(legacy, strict=True)
