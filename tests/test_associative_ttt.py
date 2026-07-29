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
from ttt_svcbench_qwen.stage_a_runtime import StageASoftWriteOutput
from ttt_svcbench_qwen.state_bank import HeadType


def _view(embeddings: torch.Tensor, present: torch.Tensor, valid: torch.Tensor) -> object:
    return SimpleNamespace(
        embeddings=embeddings,
        present_mask=present,
        record_valid_mask=valid,
        bank_versions=tuple(range(embeddings.shape[0])),
        hard_payload=("ignored", 99, 123.5),
    )


def _soft_write(
    *,
    batch_size: int,
    source_requires_grad: bool = False,
) -> StageASoftWriteOutput:
    source = torch.arange(
        batch_size * 768,
        dtype=torch.float32,
    ).reshape(batch_size, 768)
    source = (source + 1.0).requires_grad_(source_requires_grad)
    o2 = torch.stack((source + 1.0, source + 2.0), dim=1)
    present = torch.ones(batch_size, dtype=torch.bool)
    slot_present = torch.ones(batch_size, 2, dtype=torch.bool)
    return StageASoftWriteOutput(
        o1_semantics=torch.zeros(batch_size, 512),
        o1_present_mask=present,
        o2_semantics=torch.zeros(batch_size, 2, 512),
        o2_present_mask=slot_present,
        e1_semantics=torch.zeros(batch_size, 2, 512),
        e1_present_mask=slot_present,
        e2_semantics=torch.zeros(batch_size, 2, 512),
        e2_present_mask=slot_present,
        o1_sources=source,
        o2_sources=o2,
        e1_sources=source + 3.0,
        e2_sources=source + 4.0,
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


def test_adapter_key_is_bank_conditioned_and_state_write_target_is_detached() -> None:
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

    state_write = _soft_write(batch_size=1, source_requires_grad=True)
    loss = compute_associative_ttt_loss(
        second_items,
        state_write,
        (HeadType.O1,),
    )
    fast_grads = torch.autograd.grad(
        loss.total,
        (adapter.w0_1, adapter.w0_2),
        retain_graph=True,
    )
    assert all(
        torch.isfinite(value).all() and torch.count_nonzero(value) > 0
        for value in fast_grads
    )
    source_grad = torch.autograd.grad(
        loss.total,
        state_write.o1_sources,
        allow_unused=True,
    )[0]
    assert source_grad is None


def test_normalized_fp32_cosine_and_empty_target_skip_are_exact() -> None:
    keys = torch.zeros(2, 3, 768, dtype=torch.float16)
    predictions = torch.ones_like(keys)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    state_write = _soft_write(batch_size=2)
    output = compute_associative_ttt_loss(
        AssociativeTTTIntermediates(
            keys=keys,
            predictions=predictions,
            valid_mask=mask,
            bank_record_counts=torch.tensor([1, 0]),
            bank_versions=(3, 0),
        ),
        state_write,
        (HeadType.O1, None),
    )
    assert output.total.dtype == torch.float32
    target = state_write.o1_sources[0].float()
    normalized_target = target * torch.rsqrt(target.square().sum() + 1.0e-12)
    prediction = torch.ones_like(target)
    normalized_prediction = prediction * torch.rsqrt(
        prediction.square().sum() + 1.0e-12
    )
    expected = 1.0 - float((normalized_target * normalized_prediction).sum().item())
    assert output.total.item() == pytest.approx(expected)
    assert output.per_row_total.tolist() == pytest.approx([expected, 0.0])
    assert output.valid_token_counts.tolist() == [2, 0]
    assert output.update_valid_mask.tolist() == [True, False]
    assert output.target_audit.unsupported_count == 1
    assert output.target_audit.prediction_target_cosine_count.item() == 1


def test_all_active_heads_and_o2_masked_mean_select_the_predicted_source() -> None:
    batch_size = 4
    state_write = _soft_write(batch_size=batch_size)
    o2_mask = state_write.o2_present_mask.clone()
    o2_mask[1, 1] = False
    state_write = StageASoftWriteOutput(
        **{
            **{
                name: getattr(state_write, name)
                for name in (
                    "o1_semantics",
                    "o1_present_mask",
                    "o2_semantics",
                    "e1_semantics",
                    "e1_present_mask",
                    "e2_semantics",
                    "e2_present_mask",
                    "o1_sources",
                    "o2_sources",
                    "e1_sources",
                    "e2_sources",
                )
            },
            "o2_present_mask": o2_mask,
        }
    )
    selected = torch.stack(
        (
            state_write.o1_sources[0],
            state_write.o2_sources[1, 0],
            state_write.e1_sources[2],
            state_write.e2_sources[3],
        )
    )
    predictions = selected[:, None, :].repeat(1, 2, 1).requires_grad_(True)
    output = compute_associative_ttt_loss(
        AssociativeTTTIntermediates(
            keys=torch.zeros_like(predictions),
            predictions=predictions,
            valid_mask=torch.ones(batch_size, 2, dtype=torch.bool),
            bank_record_counts=torch.zeros(batch_size, dtype=torch.int64),
            bank_versions=(0, 0, 0, 0),
        ),
        state_write,
        (HeadType.O1, HeadType.O2, HeadType.E1, HeadType.E2),
    )
    assert output.total.item() == pytest.approx(0.0, abs=1.0e-6)
    assert output.target_audit.active_head_counts == (1, 1, 1, 1)
    assert output.target_audit.valid_target_counts == (1, 1, 1, 1)


def test_missing_active_head_source_skips_without_fabricating_zero_target() -> None:
    state_write = _soft_write(batch_size=1)
    empty = torch.zeros_like(state_write.e1_present_mask)
    state_write = StageASoftWriteOutput(
        o1_semantics=state_write.o1_semantics,
        o1_present_mask=state_write.o1_present_mask,
        o2_semantics=state_write.o2_semantics,
        o2_present_mask=state_write.o2_present_mask,
        e1_semantics=state_write.e1_semantics,
        e1_present_mask=empty,
        e2_semantics=state_write.e2_semantics,
        e2_present_mask=state_write.e2_present_mask,
        o1_sources=state_write.o1_sources,
        o2_sources=state_write.o2_sources,
        e1_sources=state_write.e1_sources,
        e2_sources=state_write.e2_sources,
    )
    output = compute_associative_ttt_loss(
        AssociativeTTTIntermediates(
            keys=torch.randn(1, 2, 768),
            predictions=torch.randn(1, 2, 768, requires_grad=True),
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            bank_record_counts=torch.zeros(1, dtype=torch.int64),
            bank_versions=(0,),
        ),
        state_write,
        (HeadType.E1,),
    )
    assert output.total.item() == 0.0
    assert output.update_valid_mask.tolist() == [False]
    assert output.valid_token_counts.tolist() == [0]
    assert output.target_audit.empty_target_count == 1


def test_associative_contract_version_is_persistent_and_strict() -> None:
    adapter = build_fast_ttt_adapter(load_config())
    state = adapter.state_dict()
    assert int(state["associative_contract_version"].item()) == ASSOCIATIVE_CONTRACT_VERSION
    legacy = dict(state)
    legacy.pop("associative_contract_version")
    with pytest.raises(RuntimeError, match="Missing key"):
        adapter.load_state_dict(legacy, strict=True)
