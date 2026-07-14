from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.fast_ttt import build_fast_ttt_adapter
from ttt_svcbench_qwen.functional_sgd import (
    audit_gradient_delta_group,
    functional_sgd_step,
    initialize_optimizer_state,
    snapshot_gradient_delta_group,
)
from ttt_svcbench_qwen.losses import O1StateTarget, StateLossInput, compute_state_loss
from ttt_svcbench_qwen.observation_heads import O1CurrentCountDecoder
from ttt_svcbench_qwen.state_bank import SemanticProjector


def _storage_pointer(value: Tensor) -> int:
    return int(value.untyped_storage().data_ptr())


def test_actual_fast_bridge_observation_chain_has_exact_inner_update_boundary() -> None:
    torch.manual_seed(1413)
    config = load_config()
    optimizer_config = config.fast_ttt.optimizer
    adapter = build_fast_ttt_adapter(config).eval()
    state = adapter.initialize_fast_state()
    bridge = nn.Linear(4096, 768, bias=False).eval().requires_grad_(False)
    observation_head = (
        O1CurrentCountDecoder(config.observation_heads.o1).eval().requires_grad_(False)
    )
    semantic_projector = SemanticProjector(config.state_bank.semantic_projector)
    semantic_projector.set_online_frozen()

    groups = {
        "online_fast": (
            snapshot_gradient_delta_group(
                name="online_fast",
                parameters=state.fast_parameters,
                gradient_expected=True,
                update_allowed=True,
            ),
            state.fast_parameters,
        ),
        "adapter_checkpointed": (
            snapshot_gradient_delta_group(
                name="adapter_checkpointed",
                parameters=adapter.parameters(),
                gradient_expected=False,
                update_allowed=False,
            ),
            tuple(adapter.parameters()),
        ),
        "frozen_bridge": (
            snapshot_gradient_delta_group(
                name="frozen_bridge",
                parameters=bridge.parameters(),
                gradient_expected=False,
                update_allowed=False,
            ),
            tuple(bridge.parameters()),
        ),
        "observation_o1": (
            snapshot_gradient_delta_group(
                name="observation_o1",
                parameters=observation_head.parameters(),
                gradient_expected=False,
                update_allowed=False,
            ),
            tuple(observation_head.parameters()),
        ),
        "state_bank.semantic_projector": (
            snapshot_gradient_delta_group(
                name="state_bank.semantic_projector",
                parameters=semantic_projector.parameters(),
                gradient_expected=False,
                update_allowed=False,
            ),
            tuple(semantic_projector.parameters()),
        ),
        "hard_bank_fsm": (
            snapshot_gradient_delta_group(
                name="hard_bank_fsm",
                parameters=(),
                gradient_expected=False,
                update_allowed=False,
            ),
            (),
        ),
        "excluded_query_reader_llm": (
            snapshot_gradient_delta_group(
                name="excluded_query_reader_llm",
                parameters=(),
                gradient_expected=False,
                update_allowed=False,
            ),
            (),
        ),
    }

    visual = torch.randn(1, 2, 4096)
    valid_mask = torch.tensor([[True, True]])
    q_target = torch.randn(1, 512)
    timestamps = torch.tensor([0.25], dtype=torch.float64)
    position_ids = torch.tensor([1], dtype=torch.int64)
    with adapter.use_fast_state(state):
        adapted = adapter(visual, valid_mask)
    slots = bridge(adapted)
    observation = observation_head(
        slots,
        valid_mask,
        q_target,
        timestamps,
        position_ids,
    )
    state_loss = compute_state_loss(
        StateLossInput(
            batch_size=1,
            o1=O1StateTarget(
                row_indices=torch.tensor([0]),
                logits=observation.logits,
                targets=torch.zeros_like(observation.logits),
                slot_mask=valid_mask,
            ),
        )
    )
    loss = state_loss.total * 1_000_000.0
    fast_gradients = torch.autograd.grad(
        loss,
        state.fast_parameters,
        retain_graph=True,
    )

    result = functional_sgd_step(
        loss=loss,
        fast_state=state,
        optimizer_config=optimizer_config,
        optimizer_state=initialize_optimizer_state(optimizer_config),
        valid_term_count=1,
    )
    assert result.did_update is True

    audits = {
        name: audit_gradient_delta_group(
            snapshot,
            parameters=(result.fast_state.fast_parameters if name == "online_fast" else parameters),
            gradients=fast_gradients if name == "online_fast" else None,
        )
        for name, (snapshot, parameters) in groups.items()
    }

    fast = audits["online_fast"]
    assert fast.parameter_count == 2 * 768 * 768 == 1_179_648
    assert fast.gradient_present is True
    assert fast.gradient_norm > 0.0
    assert fast.delta_norm > 0.0
    assert fast.gradient_expected is True
    assert fast.update_allowed is True
    assert all(
        _storage_pointer(before) != _storage_pointer(after)
        for before, after in zip(
            state.fast_parameters,
            result.fast_state.fast_parameters,
            strict=True,
        )
    )

    assert audits["adapter_checkpointed"].parameter_count == 7_480_064
    assert audits["frozen_bridge"].parameter_count == 3_145_728
    assert audits["observation_o1"].parameter_count == 2_632_710
    assert audits["state_bank.semantic_projector"].parameter_count == 1_316_864
    for name in (
        "adapter_checkpointed",
        "frozen_bridge",
        "observation_o1",
        "state_bank.semantic_projector",
        "hard_bank_fsm",
        "excluded_query_reader_llm",
    ):
        audit = audits[name]
        assert audit.gradient_present is False
        assert audit.gradient_norm == 0.0
        assert audit.delta_norm == 0.0
        assert audit.gradient_expected is False
        assert audit.update_allowed is False
    assert audits["hard_bank_fsm"].parameter_count == 0
    assert audits["excluded_query_reader_llm"].parameter_count == 0


def test_gradient_delta_audit_fails_closed_on_boundary_violations() -> None:
    parameter = torch.tensor([1.0], requires_grad=True)

    expected = snapshot_gradient_delta_group(
        name="expected",
        parameters=(parameter,),
        gradient_expected=True,
        update_allowed=False,
    )
    with pytest.raises(ValueError, match="expected gradient is missing"):
        audit_gradient_delta_group(
            expected,
            parameters=(parameter,),
            gradients=(None,),
        )

    forbidden = snapshot_gradient_delta_group(
        name="forbidden",
        parameters=(parameter,),
        gradient_expected=False,
        update_allowed=False,
    )
    with pytest.raises(ValueError, match="forbidden gradient appeared"):
        audit_gradient_delta_group(
            forbidden,
            parameters=(parameter,),
            gradients=(torch.zeros_like(parameter),),
        )
    with pytest.raises(ValueError, match="forbidden parameter delta"):
        audit_gradient_delta_group(
            forbidden,
            parameters=(parameter.detach().clone() + 1.0,),
            gradients=(None,),
        )

    allowed = snapshot_gradient_delta_group(
        name="allowed",
        parameters=(parameter,),
        gradient_expected=False,
        update_allowed=True,
    )
    with pytest.raises(ValueError, match="allowed update did not occur"):
        audit_gradient_delta_group(
            allowed,
            parameters=(parameter,),
            gradients=(None,),
        )

    with pytest.raises(ValueError, match="gradients must be finite"):
        audit_gradient_delta_group(
            forbidden,
            parameters=(parameter,),
            gradients=(torch.tensor([float("nan")]),),
        )
    with pytest.raises(ValueError, match="current parameters must be finite"):
        audit_gradient_delta_group(
            forbidden,
            parameters=(torch.tensor([float("inf")]),),
            gradients=(None,),
        )
