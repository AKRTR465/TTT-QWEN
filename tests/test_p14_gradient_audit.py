from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.fast_ttt import build_fast_ttt_adapter
from ttt_svcbench_qwen.llamafactory_trainer import (
    _A5ParameterGroupStepAuditor,
    _SemanticProjectorStepAuditor,
)
from ttt_svcbench_qwen.losses import O1StateTarget, StateLossInput, compute_state_loss
from ttt_svcbench_qwen.memory_write import apply_memory_write
from ttt_svcbench_qwen.observation_heads import O1CurrentCountDecoder
from ttt_svcbench_qwen.outer_gradient_control import GroupGradientAudit, OuterGradientAudit
from ttt_svcbench_qwen.state_bank import SemanticProjector


def _storage_pointer(value: Tensor) -> int:
    return int(value.untyped_storage().data_ptr())


def test_semantic_projector_training_log_uses_pre_step_group_and_real_delta() -> None:
    projector = SemanticProjector(load_config().state_bank.semantic_projector)
    wrapper = nn.Module()
    wrapper.add_module("semantic_projector", projector)
    parameters = tuple(projector.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.SGD(
        [{"params": parameters, "lr": 1.0e-3, "group_name": "state_retrieval"}]
    )
    sum(parameter.float().sum() for parameter in parameters).backward()
    pre_norm = math.sqrt(parameter_count)
    audit = OuterGradientAudit(
        attempted_update_count=1,
        successful_update_count=1,
        skipped_update_count=0,
        within_initial_audit_window=True,
        skipped_nonfinite=False,
        skipped_nonfinite_loss=False,
        nonfinite_loss_sources=(),
        groups=(
            GroupGradientAudit(
                name="state_retrieval",
                learning_rate=1.0e-3,
                max_norm=0.05,
                pre_clip_norm=pre_norm,
                post_clip_norm=0.05,
                clip_coefficient=0.05 / pre_norm,
                rms=1.0,
                max_abs=1.0,
                active_elements=parameter_count,
                nonfinite_elements=0,
            ),
        ),
    )
    auditor = _SemanticProjectorStepAuditor(wrapper)
    snapshot = auditor.before_step(optimizer, audit)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    auditor.after_step(snapshot, audit)

    assert auditor.last_metrics["grad/semantic_projector/pre_clip_norm"] == pytest.approx(pre_norm)
    assert auditor.last_metrics["grad/semantic_projector/parameter_delta_l2"] > 0.0
    assert auditor.last_metrics["grad/semantic_projector/parameter_delta_nonzero"] == 1.0

    wrong = torch.optim.SGD(
        [{"params": parameters[:-1], "lr": 1.0e-3, "group_name": "state_retrieval"}]
    )
    with pytest.raises(RuntimeError, match="must equal"):
        _SemanticProjectorStepAuditor(wrapper).before_step(wrong, audit)


def test_a5_parameter_group_audit_measures_real_post_optimizer_delta() -> None:
    model = nn.Module()
    model.add_module("p_context", nn.Linear(3, 2))
    meta_fast = nn.Module()
    meta_fast.register_parameter("w0_1", nn.Parameter(torch.ones(2, 2)))
    meta_fast.register_parameter("w0_2", nn.Parameter(torch.ones(2, 2)))
    model.add_module("meta_fast", meta_fast)
    model.add_module("temporal_encoder", nn.Linear(3, 3))
    audit = OuterGradientAudit(
        attempted_update_count=1,
        successful_update_count=1,
        skipped_update_count=0,
        within_initial_audit_window=True,
        skipped_nonfinite=False,
        skipped_nonfinite_loss=False,
        nonfinite_loss_sources=(),
        groups=(),
    )
    auditor = _A5ParameterGroupStepAuditor(model, delta_audit_steps=2)
    snapshot = auditor.before_step(audit)
    assert snapshot is not None
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.25)
    auditor.after_step(snapshot, audit)

    for group in ("associative", "w0", "state_shared"):
        prefix = f"a5/parameter_delta/{group}"
        assert auditor.last_metrics[f"{prefix}/l2"] > 0.0
        assert auditor.last_metrics[f"{prefix}/rms"] == pytest.approx(0.25)
        assert auditor.last_metrics[f"{prefix}/nonzero_fraction"] == 1.0


def test_actual_fast_bridge_observation_chain_has_exact_write_boundary() -> None:
    """The P14 boundary contract: the memory write publishes a new online tensor,
    the read path gives online gradients only to the memory, and every
    checkpointed module stays bitwise untouched."""

    torch.manual_seed(1413)
    config = load_config()
    adapter = build_fast_ttt_adapter(config).eval()
    state = adapter.initialize_fast_state()
    bridge = nn.Linear(4096, 768, bias=False).eval().requires_grad_(False)
    observation_head = (
        O1CurrentCountDecoder(config.observation_heads.o1).eval().requires_grad_(False)
    )
    semantic_projector = SemanticProjector(config.state_bank.semantic_projector)
    semantic_projector.set_online_frozen()

    frozen_groups = {
        "adapter_checkpointed": tuple(adapter.parameters()),
        "frozen_bridge": tuple(bridge.parameters()),
        "observation_o1": tuple(observation_head.parameters()),
        "state_bank.semantic_projector": tuple(semantic_projector.parameters()),
    }
    before_values = {
        name: tuple(parameter.detach().clone() for parameter in parameters)
        for name, parameters in frozen_groups.items()
    }

    visual = torch.randn(1, 2, 4096)
    valid_mask = torch.tensor([[True, True]])
    q_target = torch.randn(1, 512)
    timestamps = torch.tensor([0.25], dtype=torch.float64)
    position_ids = torch.tensor([1], dtype=torch.int64)
    with adapter.use_fast_state(state):
        adapted = adapter(visual, valid_mask)
    intermediates = adapter.consume_associative_intermediates()
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
    state_loss.total.backward()
    memory_gradient = state.m.grad
    assert memory_gradient is not None
    assert torch.isfinite(memory_gradient).all()
    assert memory_gradient.abs().sum() > 0
    state.m.grad = None

    slot_view = SimpleNamespace(
        slots=slots.detach()[:, :2, :],
        slot_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        slot_confidence=torch.ones(1, 2),
    )
    with torch.no_grad():
        batch = adapter.prepare_write(intermediates, slot_view)
    result = apply_memory_write(fast_state=state, batch=batch, row=0)
    assert result.did_write is True
    assert result.fast_state.write_version == 1
    assert _storage_pointer(result.fast_state.m) != _storage_pointer(state.m)
    assert result.fast_state.m.is_leaf and result.fast_state.m.requires_grad

    for name, parameters in frozen_groups.items():
        for parameter, before in zip(parameters, before_values[name], strict=True):
            assert parameter.grad is None, name
            assert torch.equal(parameter.detach(), before), name
