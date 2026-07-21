from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.outer_gradient_control import OuterGradientController


def _parameter(value: float, gradient: tuple[float, ...]) -> nn.Parameter:
    parameter = nn.Parameter(torch.full((len(gradient),), value))
    parameter.grad = torch.tensor(gradient)
    return parameter


def _optimizer(groups: tuple[tuple[str, float, nn.Parameter], ...]) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        [
            {"params": [parameter], "lr": learning_rate, "group_name": name}
            for name, learning_rate, parameter in groups
        ]
    )


def test_plain_group_clipping_isolates_task_from_retrieval() -> None:
    qwen = _parameter(1.0, (0.6, 0.8))
    shared = _parameter(1.0, (0.01, 0.02))
    task = _parameter(1.0, (300.0, 400.0))
    router_time = _parameter(1.0, (0.01, 0.02))
    retrieval = _parameter(1.0, (0.03, 0.04))
    w0 = _parameter(1.0, (0.03, 0.04))
    optimizer = _optimizer(
        (
            ("qwen", 1.0e-5, qwen),
            ("state_shared", 1.0e-4, shared),
            ("state_task", 1.0e-4, task),
            ("state_router_time", 1.0e-4, router_time),
            ("state_retrieval", 1.0e-4, retrieval),
            ("w0", 1.0e-4, w0),
        )
    )
    controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=(
            "qwen",
            "state_shared",
            "state_task",
            "state_router_time",
            "state_retrieval",
            "w0",
        ),
    )
    qwen_before = qwen.grad.detach().clone()
    retrieval_before = retrieval.grad.detach().clone()
    w0_before = w0.grad.detach().clone()

    should_step, audit = controller.apply_plain(optimizer)

    assert should_step
    assert torch.equal(qwen.grad, qwen_before)
    assert torch.equal(retrieval.grad, retrieval_before)
    assert torch.equal(w0.grad, w0_before)
    assert task.grad is not None
    assert float(task.grad.norm()) == pytest.approx(0.05)
    by_name = {group.name: group for group in audit.groups}
    assert by_name["qwen"].clip_coefficient == 1.0
    assert by_name["state_task"].pre_clip_norm == pytest.approx(500.0)
    assert by_name["state_task"].post_clip_norm == pytest.approx(0.05)
    assert by_name["state_task"].clip_coefficient == pytest.approx(0.0001)
    assert by_name["state_retrieval"].clip_coefficient == 1.0
    assert by_name["w0"].clip_coefficient == 1.0
    assert audit.successful_update_count == 1


def test_nonfinite_gradient_skips_the_complete_plain_update() -> None:
    qwen = _parameter(1.0, (1.0,))
    state = _parameter(1.0, (float("nan"),))
    w0 = _parameter(1.0, (1.0,))
    optimizer = _optimizer(
        (("qwen", 1.0e-5, qwen), ("state_task", 1.0e-4, state), ("w0", 1.0e-4, w0))
    )
    controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen", "state_task", "w0"),
    )

    should_step, audit = controller.apply_plain(optimizer)

    assert not should_step
    assert audit.skipped_nonfinite
    assert audit.successful_update_count == 0
    assert audit.skipped_update_count == 1
    assert all(parameter.grad is None for parameter in (qwen, state, w0))


class _FakeZero:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.averaged_gradients = {
            index: [parameter.grad for parameter in group["params"]]
            for index, group in enumerate(optimizer.param_groups)
        }
        self.params_in_partition = [group["params"] for group in optimizer.param_groups]
        self.real_dp_process_group = [None for _ in optimizer.param_groups]
        self.loss_scale = 1.0
        self.partition_gradients = True
        self.clip_grad = 0.0

    @staticmethod
    def get_grad_norm_direct(gradients: list[torch.Tensor], _params: object) -> torch.Tensor:
        values = torch.stack([gradient.double().square().sum() for gradient in gradients])
        return values.sum().sqrt()

    def has_overflow(self, *, partition_gradients: bool) -> bool:
        assert partition_gradients
        return any(
            not bool(torch.isfinite(gradient).all())
            for gradients in self.averaged_gradients.values()
            for gradient in gradients
        )


def test_zero_partition_groups_match_plain_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ttt_svcbench_qwen.outer_gradient_control.version", lambda _name: "0.18.8")
    parameters = (
        _parameter(1.0, (3.0, 4.0)),
        _parameter(1.0, (30.0, 40.0)),
        _parameter(1.0, (30.0, 40.0)),
        _parameter(1.0, (30.0, 40.0)),
        _parameter(1.0, (30.0, 40.0)),
        _parameter(1.0, (0.06, 0.08)),
        _parameter(1.0, (60.0, 80.0)),
    )
    optimizer = _optimizer(
        (
            ("qwen", 1.0e-5, parameters[0]),
            ("state_shared", 1.0e-4, parameters[1]),
            ("state_task", 1.0e-4, parameters[2]),
            ("state_router_time", 1.0e-4, parameters[3]),
            ("state_retrieval", 1.0e-4, parameters[4]),
            ("w0", 1.0e-4, parameters[5]),
            ("predictor", 1.0e-4, parameters[6]),
        )
    )
    controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=(
            "qwen",
            "state_shared",
            "state_task",
            "state_router_time",
            "state_retrieval",
            "w0",
            "predictor",
        ),
    )

    audit = controller.apply_deepspeed(_FakeZero(optimizer))

    assert [float(parameter.grad.norm()) for parameter in parameters] == pytest.approx(
        [1.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1]
    )
    assert tuple(group.name for group in audit.groups) == (
        "qwen",
        "state_shared",
        "state_task",
        "state_router_time",
        "state_retrieval",
        "w0",
        "predictor",
    )
    assert dict(audit.metrics())["outer_grad/predictor/lr_x_post_norm"] == pytest.approx(1.0e-5)


def test_group_order_and_global_clip_contract_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ttt_svcbench_qwen.outer_gradient_control.version", lambda _name: "0.18.8")
    optimizer = _optimizer(
        (
            ("state_shared", 1.0e-4, _parameter(1.0, (1.0,))),
            ("qwen", 1.0e-5, _parameter(1.0, (1.0,))),
            ("w0", 1.0e-4, _parameter(1.0, (1.0,))),
        )
    )
    controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen", "state_shared", "w0"),
    )
    zero = _FakeZero(optimizer)
    with pytest.raises(ValueError, match="must be"):
        controller.apply_deepspeed(zero)

    zero.optimizer.param_groups[0]["group_name"] = "qwen"
    zero.optimizer.param_groups[1]["group_name"] = "state_shared"
    zero.clip_grad = 1.0
    with pytest.raises(ValueError, match="must be disabled"):
        controller.apply_deepspeed(zero)


def test_initial_audit_window_does_not_change_thresholds() -> None:
    parameter = _parameter(1.0, (2.0,))
    config = load_config().outer_gradient_control
    controller = OuterGradientController(config, expected_groups=("qwen",))
    optimizer = _optimizer((("qwen", 1.0e-5, parameter),))

    for _ in range(config.audit_steps + 1):
        parameter.grad = torch.tensor((2.0,))
        _, audit = controller.apply_plain(optimizer)

    assert not audit.within_initial_audit_window
    assert audit.groups[0].max_norm == 1.0
    assert audit.groups[0].post_clip_norm == pytest.approx(1.0)


def test_metrics_are_stateless_with_respect_to_optimizer_checkpoint() -> None:
    controller = OuterGradientController(
        load_config().outer_gradient_control, expected_groups=("qwen",)
    )
    assert not hasattr(controller, "state_dict")
    assert SimpleNamespace(config=controller.config).config.audit_steps == 32
