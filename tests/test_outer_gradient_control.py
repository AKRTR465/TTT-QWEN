from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.outer_gradient_control import (
    OuterGradientController,
    sanitize_scalar_loss,
)


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


def test_initial_audit_window_does_not_change_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ttt_svcbench_qwen.outer_gradient_control.version", lambda _name: "0.18.8")
    parameter = _parameter(1.0, (2.0,))
    config = load_config().outer_gradient_control
    controller = OuterGradientController(config, expected_groups=("qwen",))
    optimizer = _optimizer((("qwen", 1.0e-5, parameter),))

    for _ in range(config.audit_steps + 1):
        parameter.grad = torch.tensor((2.0,))
        audit = controller.apply_deepspeed(_FakeZero(optimizer))

    assert not audit.within_initial_audit_window
    assert audit.groups[0].max_norm == 1.0
    assert audit.groups[0].post_clip_norm == pytest.approx(1.0)


def test_metrics_are_stateless_with_respect_to_optimizer_checkpoint() -> None:
    controller = OuterGradientController(
        load_config().outer_gradient_control, expected_groups=("qwen",)
    )
    assert not hasattr(controller, "state_dict")
    assert SimpleNamespace(config=controller.config).config.audit_steps == 32


def test_nonfinite_loss_sanitizer_preserves_ga_backward_and_skips_one_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ttt_svcbench_qwen.outer_gradient_control.version", lambda _name: "0.18.8")
    parameter = nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.zeros_like(parameter)
    optimizer = _optimizer((("qwen", 1.0e-5, parameter),))
    zero = _FakeZero(optimizer)
    controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen",),
    )
    backward_count = 0
    for source, factor in (("first", 1.0), ("middle", float("nan")), ("last", 2.0)):
        sanitized = sanitize_scalar_loss(
            parameter * factor,
            source=source,
            controller=controller,
        )
        assert torch.isfinite(sanitized)
        sanitized.backward()
        backward_count += 1

    before = parameter.detach().clone()
    with pytest.warns(RuntimeWarning, match="nonfinite Outer loss"):
        audit = controller.apply_deepspeed(zero)
    scheduler_steps = 0
    if not zero.has_overflow(partition_gradients=True):
        optimizer.step()
        scheduler_steps += 1

    assert backward_count == 3
    assert torch.equal(parameter.detach(), before)
    assert scheduler_steps == 0
    assert audit.skipped_nonfinite
    assert audit.skipped_nonfinite_loss
    assert audit.nonfinite_loss_sources == ("middle",)
    assert audit.attempted_update_count == 1
    assert audit.skipped_update_count == 1
    assert controller.skipped_update_count == 1


def test_remote_rank_nonfinite_loss_injects_local_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ttt_svcbench_qwen.outer_gradient_control.version", lambda _name: "0.18.8")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    collective_count = 0

    def all_reduce(value: torch.Tensor, **_kwargs: object) -> None:
        nonlocal collective_count
        if collective_count == 0:
            value.fill_(1)
        collective_count += 1

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    parameter = _parameter(1.0, (1.0,))
    optimizer = _optimizer((("qwen", 1.0e-5, parameter),))
    controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen",),
    )
    sanitize_scalar_loss(
        parameter.sum(),
        source="finite_local",
        controller=controller,
    )

    with pytest.warns(RuntimeWarning, match="remote_rank"):
        audit = controller.apply_deepspeed(_FakeZero(optimizer))

    assert audit.nonfinite_loss_sources == ("remote_rank",)
    assert audit.skipped_nonfinite_loss
    assert parameter.grad is not None and not torch.isfinite(parameter.grad).all()


def test_nonfinite_loss_without_zero_gradient_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ttt_svcbench_qwen.outer_gradient_control.version", lambda _name: "0.18.8")
    parameter = _parameter(1.0, (1.0,))
    optimizer = _optimizer((("qwen", 1.0e-5, parameter),))
    zero = _FakeZero(optimizer)
    zero.averaged_gradients[0] = []
    controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen",),
    )
    sanitize_scalar_loss(
        parameter.sum() * float("nan"),
        source="missing_gradient",
        controller=controller,
    )

    with pytest.raises(RuntimeError, match="no ZeRO averaged gradient"):
        controller.apply_deepspeed(zero)


def test_gradient_nonfinite_remains_owned_by_zero_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ttt_svcbench_qwen.outer_gradient_control.version", lambda _name: "0.18.8")
    parameter = _parameter(1.0, (float("nan"),))
    optimizer = _optimizer((("qwen", 1.0e-5, parameter),))
    controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen",),
    )

    audit = controller.apply_deepspeed(_FakeZero(optimizer))

    assert audit.skipped_nonfinite
    assert not audit.skipped_nonfinite_loss
    assert audit.nonfinite_loss_sources == ()
