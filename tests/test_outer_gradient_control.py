from __future__ import annotations

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


_ALL_GROUPS = (
    "qwen",
    "state_shared",
    "state_task",
    "state_router_time",
    "state_retrieval",
    "w0",
    "associative",
)


def test_zero_partition_groups_match_plain_reference() -> None:
    parameters = (
        _parameter(1.0, (3.0, 4.0)),
        _parameter(1.0, (30.0, 40.0)),
        _parameter(1.0, (30.0, 40.0)),
        _parameter(1.0, (30.0, 40.0)),
        _parameter(1.0, (30.0, 40.0)),
        _parameter(1.0, (0.06, 0.08)),
        _parameter(1.0, (60.0, 80.0)),
    )
    learning_rates = (1.0e-5, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4)
    optimizer = _optimizer(
        tuple(zip(_ALL_GROUPS, learning_rates, parameters, strict=True)),
    )
    controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=_ALL_GROUPS,
    )

    audit = controller.apply_deepspeed(_FakeZero(optimizer))

    # Each of the four state groups is capped at 0.05 on its own, so no group can
    # scale another.
    assert [float(parameter.grad.norm()) for parameter in parameters] == pytest.approx(
        [1.0, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1]
    )
    assert tuple(group.name for group in audit.groups) == _ALL_GROUPS
    assert audit.group("state_task").max_norm == 0.05
    assert dict(audit.metrics())["outer_grad/associative/post_norm"] == pytest.approx(0.1)


def test_global_clip_must_stay_disabled() -> None:
    optimizer = _optimizer((("qwen", 1.0e-5, _parameter(1.0, (1.0,))),))
    controller = OuterGradientController(
        load_config().outer_gradient_control,
        expected_groups=("qwen",),
    )
    zero = _FakeZero(optimizer)
    zero.clip_grad = 1.0
    with pytest.raises(ValueError, match="must be disabled"):
        controller.apply_deepspeed(zero)


def test_nonfinite_loss_sanitizer_preserves_ga_backward_and_skips_one_update() -> None:
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
        sanitized = sanitize_scalar_loss(parameter * factor, source=source, controller=controller)
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
    sanitize_scalar_loss(parameter.sum(), source="finite_local", controller=controller)

    with pytest.warns(RuntimeWarning, match="remote_rank"):
        audit = controller.apply_deepspeed(_FakeZero(optimizer))

    assert audit.nonfinite_loss_sources == ("remote_rank",)
    assert audit.skipped_nonfinite_loss
    assert parameter.grad is not None and not torch.isfinite(parameter.grad).all()


def test_gradient_nonfinite_remains_owned_by_zero_overflow() -> None:
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
