"""Test-only builders and topology counters excluded from the production wheel."""

from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor, nn

from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.model import (
    ModelComponents,
    ModelFeatureFlags,
    StateTTTModel,
)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def tensor_count(values: Iterable[Tensor]) -> int:
    return sum(value.numel() for value in values)


def make_test_model(
    config: ProjectConfig,
    *,
    components: ModelComponents,
    feature_flags: ModelFeatureFlags | None = None,
) -> StateTTTModel:
    return StateTTTModel(config, components, feature_flags or ModelFeatureFlags())
