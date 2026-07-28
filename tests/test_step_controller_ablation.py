from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.step_controller import (
    InnerStepController,
    build_inner_step_controller,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "model_state_ttt_8b.yaml"
CONFIG_BUILDER = ROOT / "scripts" / "build_a5_ttt_effect_config.py"
LEARNED_LAUNCHER = ROOT / "scripts" / "h200" / "train_a5_learned_step_ablation.sh"


def test_fixed_mode_has_no_controller_module_or_parameters() -> None:
    config = load_config(BASE_CONFIG)

    controller = build_inner_step_controller(config.fast_ttt.step_controller)

    assert config.fast_ttt.step_controller.mode == "fixed"
    assert controller is None


def test_learned_controller_initializes_exactly_to_fixed_step_and_is_bounded() -> None:
    raw = load_config(BASE_CONFIG).model_dump(mode="python")
    raw["fast_ttt"]["step_controller"]["mode"] = "learned"
    config = ProjectConfig.model_validate(raw)
    controller = build_inner_step_controller(config.fast_ttt.step_controller)
    assert isinstance(controller, InnerStepController)

    features = torch.randn(19, 7)
    step_sizes = controller(features.detach())

    assert step_sizes.tolist() == pytest.approx([1.0e-4] * 19, rel=1.0e-6)
    assert bool(torch.all(step_sizes > 0.0))
    assert bool(torch.all(step_sizes < 3.0e-4))
    assert sum(parameter.numel() for parameter in controller.parameters()) == 289


@pytest.mark.parametrize(
    ("variant", "path", "expected"),
    [
        ("A", None, None),
        ("B", ("fast_ttt", "optimizer", "learning_rate"), 2.0e-4),
        ("C", ("a5", "optimizer", "predictor_learning_rate"), 1.0e-4),
        ("D", ("loss", "auxiliary_outer_weight"), 0.2),
        ("E", ("outer_gradient_control", "max_grad_norm", "w0"), 0.15),
    ],
)
def test_config_builder_keeps_learned_step_as_explicit_ablation_layer(
    tmp_path: Path,
    variant: str,
    path: tuple[str, ...] | None,
    expected: float | None,
) -> None:
    output = tmp_path / variant / "project_config.yaml"
    completed = subprocess.run(
        [
            sys.executable,
            str(CONFIG_BUILDER),
            "--base",
            str(BASE_CONFIG),
            "--output",
            str(output),
            "--fixed-variant",
            variant,
            "--step-controller",
            "learned",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    raw = yaml.safe_load(output.read_text(encoding="utf-8"))
    config = ProjectConfig.model_validate(raw)

    assert summary["fixed_variant"] == variant
    assert summary["step_controller_mode"] == "learned"
    assert config.fast_ttt.step_controller.mode == "learned"
    assert config.a5.counterfactual_audit.enabled is True
    if path is not None:
        value: object = raw
        for key in path:
            assert isinstance(value, dict)
            value = value[key]
        assert value == expected

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                str(CONFIG_BUILDER),
                "--base",
                str(BASE_CONFIG),
                "--output",
                str(output),
                "--fixed-variant",
                variant,
                "--step-controller",
                "learned",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def test_learned_step_launcher_matches_variant_a_v4_training_contract() -> None:
    launcher = LEARNED_LAUNCHER.read_text(encoding="utf-8")

    assert "[[ $# -eq 2 ]] || usage" in launcher
    assert 'exec bash "$ABLATION_LAUNCHER" A learned "$@"' in launcher
    assert "a5_dense_querybundle_train_support_statequery_fp16_v4" in launcher
    assert "260726_a5_dense_querybundle_v4_fp16" in launcher
    assert 'TTT_CHECKPOINT_POLICY="atomic_final_only"' in launcher
    assert 'TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"' in launcher
    assert "a5_learned_step_dense_querybundle_v4_4epoch_finalonly" in launcher
