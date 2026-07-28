from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.step_controller import (
    STEP_CONTROLLER_FEATURE_CONTRACT_VERSION,
    CausalSupportPosition,
    InnerStepController,
    build_inner_step_controller,
    build_step_controller_features,
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
    assert (
        int(controller.feature_contract_version.item())
        == STEP_CONTROLLER_FEATURE_CONTRACT_VERSION
    )

    restored = InnerStepController(config.fast_ttt.step_controller)
    restored.load_state_dict(controller.state_dict(), strict=True)
    legacy = dict(controller.state_dict())
    del legacy["feature_contract_version"]
    with pytest.raises(RuntimeError, match="feature_contract_version"):
        restored.load_state_dict(legacy, strict=True)


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
def test_config_builder_materializes_each_fixed_single_factor_variant(
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
            "fixed",
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
    assert summary["step_controller_mode"] == "fixed"
    assert config.fast_ttt.step_controller.mode == "fixed"
    assert config.a5.effect_ablation.fixed_variant == variant
    assert config.outer_gradient_control.mode.value == (
        "per_group_l2_single_factor_ablation"
        if variant in {"C", "E"}
        else "per_group_l2_equal_update_cap"
    )
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
                "fixed",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


def test_learned_step_is_allowed_only_on_variant_a(tmp_path: Path) -> None:
    learned = tmp_path / "learned.yaml"
    subprocess.run(
        [
            sys.executable,
            str(CONFIG_BUILDER),
            "--base",
            str(BASE_CONFIG),
            "--output",
            str(learned),
            "--fixed-variant",
            "A",
            "--step-controller",
            "learned",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert load_config(learned).fast_ttt.step_controller.mode == "learned"

    for variant in "BCDE":
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(
                [
                    sys.executable,
                    str(CONFIG_BUILDER),
                    "--base",
                    str(BASE_CONFIG),
                    "--output",
                    str(tmp_path / f"learned-{variant}.yaml"),
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


@pytest.mark.parametrize(
    ("support_index", "support_count", "expected_window", "expected_episode"),
    [(0, 3, 1 / 8, 1 / 3), (7, 8, 1.0, 1.0), (8, 10, 1 / 8, 0.9)],
)
def test_controller_features_use_causal_k8_and_episode_progress(
    support_index: int,
    support_count: int,
    expected_window: float,
    expected_episode: float,
) -> None:
    config = load_config().fast_ttt.step_controller.model_copy(update={"mode": "learned"})
    controller = InnerStepController(config)
    term = SimpleNamespace(valid_counts=torch.ones(1, dtype=torch.int64))
    scale = SimpleNamespace(
        pair_element_count=torch.tensor(1.0),
        target_sum_squares=torch.tensor(1.0),
        error_sum_squares=torch.tensor(1.0),
    )
    output = SimpleNamespace(
        per_row_total=torch.tensor([0.5]),
        temporal_scale_audit=scale,
        pred=term,
        identity=term,
        event=term,
    )

    features = build_step_controller_features(
        ttt_output=output,  # type: ignore[arg-type]
        start_time=1.0,
        end_time=2.0,
        previous_end_time=1.0,
        position=CausalSupportPosition(support_index, support_count, 8),
        controller=controller,
    )

    assert float(features[0, 0]) == pytest.approx(expected_window)
    assert float(features[0, 1]) == pytest.approx(expected_episode)


def test_learned_step_launcher_matches_variant_a_v4_training_contract() -> None:
    launcher = LEARNED_LAUNCHER.read_text(encoding="utf-8")

    assert "[[ $# -eq 2 ]] || usage" in launcher
    assert 'exec bash "$ABLATION_LAUNCHER" A learned "$@"' in launcher
    assert "a5_dense_querybundle_train_support_statequery_fp16_v4" in launcher
    assert "260726_a5_dense_querybundle_v4_fp16" in launcher
    assert 'TTT_CHECKPOINT_POLICY="atomic_final_only"' in launcher
    assert 'TTT_SMOKE_SHORTEST_FIRST="${TTT_SMOKE_SHORTEST_FIRST:-0}"' in launcher
    assert "a5_learned_step_dense_querybundle_v4_4epoch_finalonly" in launcher
