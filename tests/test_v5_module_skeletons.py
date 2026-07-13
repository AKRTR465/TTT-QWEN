from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "ttt_svcbench_qwen"

SKELETONS = (
    ("model", "build_model", "P13"),
    ("fast_ttt", "build_fast_ttt_adapter", "P5"),
    ("state_encoder", "build_state_encoders", "P6-P7"),
    ("observation_heads", "build_observation_heads", "P8"),
    ("state_bank", "build_state_bank", "P9"),
    ("identity_bank", "build_identity_bank", "P10"),
    ("state_retriever", "build_state_retriever", "P11"),
    ("state_reader", "build_state_reader", "P12"),
    ("input_composer", "compose_inputs", "P13"),
    ("losses", "compute_losses", "P14"),
    ("functional_sgd", "functional_sgd_step", "P14"),
    ("trainer", "build_trainer", "P15-P19"),
    ("inference", "run_inference", "P18"),
)


def import_module(name: str) -> ModuleType:
    return importlib.import_module(f"ttt_svcbench_qwen.{name}")


@pytest.mark.parametrize(("module_name", "entrypoint", "owner"), SKELETONS)
def test_recommended_modules_import_and_unimplemented_paths_fail_explicitly(
    module_name: str, entrypoint: str, owner: str
) -> None:
    module = import_module(module_name)
    doc = module.__doc__ or ""

    assert "Inputs:" in doc
    assert "Outputs:" in doc
    assert "Forbidden:" in doc
    with pytest.raises(NotImplementedError, match=owner):
        getattr(module, entrypoint)()


def test_all_required_p1_module_files_exist() -> None:
    actual = {path.stem for path in PACKAGE.glob("*.py")}
    expected = {name for name, _, _ in SKELETONS} | {
        "__init__",
        "config",
        "query_encoder",
        "qwen_adapter",
    }

    assert expected <= actual


def test_model_skeleton_contains_composition_only() -> None:
    source = (PACKAGE / "model.py").read_text(encoding="utf-8")

    assert "import torch" not in source
    assert "nn.Module" not in source
    assert "Linear(" not in source
    assert "Optimizer" not in source
    assert "def build_model" in source
