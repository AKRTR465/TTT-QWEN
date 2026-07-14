from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "ttt_svcbench_qwen"

IMPLEMENTED = (
    ("losses", "compute_losses"),
    ("functional_sgd", "functional_sgd_step"),
)

SKELETONS = (
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


@pytest.mark.parametrize(("module_name", "entrypoint"), IMPLEMENTED)
def test_p14_modules_import_with_documented_callable_entrypoints(
    module_name: str, entrypoint: str
) -> None:
    module = import_module(module_name)
    doc = module.__doc__ or ""

    assert "Inputs:" in doc
    assert "Outputs:" in doc
    assert "Forbidden:" in doc
    assert callable(getattr(module, entrypoint))


def test_all_required_p1_module_files_exist() -> None:
    actual = {path.stem for path in PACKAGE.glob("*.py")}
    expected = (
        {name for name, _ in IMPLEMENTED}
        | {name for name, _, _ in SKELETONS}
        | {
            "__init__",
            "config",
            "fast_ttt",
            "identity_bank",
            "input_composer",
            "model",
            "query_encoder",
            "qwen_adapter",
            "state_encoder",
            "state_reader",
            "state_retriever",
        }
    )

    assert expected <= actual


def test_p13_model_remains_composition_only() -> None:
    source = (PACKAGE / "model.py").read_text(encoding="utf-8")

    assert "Linear(" not in source
    assert "Optimizer" not in source
    assert ".backward(" not in source
    assert "functional_sgd" not in source
    assert "def observe_chunk" in source
    assert "def answer_query" in source
    assert "def build_model" in source
