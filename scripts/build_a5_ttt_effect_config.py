#!/usr/bin/env python3
"""Materialize one auditable A5 TTT-effect ablation config without editing the base YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml

from ttt_svcbench_qwen.config import ProjectConfig

_FIXED_VARIANTS: dict[str, tuple[str, float] | None] = {
    "A": None,
    "B": ("fast_ttt.optimizer.learning_rate", 2.0e-4),
    "C": ("a5.optimizer.predictor_learning_rate", 1.0e-4),
    "D": ("loss.auxiliary_outer_weight", 0.2),
    "E": ("outer_gradient_control.max_grad_norm.w0", 0.15),
}


def _set_path(raw: dict[str, Any], path: str, value: object) -> None:
    current = raw
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            raise ValueError(f"config path is not a mapping: {path}")
        current = cast(dict[str, Any], child)
    current[parts[-1]] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-variant", choices=tuple(_FIXED_VARIANTS), required=True)
    parser.add_argument("--step-controller", choices=("fixed", "learned"), default="fixed")
    arguments = parser.parse_args()

    raw_object = yaml.safe_load(arguments.base.read_text(encoding="utf-8"))
    if not isinstance(raw_object, dict):
        raise ValueError("base project config must be one mapping")
    raw = cast(dict[str, Any], raw_object)
    mutation = _FIXED_VARIANTS[arguments.fixed_variant]
    if mutation is not None:
        _set_path(raw, mutation[0], mutation[1])
    _set_path(raw, "fast_ttt.step_controller.mode", arguments.step_controller)
    _set_path(raw, "a5.counterfactual_audit.enabled", True)
    config = ProjectConfig.model_validate(raw)

    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite an existing config: {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(
        config.model_dump(mode="json"),
        sort_keys=False,
        allow_unicode=True,
    )
    arguments.output.write_text(serialized, encoding="utf-8")
    summary = {
        "fixed_variant": arguments.fixed_variant,
        "step_controller_mode": arguments.step_controller,
        "counterfactual_audit_enabled": config.a5.counterfactual_audit.enabled,
        "output": str(arguments.output.resolve()),
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "single_fixed_mutation": mutation,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
