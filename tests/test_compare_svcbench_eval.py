from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from scripts.compare_svcbench_eval import main as compare_main
from scripts.stamp_svcbench_eval_metrics import main as stamp_main


def _metrics(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "metadata": {"evaluation_scope": "train_set"},
                "overall": {
                    "n_query_points": 3706,
                    "gpa": 1.0,
                    "moc": 2.0,
                    "uda": 3.0,
                    "exact_match": 0.1,
                    "exact_match_macro": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )


def _stamp(
    monkeypatch: pytest.MonkeyPatch,
    metrics: Path,
    *,
    method: str,
    inputs: dict[str, Path],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stamp_svcbench_eval_metrics.py",
            "--metrics",
            str(metrics),
            "--method",
            method,
            "--selection",
            str(inputs["selection"]),
            "--prepared-sft",
            str(inputs["prepared_sft"]),
            "--score-annotations",
            str(inputs["annotations"]),
            "--scorer",
            str(inputs["scorer"]),
            "--manifest",
            str(inputs["manifest"]),
            "--evaluation-config",
            str(inputs["config"]),
            "--model-identity",
            f"/{method}/model",
        ],
    )
    assert stamp_main() == 0


def _inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: tmp_path / f"{name}.txt"
        for name in ("selection", "prepared_sft", "annotations", "scorer", "manifest", "config")
    }
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    return paths


def test_comparison_requires_matching_stamped_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline.json"
    a2_static = tmp_path / "a2.json"
    _metrics(baseline)
    _metrics(a2_static)
    inputs = _inputs(tmp_path)
    _stamp(monkeypatch, baseline, method="baseline", inputs=inputs)
    _stamp(monkeypatch, a2_static, method="a2_static", inputs=inputs)

    output = tmp_path / "comparison"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_svcbench_eval.py",
            "--baseline",
            str(baseline),
            "--a2-static",
            str(a2_static),
            "--output-dir",
            str(output),
        ],
    )
    assert compare_main() == 0
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["comparison_provenance"]["baseline"]["method"] == "baseline"
    assert comparison["comparison_provenance"]["a2_static"]["method"] == "a2_static"


def test_comparison_rejects_different_selections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "baseline.json"
    a2_static = tmp_path / "a2.json"
    _metrics(baseline)
    _metrics(a2_static)
    inputs = _inputs(tmp_path)
    _stamp(monkeypatch, baseline, method="baseline", inputs=inputs)
    inputs["selection"].write_text("different selection", encoding="utf-8")
    _stamp(monkeypatch, a2_static, method="a2_static", inputs=inputs)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_svcbench_eval.py",
            "--baseline",
            str(baseline),
            "--a2-static",
            str(a2_static),
            "--output-dir",
            str(tmp_path / "comparison"),
        ],
    )
    with pytest.raises(ValueError, match="selection_sha256"):
        compare_main()
