from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from ttt_svcbench_qwen.a0_baseline import run_a0
from ttt_svcbench_qwen.data import (
    DatasetPurpose,
    DatasetSource,
    RuntimeSample,
    SVCBenchDataset,
    create_group_kfold_manifest,
    load_annotations,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "svcbench"
ARTIFACTS = ROOT / "docs" / "p2" / "artifacts"


class SyntheticA0Predictor:
    model_id = "synthetic/p2-a0-engineering-fixture"
    model_revision = "fixture-v1"
    state_modules_enabled = False
    ttt_enabled = False
    prompt_template = "SYNTHETIC ENGINEERING DRY-RUN: {question} Answer with ONLY one integer."
    generation_parameters = {"do_sample": False, "max_new_tokens": 8, "synthetic": True}

    def __init__(self) -> None:
        self._answers = iter(("2", "not-an-integer"))

    def predict(self, sample: RuntimeSample) -> tuple[str, float, int]:
        assert sample.model_input.query_time <= 2.0
        return next(self._answers), 0.01, 0


def _read_json(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _json_normalize(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def test_committed_synthetic_fold_manifest_is_reproducible_and_video_disjoint() -> None:
    annotations = load_annotations(
        FIXTURES / "synthetic_training.jsonl",
        source=DatasetSource("synthetic-p2-training", "fixture-v1", False),
        purpose=DatasetPurpose.TRAINING,
    )
    manifest = create_group_kfold_manifest(annotations, n_splits=3, seed=42)

    assert _read_json(ARTIFACTS / "synthetic-fold-manifest.json") == _json_normalize(
        asdict(manifest)
    )
    assert all(
        not (set(fold.train_video_ids) & set(fold.validation_video_ids)) for fold in manifest.folds
    )


def test_committed_synthetic_a0_report_covers_metrics_and_failure_case() -> None:
    annotations = load_annotations(
        FIXTURES / "flat.jsonl",
        source=DatasetSource("synthetic-p2-evaluation", "fixture-v1", False),
        purpose=DatasetPurpose.OFFICIAL_CLEAN_EVALUATION,
    )
    report = run_a0(
        SVCBenchDataset(annotations, FIXTURES / "videos"),
        SyntheticA0Predictor(),
    )

    assert _read_json(ARTIFACTS / "synthetic-a0-report.json") == _json_normalize(asdict(report))
    assert report.model_id.startswith("synthetic/")
    assert report.state_modules_enabled is False
    assert report.ttt_enabled is False
    assert report.metrics.query_count == 2
    assert report.metrics.valid_prediction_count == 1
    assert any(prediction.predicted_count is None for prediction in report.predictions)


def test_p2_gate_is_marked_without_erasing_the_real_a0_requirement() -> None:
    todo = (ROOT / "TODO.md").read_text(encoding="utf-8")
    p2 = todo.split("## P2.", maxsplit=1)[1].split("## P3.", maxsplit=1)[0]
    p2_checkboxes = [line for line in p2.splitlines() if line.startswith("- [")]
    p2_readme = (ROOT / "docs" / "p2" / "README.md").read_text(encoding="utf-8")
    decisions = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")

    assert len(p2_checkboxes) == 36
    assert all(line.startswith("- [x]") for line in p2_checkboxes)
    assert "GATE_STATUS: `passed`" in p2_readme
    assert "P3_ALLOWED: `true`" in p2_readme
    assert "P19/P21/P22" in p2_readme
    assert "## P2 合成退出决策" in decisions
