from __future__ import annotations

from pathlib import Path

import pytest

from ttt_svcbench_qwen.a0_baseline import audit_a0_assets, parse_integer_answer, run_a0
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.data import (
    DatasetPurpose,
    DatasetSource,
    RuntimeSample,
    SVCBenchDataset,
    load_annotations,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "svcbench" / "flat.jsonl"


class FixtureBasePredictor:
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    model_revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    state_modules_enabled = False
    ttt_enabled = False
    prompt_template = "<video>\n{question}\nAnswer with one integer."
    generation_parameters = {"do_sample": False, "max_new_tokens": 8}

    def __init__(self, answers: tuple[str, ...]) -> None:
        self._answers = iter(answers)

    def predict(self, sample: RuntimeSample) -> tuple[str, float, int]:
        assert set(sample.model_input.as_payload()) == {
            "video",
            "question",
            "query_time",
            "explicit_time_values",
        }
        return next(self._answers), 0.25, 1024


def make_dataset() -> SVCBenchDataset:
    annotations = load_annotations(
        FIXTURE,
        source=DatasetSource("buaaplay/SVCBench", "fixture-revision", True),
        purpose=DatasetPurpose.OFFICIAL_CLEAN_EVALUATION,
    )
    return SVCBenchDataset(annotations, FIXTURE.parent / "videos")


def test_a0_metric_runner_requires_base_model_mode_and_keeps_labels_out_of_predictor() -> None:
    report = run_a0(make_dataset(), FixtureBasePredictor(("2", "3")))

    assert report.state_modules_enabled is False
    assert report.ttt_enabled is False
    assert report.metrics.query_count == 2
    assert report.metrics.exact_count_accuracy == 0.5
    assert report.metrics.count_mae_on_valid_predictions == 0.5
    assert report.metrics.answer_accuracy == 0.5
    assert report.metrics.mean_latency_seconds == 0.25
    assert report.metrics.peak_gpu_memory_bytes == 1024


def test_a0_rejects_state_or_ttt_enabled_predictor() -> None:
    predictor = FixtureBasePredictor(("2", "4"))
    predictor.ttt_enabled = True
    with pytest.raises(ValueError, match="disable State-TTT"):
        run_a0(make_dataset(), predictor)


def test_integer_answer_parser_is_strict() -> None:
    assert parse_integer_answer(" 42 ") == 42
    assert parse_integer_answer("The answer is 42") is None
    assert parse_integer_answer("-1") is None


def test_a0_asset_audit_reports_missing_8b_without_downloading() -> None:
    status = audit_a0_assets(load_config(), {})

    assert status.ready is False
    assert status.missing == ("QWEN_MODEL_ROOT", "SVCBENCH_ROOT")


def test_a0_asset_audit_uses_official_data_videos_layout(tmp_path: Path) -> None:
    config = load_config()
    data_root = tmp_path / "SVCBench"
    annotation = data_root / config.data.flat_annotation_file
    annotation.parent.mkdir(parents=True)
    annotation.touch()

    missing_videos = audit_a0_assets(config, {"SVCBENCH_ROOT": str(data_root)})
    assert missing_videos.missing == ("QWEN_MODEL_ROOT", "SVCBENCH_ROOT/data/videos")

    (data_root / config.data.video_directory).mkdir()
    annotations_only = audit_a0_assets(config, {"SVCBENCH_ROOT": str(data_root)})
    assert annotations_only.missing == ("QWEN_MODEL_ROOT",)
