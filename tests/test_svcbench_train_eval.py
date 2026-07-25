from pathlib import Path

import pytest
import torch

from ttt_svcbench_qwen.a2_static_eval import prompt_only_answer_inputs
from ttt_svcbench_qwen.data import (
    AnnotationFormat,
    DatasetPurpose,
    DatasetSource,
    LoadedAnnotations,
    OccurrenceAnnotations,
    SampleIdentity,
    SupervisionLabels,
    SVCBenchRecord,
    canonical_video_id,
)
from ttt_svcbench_qwen.svcbench_train_eval import resolve_sft_annotation_rows
from ttt_svcbench_qwen.trainer import StageAEpisodeAnswerInputs


def _record(*, query_id: str, query_index: int, query_time: float) -> SVCBenchRecord:
    source = "Demo"
    video = "clip.mp4"
    return SVCBenchRecord(
        identity=SampleIdentity(
            query_id=query_id,
            query_index=query_index,
            video_id=canonical_video_id(source, video),
            trajectory_id="trajectory",
        ),
        source_dataset=source,
        relative_video_path=video,
        question="How many events?",
        query_time=query_time,
        labels=SupervisionLabels(
            answer="3",
            count=3,
            occurrence_times=OccurrenceAnnotations((), (), ()),
            counting_type="O2",
            counting_subtype="O2-Unique",
        ),
    )


def test_resolve_sft_annotation_rows_uses_nearest_time_with_training_tolerance() -> None:
    annotations = LoadedAnnotations(
        records=(
            _record(query_id="trajectory:0", query_index=0, query_time=10.0),
            _record(query_id="trajectory:1", query_index=1, query_time=20.0),
        ),
        source=DatasetSource("demo", "v1", False),
        purpose=DatasetPurpose.TRAINING,
        annotation_format=AnnotationFormat.GROUPED,
        annotation_sha256="abc",
        annotation_path=Path("annotations.jsonl"),
    )
    rows = [
        {
            "source_dataset": "Demo",
            "source_video_path": "clip.mp4",
            "question": "How many events?",
            "query_index": 0,
            "query_time": 11.0,
        },
        {
            "source_dataset": "Demo",
            "source_video_path": "clip.mp4",
            "question": "How many events?",
            "query_index": 1,
            "query_time": 20.0,
        },
    ]
    resolved = resolve_sft_annotation_rows(annotations, rows)
    assert tuple(row.identity.query_id for row in resolved) == (
        "trajectory:0",
        "trajectory:1",
    )


def test_resolve_sft_annotation_rows_rejects_time_drift() -> None:
    annotations = LoadedAnnotations(
        records=(_record(query_id="trajectory:0", query_index=0, query_time=10.0),),
        source=DatasetSource("demo", "v1", False),
        purpose=DatasetPurpose.TRAINING,
        annotation_format=AnnotationFormat.GROUPED,
        annotation_sha256="abc",
        annotation_path=Path("annotations.jsonl"),
    )
    with pytest.raises(ValueError, match="query_time"):
        resolve_sft_annotation_rows(
            annotations,
            [
                {
                    "source_dataset": "Demo",
                    "source_video_path": "clip.mp4",
                    "question": "How many events?",
                    "query_index": 0,
                    "query_time": 12.0,
                }
            ],
        )


def test_prompt_only_answer_inputs_removes_teacher_forced_answer() -> None:
    answer = StageAEpisodeAnswerInputs(
        base_input_ids=torch.tensor([[10, 11, 12, 3, 4]], dtype=torch.int64),
        base_attention_mask=torch.ones((1, 5), dtype=torch.int64),
        pixel_values_videos=torch.zeros((2, 4), dtype=torch.float32),
        video_grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.int64),
        tokenizer=object(),
        embedding_owner=object(),
        rope_indexer=object(),
        qwen_kwargs=(("use_cache", False),),
    )
    labels = torch.tensor([[-100, -100, -100, 3, 4]], dtype=torch.int64)
    prompt = prompt_only_answer_inputs(answer, labels)
    assert prompt.base_input_ids.tolist() == [[10, 11, 12]]
    assert prompt.base_attention_mask.tolist() == [[1, 1, 1]]
    assert prompt.qwen_kwargs == (("use_cache", True),)


def test_prompt_only_answer_inputs_requires_assistant_boundary() -> None:
    answer = StageAEpisodeAnswerInputs(
        base_input_ids=torch.tensor([[10, 11]], dtype=torch.int64),
        base_attention_mask=torch.ones((1, 2), dtype=torch.int64),
        pixel_values_videos=torch.zeros((2, 4), dtype=torch.float32),
        video_grid_thw=torch.tensor([[1, 1, 2]], dtype=torch.int64),
        tokenizer=object(),
        embedding_owner=object(),
        rope_indexer=object(),
    )
    with pytest.raises(ValueError, match="assistant answer boundary"):
        prompt_only_answer_inputs(answer, torch.full((1, 2), -100))


@pytest.mark.parametrize(
    ("script_name", "forwarded_names"),
    (
        (
            "eval_svcbench_train3706_baseline.sh",
            (
                "TTT_PROJECT_ROOT",
                "LLAMAFACTORY_ROOT",
                "SOURCE_SFT",
                "RAW_ANNOTATIONS",
                "SCORE_ANNOTATIONS",
                "SVCBENCH_SCORER",
            ),
        ),
        (
            "eval_svcbench_train3706_a2_static.sh",
            (
                "TTT_PROJECT_ROOT",
                "LLAMAFACTORY_ROOT",
                "TTT_H200_VENV",
                "SOURCE_SFT",
                "RAW_ANNOTATIONS",
                "SCORE_ANNOTATIONS",
                "SVCBENCH_VIDEO_ROOT",
                "SVCBENCH_SCORER",
                "TTT_PREPROCESS_CACHE_ROOT",
                "TTT_PREPROCESS_CACHE_NAMESPACE",
                "VISUAL_COST_INDEX",
            ),
        ),
    ),
)
def test_eval_tmux_launchers_forward_resolved_environment(
    script_name: str,
    forwarded_names: tuple[str, ...],
) -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "h200" / script_name).read_text(encoding="utf-8")
    assert 'command_text="set -o pipefail;' in script
    for name in forwarded_names:
        assert f'"{name}=$' in script
