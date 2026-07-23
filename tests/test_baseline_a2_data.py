from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ttt_svcbench_qwen.baseline_a2_data import (
    BaselineA2ClipDataset,
    build_baseline_a2_train_sampler,
    load_baseline_a2_clip_dataset,
)
from ttt_svcbench_qwen.production_runtime import _a2_support_chunk_specs, _user_message


def _write_dataset_info(root: Path) -> None:
    (root / "dataset_info.json").write_text(
        json.dumps(
            {
                "svcbench_qwen3vl_sft": {
                    "file_name": "svcbench_qwen3vl_sft.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages", "videos": "videos"},
                }
            }
        ),
        encoding="utf-8",
    )


def _sft_row(
    index: int,
    *,
    query_index: int | None,
    query_time: float,
    answer: int,
    subtype: str = "O1-Snapshot",
) -> dict[str, object]:
    q_id = f"{index:04d}"
    question = "How many pillows are visible at this moment?"
    user = (
        "<video>\nAnswer the counting question based only on the provided video. "
        "Return only the final count as an integer.\nQuestion: " + question
    )
    return {
        "id": f"svcbench-{q_id}",
        "q_id": q_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(answer)},
        ],
        "videos": [f"videos/{q_id}.mp4"],
        "answer": str(answer),
        "question": question,
        "source_dataset": "Demo",
        "source_video_path": "source.mp4",
        "counting_subtype": subtype,
        "query_time": query_time,
        "query_index": query_index,
    }


def _fixture_dataset(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "dataset"
    (root / "videos").mkdir(parents=True)
    _write_dataset_info(root)
    rows = [
        _sft_row(0, query_index=0, query_time=100.0, answer=2),
        _sft_row(1, query_index=1, query_time=110.0, answer=4),
    ]
    content = json.dumps(rows).encode()
    (root / "svcbench_qwen3vl_sft.json").write_bytes(content)
    for index in range(2):
        (root / "videos" / f"{index:04d}.mp4").touch()
    sidecar = root / "raw.jsonl"
    sidecar.write_text(
        json.dumps(
            {
                "id": "trajectory",
                "source_dataset": "Demo",
                "video_path": "source.mp4",
                "question": "How many pillows are visible at this moment?",
                "query_points": {"time": [100.0, 110.0], "count": [2, 4]},
                "occurrence_times": [94.0, 100.0, 101.0, 111.0],
                "counting_type": "O1",
                "counting_subtype": "O1-Snap",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root, sidecar, hashlib.sha256(content).hexdigest()


def test_baseline_a2_preserves_sft_order_prompt_and_standard_sampling(tmp_path: Path) -> None:
    root, sidecar, digest = _fixture_dataset(tmp_path)
    durations = {"0000.mp4": 5.0, "0001.mp4": 10.0}
    dataset = load_baseline_a2_clip_dataset(
        root,
        dataset_name="svcbench_qwen3vl_sft",
        weak_sidecar_path=sidecar,
        expected_sha256=digest,
        expected_rows=2,
        duration_resolver=lambda path: durations[path.name],
    )

    assert isinstance(dataset, BaselineA2ClipDataset)
    assert [row.query.runtime.query_id for row in dataset.records] == [
        "svcbench-0000",
        "svcbench-0001",
    ]
    assert [row.query.answer.answer for row in dataset.records] == ["2", "4"]
    content = dataset.records[0].answer_user_content
    assert content is not None
    message = _user_message("ignored", user_content=content)
    assert message["content"][1]["text"] == content.removeprefix("<video>\n")  # type: ignore[index]
    sampler = build_baseline_a2_train_sampler(dataset, rank=0, world_size=4)
    assert sorted(iter(sampler)) == [0, 1]


def test_baseline_a2_accepts_single_llamafactory_dataset_list(tmp_path: Path) -> None:
    root, sidecar, digest = _fixture_dataset(tmp_path)
    dataset = load_baseline_a2_clip_dataset(
        root,
        dataset_name=["svcbench_qwen3vl_sft"],
        weak_sidecar_path=sidecar,
        expected_sha256=digest,
        expected_rows=2,
        duration_resolver=lambda _path: 5.0,
    )
    assert len(dataset) == 2


def test_baseline_a2_rejects_multiple_llamafactory_datasets(tmp_path: Path) -> None:
    root, sidecar, digest = _fixture_dataset(tmp_path)
    with pytest.raises(ValueError, match="exactly one LLaMA-Factory dataset"):
        load_baseline_a2_clip_dataset(
            root,
            dataset_name=["svcbench_qwen3vl_sft", "other"],
            weak_sidecar_path=sidecar,
            expected_sha256=digest,
            expected_rows=2,
            duration_resolver=lambda _path: 5.0,
        )


def test_baseline_a2_uses_clip_local_time_and_zero_support_for_short_clip(
    tmp_path: Path,
) -> None:
    root, sidecar, digest = _fixture_dataset(tmp_path)
    dataset = load_baseline_a2_clip_dataset(
        root,
        dataset_name="svcbench_qwen3vl_sft",
        weak_sidecar_path=sidecar,
        expected_sha256=digest,
        expected_rows=2,
        duration_resolver=lambda path: 5.0 if path.name == "0000.mp4" else 10.0,
    )

    first, second = dataset.records
    assert first.query.runtime.query_time == first.query.weak.query_time == 5.0
    assert first.query.weak.occurrence_points == (5.0,)
    assert second.query.weak.occurrence_points == (0.0, 1.0)
    assert _a2_support_chunk_specs(first, root / first.relative_video_path) == ()
    assert _a2_support_chunk_specs(second, root / second.relative_video_path)
    assert dataset.audit.masked_occurrence_points > 0


def test_baseline_a2_rejects_incomplete_join_and_hash_drift(tmp_path: Path) -> None:
    root, sidecar, digest = _fixture_dataset(tmp_path)
    with pytest.raises(ValueError, match="SHA256 drift"):
        load_baseline_a2_clip_dataset(
            root,
            dataset_name="svcbench_qwen3vl_sft",
            weak_sidecar_path=sidecar,
            expected_sha256="0" * 64,
            expected_rows=2,
            duration_resolver=lambda _path: 5.0,
        )

    rows = json.loads((root / "svcbench_qwen3vl_sft.json").read_text())
    rows[0]["question"] = "unmatched question"
    content = json.dumps(rows).encode()
    (root / "svcbench_qwen3vl_sft.json").write_bytes(content)
    with pytest.raises(ValueError, match="no official-weak sidecar match"):
        load_baseline_a2_clip_dataset(
            root,
            dataset_name="svcbench_qwen3vl_sft",
            weak_sidecar_path=sidecar,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            expected_rows=2,
            duration_resolver=lambda _path: 5.0,
        )
