"""Run and audit the A0 base-Qwen zero-shot baseline without State-TTT components.

Inputs: official-clean runtime samples, separated evaluator labels, and a base-model predictor.
Outputs: per-query predictions, exact/MAE/answer metrics, latency, memory, and run manifest.
Forbidden: State Bank, Fast Adapter, TTT updates, label-bearing prompts, or partial metrics claims.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import ItemsView, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Protocol, Self, cast

import torch
from torch import Tensor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.data import DatasetPurpose, RuntimeSample, SVCBenchDataset
from ttt_svcbench_qwen.video_preprocessing import decode_video_causally

_INTEGER_ANSWER = re.compile(r"^[+]?\d+$")


class BaseQwenPredictor(Protocol):
    model_id: str
    model_revision: str
    state_modules_enabled: bool
    ttt_enabled: bool
    prompt_template: str
    generation_parameters: Mapping[str, object]

    def predict(self, sample: RuntimeSample) -> tuple[str, float, int]:
        """Return raw answer text, latency seconds, and peak GPU-memory bytes."""


class ProcessorBatchProtocol(Protocol):
    def to(self, device: torch.device) -> Self: ...

    def __getitem__(self, key: str) -> Tensor: ...

    def items(self) -> ItemsView[str, Tensor]: ...


class ProcessorProtocol(Protocol):
    def apply_chat_template(self, conversation: object, **kwargs: object) -> str: ...

    def __call__(self, **kwargs: object) -> object: ...

    def batch_decode(self, sequences: Tensor, **kwargs: object) -> list[str]: ...


class ModelProtocol(Protocol):
    device: torch.device

    def eval(self) -> Self: ...

    def generate(self, **kwargs: object) -> Tensor: ...


class Qwen3VLA0Predictor:
    """Local-files-only original Qwen3-VL predictor; construction never downloads assets."""

    state_modules_enabled = False
    ttt_enabled = False
    prompt_template = (
        "The video clip ends at the exact moment being asked about. "
        "{question} Answer with ONLY a single integer number, nothing else."
    )

    def __init__(self, config: ProjectConfig, model_root: str | Path) -> None:
        root = Path(model_root)
        if not root.is_dir():
            raise FileNotFoundError(f"local Qwen model root does not exist: {root}")
        self.model_id = config.model.base_model
        self.model_revision = config.model.revision
        self.generation_parameters: dict[str, object] = {
            "do_sample": False,
            "max_new_tokens": 16,
        }
        self._sample_fps = config.video_preprocessing.sample_fps
        processor = AutoProcessor.from_pretrained(root, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            root,
            local_files_only=True,
            dtype="auto",
            device_map="auto",
        )
        self._processor = cast(ProcessorProtocol, processor)
        self._model = cast(ModelProtocol, model)
        self._model.eval()

    def predict(self, sample: RuntimeSample) -> tuple[str, float, int]:
        decoded = decode_video_causally(
            sample.model_input.video,
            query_time=sample.model_input.query_time,
            sample_fps=self._sample_fps,
        )
        if decoded.frames.shape[0] < 2:
            raise ValueError("A0 requires at least one complete temporal tubelet")
        prompt = self.prompt_template.format(question=sample.model_input.question)
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": decoded.frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        raw_inputs = self._processor(
            text=[text],
            videos=[decoded.frames],
            do_sample_frames=False,
            padding=True,
            return_tensors="pt",
        )
        inputs = cast(ProcessorBatchProtocol, raw_inputs).to(self._model.device)
        prompt_length = inputs["input_ids"].shape[1]
        model_inputs = dict(inputs.items())
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        generated = self._model.generate(**model_inputs, **self.generation_parameters)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        answer_ids = generated[:, prompt_length:]
        answers = self._processor.batch_decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if len(answers) != 1:
            raise RuntimeError("A0 predictor expected exactly one decoded answer")
        return answers[0], latency, peak_memory


@dataclass(frozen=True, slots=True)
class A0Prediction:
    query_id: str
    trajectory_id: str
    video_id: str
    query_time: float
    raw_answer: str
    predicted_count: int | None
    ground_truth_count: int
    expected_answer: str
    latency_seconds: float
    peak_gpu_memory_bytes: int

    def __post_init__(self) -> None:
        if self.ground_truth_count < 0 or self.latency_seconds < 0.0:
            raise ValueError("A0 ground truth and latency must be non-negative")
        if self.peak_gpu_memory_bytes < 0:
            raise ValueError("A0 peak GPU memory must be non-negative")


@dataclass(frozen=True, slots=True)
class A0Metrics:
    query_count: int
    valid_prediction_count: int
    valid_prediction_rate: float
    exact_count_accuracy: float
    count_mae_on_valid_predictions: float | None
    answer_accuracy: float
    mean_latency_seconds: float
    median_latency_seconds: float
    peak_gpu_memory_bytes: int


@dataclass(frozen=True, slots=True)
class A0RunReport:
    model_id: str
    model_revision: str
    state_modules_enabled: bool
    ttt_enabled: bool
    prompt_template: str
    generation_parameters: dict[str, object]
    metrics: A0Metrics
    predictions: tuple[A0Prediction, ...]

    def __post_init__(self) -> None:
        if self.state_modules_enabled or self.ttt_enabled:
            raise ValueError("A0 must disable every state module and TTT")


@dataclass(frozen=True, slots=True)
class A0AssetStatus:
    ready: bool
    model_root: str | None
    svcbench_root: str | None
    missing: tuple[str, ...]


def parse_integer_answer(raw_answer: str) -> int | None:
    stripped = raw_answer.strip()
    if not _INTEGER_ANSWER.fullmatch(stripped):
        return None
    value = int(stripped)
    return value if value >= 0 else None


def run_a0(dataset: SVCBenchDataset, predictor: BaseQwenPredictor) -> A0RunReport:
    if dataset.annotations.purpose is not DatasetPurpose.OFFICIAL_CLEAN_EVALUATION:
        raise PermissionError("A0 requires the frozen official-clean evaluation dataset")
    if predictor.state_modules_enabled or predictor.ttt_enabled:
        raise ValueError("A0 predictor must disable State-TTT modules and updates")
    predictions: list[A0Prediction] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        labels = dataset.supervision_for(index, consumer="evaluator")
        raw_answer, latency_seconds, peak_gpu_memory_bytes = predictor.predict(sample)
        predictions.append(
            A0Prediction(
                query_id=sample.identity.query_id,
                trajectory_id=sample.identity.trajectory_id,
                video_id=sample.identity.video_id,
                query_time=sample.model_input.query_time,
                raw_answer=raw_answer,
                predicted_count=parse_integer_answer(raw_answer),
                ground_truth_count=labels.count,
                expected_answer=labels.answer if labels.answer is not None else str(labels.count),
                latency_seconds=latency_seconds,
                peak_gpu_memory_bytes=peak_gpu_memory_bytes,
            )
        )
    return A0RunReport(
        model_id=predictor.model_id,
        model_revision=predictor.model_revision,
        state_modules_enabled=predictor.state_modules_enabled,
        ttt_enabled=predictor.ttt_enabled,
        prompt_template=predictor.prompt_template,
        generation_parameters=dict(predictor.generation_parameters),
        metrics=compute_a0_metrics(predictions),
        predictions=tuple(predictions),
    )


def compute_a0_metrics(predictions: Sequence[A0Prediction]) -> A0Metrics:
    if not predictions:
        raise ValueError("A0 metrics require at least one prediction")
    valid = [item for item in predictions if item.predicted_count is not None]
    exact = sum(item.predicted_count == item.ground_truth_count for item in predictions)
    answer_matches = sum(
        item.raw_answer.strip() == item.expected_answer.strip() for item in predictions
    )
    errors = [
        abs(item.predicted_count - item.ground_truth_count)
        for item in valid
        if item.predicted_count is not None
    ]
    latencies = [item.latency_seconds for item in predictions]
    query_count = len(predictions)
    return A0Metrics(
        query_count=query_count,
        valid_prediction_count=len(valid),
        valid_prediction_rate=len(valid) / query_count,
        exact_count_accuracy=exact / query_count,
        count_mae_on_valid_predictions=mean(errors) if errors else None,
        answer_accuracy=answer_matches / query_count,
        mean_latency_seconds=mean(latencies),
        median_latency_seconds=median(latencies),
        peak_gpu_memory_bytes=max(item.peak_gpu_memory_bytes for item in predictions),
    )


def write_a0_report(report: A0RunReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def audit_a0_assets(
    config: ProjectConfig,
    environment: Mapping[str, str],
) -> A0AssetStatus:
    model_root = environment.get(config.paths.model_root_env)
    svcbench_root = environment.get(config.paths.svcbench_root_env)
    missing: list[str] = []
    if not model_root or not Path(model_root).is_dir():
        missing.append(config.paths.model_root_env)
    if not svcbench_root or not Path(svcbench_root).is_dir():
        missing.append(config.paths.svcbench_root_env)
    elif not (Path(svcbench_root) / config.data.flat_annotation_file).is_file():
        missing.append(f"{config.paths.svcbench_root_env}/{config.data.flat_annotation_file}")
    elif not (Path(svcbench_root) / config.data.video_directory).is_dir():
        missing.append(f"{config.paths.svcbench_root_env}/{config.data.video_directory}")
    return A0AssetStatus(
        ready=not missing,
        model_root=model_root,
        svcbench_root=svcbench_root,
        missing=tuple(missing),
    )
