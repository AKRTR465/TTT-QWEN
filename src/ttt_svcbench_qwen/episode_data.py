"""Build production A2/A5 manifests without exposing supervision to model runtime inputs.

The manifest is a training sidecar.  Runtime construction must still pass through
``RuntimeQueryInput``/``assert_runtime_payload_safe``; labels in this module are consumed only by
the post-forward loss builder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from functools import lru_cache
from itertools import pairwise
from pathlib import Path, PurePosixPath

import av
from torch.utils.data import Dataset, Sampler

from ttt_svcbench_qwen.data import (
    FoldManifest,
    LoadedAnnotations,
    RuntimeQueryInput,
    SVCBenchRecord,
    assert_runtime_payload_safe,
    create_group_kfold_manifest,
    extract_explicit_time_values,
)
from ttt_svcbench_qwen.json_contract import (
    integer_value,
    number_value,
    object_value,
    string_value,
)
from ttt_svcbench_qwen.query_encoder import Operator, TimeWindowMode
from ttt_svcbench_qwen.visual_cost import (
    EpochBoundaryCostEMA,
    VisualCostRecord,
    load_visual_cost_index,
)


class EpisodeSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


class ChunkRole(StrEnum):
    PREWARM = "prewarm"
    SUPPORT = "support"


@dataclass(frozen=True, slots=True)
class AdaptiveChunkSpec:
    role: ChunkRole
    start_time: float
    end_time: float
    maximum_frames: int = 16
    frame_sampling: str = "uniform"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.start_time)
            or not math.isfinite(self.end_time)
            or self.start_time < 0.0
            or self.end_time <= self.start_time
        ):
            raise ValueError("adaptive chunk times must be finite with 0 <= start < end")
        if (
            type(self.maximum_frames) is not int
            or self.maximum_frames < 2
            or self.maximum_frames > 16
            or self.maximum_frames % 2
            or self.frame_sampling != "uniform"
        ):
            raise ValueError("production adaptive chunks use even uniform frame caps in [2, 16]")


@dataclass(frozen=True, slots=True)
class AnswerSupervisionSidecar:
    """Answer-only training label kept outside :class:`RuntimeQueryInput`."""

    query_id: str
    answer: str | None
    provenance: str

    def __post_init__(self) -> None:
        if not self.query_id:
            raise ValueError("answer supervision requires a Query ID")
        expected = "official_explicit" if self.answer is not None else "missing"
        if self.provenance != expected:
            raise ValueError("answer provenance must exactly reflect label availability")


@dataclass(frozen=True, slots=True)
class WeakQuerySidecar:
    """Label-only metadata; this object is forbidden from every runtime model payload."""

    query_id: str
    query_index: int
    query_time: float
    count: int
    counting_type: str
    counting_subtype: str
    operator: str
    time_mode: str
    occurrence_points: tuple[float, ...]
    occurrence_intervals: tuple[tuple[float, float], ...]
    provenance: str = "official_weak"

    def __post_init__(self) -> None:
        if not self.query_id or self.query_index < 0 or self.count < 0:
            raise ValueError("weak Query sidecar identity/count is invalid")
        if not math.isfinite(self.query_time) or self.query_time < 0.0:
            raise ValueError("weak Query time must be finite and non-negative")
        supported = {
            operator.value for operator in Operator if operator is not Operator.UNSUPPORTED
        }
        if self.operator not in supported:
            raise ValueError("weak Query operator must be one of the eight supported operators")
        if self.time_mode not in {mode.value for mode in TimeWindowMode}:
            raise ValueError("weak Query time mode is invalid")
        if self.provenance != "official_weak":
            raise ValueError("production weak sidecars require official_weak provenance")


@dataclass(frozen=True, slots=True)
class ProductionQueryRecord:
    """One runtime Query plus two loss-only sidecars with aligned identity."""

    runtime: RuntimeQueryInput
    answer: AnswerSupervisionSidecar
    weak: WeakQuerySidecar

    def __post_init__(self) -> None:
        ids = (self.runtime.query_id, self.answer.query_id, self.weak.query_id)
        if len(set(ids)) != 1:
            raise ValueError("runtime/answer/weak Query identities must align")
        if self.runtime.query_index != self.weak.query_index:
            raise ValueError("runtime and weak Query indices must align")
        if self.runtime.query_time != self.weak.query_time:
            raise ValueError("runtime and weak Query times must align")


@dataclass(frozen=True, slots=True)
class A2QueryRecord:
    source_dataset: str
    relative_video_path: str
    video_id: str
    trajectory_id: str
    split: EpisodeSplit
    task_class: str
    query: ProductionQueryRecord
    sampling_weight: float

    def __post_init__(self) -> None:
        if not self.video_id or not self.trajectory_id or not self.relative_video_path:
            raise ValueError("A2 Query ownership/path fields must be non-empty")
        if not math.isfinite(self.sampling_weight) or self.sampling_weight <= 0.0:
            raise ValueError("A2 sampling weight must be positive and finite")
        assert_runtime_payload_safe(
            self.query.runtime.as_payload(),
            layer="A2 manifest runtime",
        )


@dataclass(frozen=True, slots=True)
class A5EpisodeRecord:
    episode_id: str
    source_dataset: str
    relative_video_path: str
    video_id: str
    trajectory_id: str
    split: EpisodeSplit
    task_class: str
    operator: str
    prewarm: AdaptiveChunkSpec
    supports: tuple[AdaptiveChunkSpec, ...]
    queries: tuple[ProductionQueryRecord, ...]
    support_count: int
    query_count: int
    truncation_horizon: int
    tbptt_segment_count: int
    sampling_weight: float
    loss_weight: float = 1.0
    padding_source_episode_id: str | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or not self.video_id or not self.trajectory_id:
            raise ValueError("episode identity fields must be non-empty")
        if self.prewarm.role is not ChunkRole.PREWARM:
            raise ValueError("A5 episode must contain one explicit prewarm chunk")
        if not self.supports or any(chunk.role is not ChunkRole.SUPPORT for chunk in self.supports):
            raise ValueError("A5 episode must contain typed Support chunks")
        if self.support_count != len(self.supports):
            raise ValueError("A5 support_count does not match its chunk list")
        if self.query_count != len(self.queries) or self.query_count < 2:
            raise ValueError("A5 production episodes require all of at least two Queries")
        if self.tbptt_segment_count != math.ceil(self.support_count / self.truncation_horizon):
            raise ValueError("A5 tbptt_segment_count must equal ceil(T/K)")
        if self.prewarm.end_time >= self.supports[0].end_time:
            raise ValueError("A5 prewarm must complete before the first Support end")
        support_ends = tuple(chunk.end_time for chunk in self.supports)
        if any(right <= left for left, right in pairwise(support_ends)):
            raise ValueError("A5 Support ends must advance strictly")
        query_times = tuple(query.runtime.query_time for query in self.queries)
        if any(right <= left for left, right in pairwise(query_times)):
            raise ValueError("A5 Query points must advance strictly")
        if support_ends[-1] > query_times[0]:
            raise ValueError("A5 Support may not extend beyond the first Query")
        if not math.isfinite(self.sampling_weight) or self.sampling_weight <= 0.0:
            raise ValueError("A5 sampling weight must be positive and finite")
        if self.loss_weight not in (0.0, 1.0):
            raise ValueError("A5 loss_weight must be exactly zero for padding or one for data")
        is_padding = self.padding_source_episode_id is not None
        if is_padding != (self.loss_weight == 0.0):
            raise ValueError("deterministic padding episodes must be the only zero-weight rows")


@dataclass(frozen=True, slots=True)
class SegmentBucket:
    split: EpisodeSplit
    tbptt_segment_count: int
    episode_ids: tuple[str, ...]
    loss_weights: tuple[float, ...]
    world_size: int

    def __post_init__(self) -> None:
        if self.tbptt_segment_count <= 0 or self.world_size <= 0:
            raise ValueError("segment bucket dimensions must be positive")
        if not self.episode_ids or len(self.episode_ids) % self.world_size:
            raise ValueError("segment buckets must be rank-aligned to world_size")
        if len(self.loss_weights) != len(self.episode_ids):
            raise ValueError("segment bucket weights must align to episodes")


@dataclass(frozen=True, slots=True)
class EpisodeFailure:
    query_id: str
    video_id: str
    source_dataset: str
    query_time: float
    video_duration: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class ProductionEpisodeManifest:
    schema_version: str
    dataset_name: str
    dataset_revision: str
    annotation_sha256: str
    fold_index: int
    seed: int
    train_fraction: float
    group_key: str
    maximum_query_span_seconds: float
    minimum_query_points: int
    truncation_horizon: int
    a2_queries: tuple[A2QueryRecord, ...]
    episodes: tuple[A5EpisodeRecord, ...]
    buckets: tuple[SegmentBucket, ...]
    task_query_counts: tuple[tuple[str, int], ...]
    failures: tuple[EpisodeFailure, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "svcbench_a2_a5_v2":
            raise ValueError("unknown production episode manifest schema")
        train_videos = {
            episode.video_id for episode in self.episodes if episode.split is EpisodeSplit.TRAIN
        }
        validation_videos = {
            episode.video_id
            for episode in self.episodes
            if episode.split is EpisodeSplit.VALIDATION
        }
        if train_videos & validation_videos:
            raise ValueError("production episode manifest leaks a video across splits")
        a2_train_videos = {
            query.video_id for query in self.a2_queries if query.split is EpisodeSplit.TRAIN
        }
        a2_validation_videos = {
            query.video_id for query in self.a2_queries if query.split is EpisodeSplit.VALIDATION
        }
        if a2_train_videos & a2_validation_videos:
            raise ValueError("A2 manifest leaks a video across splits")
        if train_videos - a2_train_videos or validation_videos - a2_validation_videos:
            raise ValueError("A2 and A5 manifests disagree on fold ownership")

    @property
    def a2_query_ids(self) -> tuple[str, ...]:
        return tuple(record.query.runtime.query_id for record in self.a2_queries)


class ManifestStage(StrEnum):
    A2 = "a2"
    A5 = "a5"


type ManifestRecord = A2QueryRecord | A5EpisodeRecord


class ProductionManifestDataset(Dataset[ManifestRecord]):  # type: ignore[misc]
    """Immutable split/stage view used by the LLaMA-Factory runtime bridge."""

    def __init__(
        self,
        manifest: ProductionEpisodeManifest,
        *,
        stage: ManifestStage,
        split: EpisodeSplit,
    ) -> None:
        if not isinstance(manifest, ProductionEpisodeManifest):
            raise TypeError("production dataset requires a validated manifest")
        if stage is ManifestStage.A2:
            records: tuple[ManifestRecord, ...] = tuple(
                record for record in manifest.a2_queries if record.split is split
            )
        else:
            records = tuple(record for record in manifest.episodes if record.split is split)
        if not records:
            raise ValueError(f"manifest contains no {stage.value}/{split.value} records")
        self.manifest = manifest
        self.stage = stage
        self.split = split
        self.records = records
        self.index_by_id = {
            _manifest_record_id(record): index for index, record in enumerate(records)
        }
        if len(self.index_by_id) != len(records):
            raise ValueError("production manifest record IDs must be unique within a dataset")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ManifestRecord:
        return self.records[index]


def load_production_manifest_views(
    manifest_path: str | Path,
    *,
    stage: ManifestStage,
) -> tuple[ProductionManifestDataset, ProductionManifestDataset]:
    """Load the one authoritative manifest and expose immutable train/validation views.

    The LLaMA-Factory bridge owns this operation centrally.  A runtime factory may materialize
    tensors from the records it receives, but it cannot substitute another dataset or split.
    """

    if not isinstance(stage, ManifestStage):
        raise TypeError("production manifest stage must be a ManifestStage")
    manifest = load_production_episode_manifest(manifest_path)
    return (
        ProductionManifestDataset(
            manifest,
            stage=stage,
            split=EpisodeSplit.TRAIN,
        ),
        ProductionManifestDataset(
            manifest,
            stage=stage,
            split=EpisodeSplit.VALIDATION,
        ),
    )


def _a2_visual_length_key(
    record: A2QueryRecord,
    *,
    query_sample_fps: float = 2.0,
    state_query_visual_mode: str = "recent_chunk",
    state_query_max_frames: int = 16,
    answer_query_visual_mode: str = "causal_prefix",
    answer_query_max_frames: int = 256,
) -> tuple[int, int, int, int]:
    """Return a cheap deterministic proxy for visual tokens and decode work.

    The first component is the exact configured upper bound on causal frames at 2 FPS across
    every Support plus the configured Query observation. Source pixel rate and encoded bitrate group
    videos with similar decode cost; the header probe is cached per unique file.  Missing local
    media is valid for manifest-only tests and simply uses zero tie breakers.
    """

    _, supports = adaptive_support_schedule(record.query.runtime.query_time)

    def frames(start: float, end: float, maximum: int = 16) -> int:
        desired = min(maximum, max(2, int(math.floor((end - start) * 2.0))))
        return max(2, desired - desired % 2)

    frame_budget = sum(
        frames(chunk.start_time, chunk.end_time, chunk.maximum_frames) for chunk in supports
    )
    history_write_units = 3 * len(supports) + sum(
        max(1, min(32, frames(chunk.start_time, chunk.end_time, chunk.maximum_frames)))
        for chunk in supports
    )
    query_end = record.query.runtime.query_time
    query_roles = (
        (state_query_visual_mode, state_query_max_frames),
        (answer_query_visual_mode, answer_query_max_frames),
    )
    for mode, maximum in query_roles:
        query_start = 0.0 if mode == "causal_prefix" else max(0.0, query_end - 8.0)
        query_desired = min(
            maximum,
            max(2, int(math.floor((query_end - query_start) * query_sample_fps))),
        )
        frame_budget += max(2, query_desired - query_desired % 2)
    file_bytes = 0
    pixel_rate = 0
    encoded_bytes_per_second = 0
    root = os.environ.get("SVCBENCH_VIDEO_ROOT")
    if root:
        root_path = Path(root).resolve()
        for candidate in (
            (root_path / record.relative_video_path).resolve(),
            (root_path / record.source_dataset / record.relative_video_path).resolve(),
        ):
            if candidate.is_relative_to(root_path) and candidate.is_file():
                file_bytes = candidate.stat().st_size
                pixel_rate, encoded_bytes_per_second = _video_decode_rate(str(candidate))
                break
    # Support count is already a hard sampler bucket.  Within it, frame budget predicts ViT
    # work, decoded source pixels/second predicts short-window CPU work, and encoded bitrate
    # breaks ties for long random-access intervals.  File bytes remain a deterministic fallback
    # when a container cannot expose stream metadata.
    return frame_budget, history_write_units, pixel_rate, encoded_bytes_per_second or file_bytes


@lru_cache(maxsize=1_024)
def _video_decode_rate(path: str) -> tuple[int, int]:
    """Probe one local video header for rank-straggler-aware A2 bucketing."""

    try:
        file_bytes = Path(path).stat().st_size
        with av.open(path) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            fps = float(rate) if rate is not None else 0.0
            width = int(stream.width or 0)
            height = int(stream.height or 0)
            duration = 0.0
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration) / float(av.time_base)
    except (OSError, ValueError, TypeError, IndexError, av.error.FFmpegError):
        return 0, 0
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 0.0
    if not math.isfinite(duration) or duration <= 0.0:
        duration = 0.0
    pixel_rate = int(width * height * fps)
    encoded_rate = int(file_bytes / duration) if duration else 0
    return max(0, pixel_rate), max(0, encoded_rate)


class BalancedA2DistributedSampler(Sampler[int]):  # type: ignore[misc]
    """Build one balanced, visual-length-bucketed stream for Accelerate.

    Consecutive groups of ``world_size`` rows share task class and Support count.  This keeps
    every ZeRO-2 rank on the same differentiable branch and the same number of chunk forwards,
    while the complete stream remains exactly balanced across O1/O2/E1/E2.  Within each branch,
    records are sorted by a causal visual-work proxy before global batches are formed, preventing
    one rank from repeatedly becoming the dynamic-resolution straggler.
    """

    def __init__(
        self,
        dataset: ProductionManifestDataset,
        *,
        rank: int,
        world_size: int,
        seed: int = 42,
        visual_length_fn: Callable[[A2QueryRecord], int] | None = None,
        visual_cost_index: Mapping[str, VisualCostRecord] | None = None,
        query_sample_fps: float = 2.0,
        state_query_visual_mode: str = "recent_chunk",
        state_query_max_frames: int = 16,
        answer_query_visual_mode: str = "causal_prefix",
        answer_query_max_frames: int = 256,
    ) -> None:
        if dataset.stage is not ManifestStage.A2 or dataset.split is not EpisodeSplit.TRAIN:
            raise ValueError("balanced A2 sampling requires the A2 train dataset")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("A2 sampler rank/world_size is invalid")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.visual_cost_index = visual_cost_index or {}
        self.query_sample_fps = query_sample_fps
        self.state_query_visual_mode = state_query_visual_mode
        self.state_query_max_frames = state_query_max_frames
        self.answer_query_visual_mode = answer_query_visual_mode
        self.answer_query_max_frames = answer_query_max_frames
        self.runtime_cost_ema = EpochBoundaryCostEMA(
            {
                record_id: record.predicted_total_seconds
                for record_id, record in self.visual_cost_index.items()
            }
        )
        if self.visual_cost_index:
            missing = {
                _require_a2(record).query.runtime.query_id for record in dataset.records
            } - set(self.visual_cost_index)
            if missing:
                raise ValueError("A2 visual cost index does not cover every manifest record")
        buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, raw in enumerate(dataset.records):
            record = _require_a2(raw)
            support_count = len(adaptive_support_schedule(record.query.runtime.query_time)[1])
            buckets[(record.task_class, support_count)].append(index)
        tasks = {task for task, _ in buckets}
        if tasks != {"O1", "O2", "E1", "E2"}:
            raise ValueError("balanced A2 production sampling requires all four task classes")
        self._buckets = {name: tuple(values) for name, values in buckets.items()}
        self._visual_lengths: dict[int, tuple[float, int, int, int]] = {
            index: self._visual_key(_require_a2(raw), visual_length_fn)
            for index, raw in enumerate(dataset.records)
        }
        if any(any(value < 0 for value in key) for key in self._visual_lengths.values()):
            raise ValueError("A2 visual-length proxy must be non-negative")
        group_counts = {
            task: sum(
                math.ceil(len(values) / world_size)
                for (bucket_task, _), values in self._buckets.items()
                if bucket_task == task
            )
            for task in ("O1", "O2", "E1", "E2")
        }
        self._groups_per_task = max(group_counts.values())
        self._global_size = 4 * self._groups_per_task * world_size

    def _visual_key(
        self,
        record: A2QueryRecord,
        visual_length_fn: Callable[[A2QueryRecord], int] | None,
    ) -> tuple[float, int, int, int]:
        if visual_length_fn is not None:
            return int(visual_length_fn(record)), 0, 0, 0
        sidecar = self.visual_cost_index.get(record.query.runtime.query_id)
        if sidecar is not None:
            if sidecar.support_count != len(
                adaptive_support_schedule(record.query.runtime.query_time)[1]
            ):
                raise ValueError("A2 visual cost Support count disagrees with manifest")
            return sidecar.sort_key
        return _a2_visual_length_key(
            record,
            query_sample_fps=self.query_sample_fps,
            state_query_visual_mode=self.state_query_visual_mode,
            state_query_max_frames=self.state_query_max_frames,
            answer_query_visual_mode=self.answer_query_visual_mode,
            answer_query_max_frames=self.answer_query_max_frames,
        )

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("sampler epoch must be a non-negative integer")
        self.runtime_cost_ema.advance_epoch(epoch)
        for index, raw in enumerate(self.dataset.records):
            record = _require_a2(raw)
            sidecar = self.visual_cost_index.get(record.query.runtime.query_id)
            if sidecar is not None:
                self._visual_lengths[index] = (
                    self.runtime_cost_ema.value(
                        sidecar.record_id,
                        sidecar.predicted_total_seconds,
                    ),
                    sidecar.history_write_units,
                    sidecar.total_visual_tokens,
                    sidecar.maximum_visual_tokens,
                )
        self.epoch = epoch

    def observe_runtime_cost(self, record_id: str, seconds: float) -> None:
        self.runtime_cost_ema.observe(record_id, seconds)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        global_batches: list[tuple[tuple[int, ...], int]] = []
        for task in ("O1", "O2", "E1", "E2"):
            task_batches: list[tuple[tuple[int, ...], int]] = []
            task_buckets = sorted(
                (
                    (support_count, values)
                    for (bucket_task, support_count), values in self._buckets.items()
                    if bucket_task == task
                ),
                key=lambda item: item[0],
            )
            for support_count, values in task_buckets:
                selected = list(values)
                # Shuffle first so equal-length rows do not retain manifest order, then sort into
                # rank-homogeneous global batches.  Global batches are shuffled again below, so
                # this does not introduce a short-to-long curriculum.
                rng.shuffle(selected)
                selected.sort(key=self._visual_lengths.__getitem__)
                remainder = len(selected) % self.world_size
                if remainder:
                    selected.extend([selected[-1]] * (self.world_size - remainder))
                for start in range(0, len(selected), self.world_size):
                    batch = selected[start : start + self.world_size]
                    rng.shuffle(batch)
                    task_batches.append((tuple(batch), support_count))
            rng.shuffle(task_batches)
            if len(task_batches) < self._groups_per_task:
                task_batches.extend(
                    rng.choices(task_batches, k=self._groups_per_task - len(task_batches))
                )
            global_batches.extend(task_batches)
        if os.environ.get("TTT_SMOKE_SHORTEST_FIRST") == "1":
            global_batches.sort(key=lambda item: item[1])
        else:
            rng.shuffle(global_batches)
        global_indices = [index for batch, _ in global_batches for index in batch]
        if len(global_indices) != self._global_size:
            raise RuntimeError("balanced A2 global sampler length drifted")
        return iter(global_indices)

    def __len__(self) -> int:
        return self._global_size


class RankAlignedA5SegmentSampler(Sampler[int]):  # type: ignore[misc]
    """Yield K-segment-homogeneous global batches for Accelerate to shard once.

    Real rows are sampled with replacement according to the manifest's task-balancing
    weights.  The precomputed zero-weight rows remain deterministic padding, so every
    rank executes the same number of backward collectives without changing the loss.
    """

    def __init__(
        self,
        dataset: ProductionManifestDataset,
        *,
        rank: int,
        world_size: int,
        seed: int = 42,
        visual_cost_index: Mapping[str, VisualCostRecord] | None = None,
        query_sample_fps: float = 2.0,
        state_query_visual_mode: str = "recent_chunk",
        state_query_max_frames: int = 16,
        answer_query_visual_mode: str = "causal_prefix",
        answer_query_max_frames: int = 256,
    ) -> None:
        if dataset.stage is not ManifestStage.A5 or dataset.split is not EpisodeSplit.TRAIN:
            raise ValueError("rank-aligned A5 sampling requires the A5 train dataset")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError("A5 sampler rank/world_size is invalid")
        if world_size != 4:
            raise ValueError("production A5 manifest is frozen to four ranks")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.visual_cost_index = visual_cost_index or {}
        self.query_sample_fps = query_sample_fps
        self.state_query_visual_mode = state_query_visual_mode
        self.state_query_max_frames = state_query_max_frames
        self.answer_query_visual_mode = answer_query_visual_mode
        self.answer_query_max_frames = answer_query_max_frames
        self.runtime_cost_ema = EpochBoundaryCostEMA(
            {
                record_id: record.predicted_total_seconds
                for record_id, record in self.visual_cost_index.items()
            }
        )
        self._buckets = tuple(
            bucket for bucket in dataset.manifest.buckets if bucket.split is EpisodeSplit.TRAIN
        )
        if not self._buckets:
            raise ValueError("A5 train manifest contains no segment buckets")
        record_ids = {_manifest_record_id(record) for record in dataset.records}
        if self.visual_cost_index and record_ids - set(self.visual_cost_index):
            raise ValueError("A5 visual cost index does not cover every manifest record")
        for bucket in self._buckets:
            shapes = {
                _a5_alignment_shape(_require_a5(dataset[dataset.index_by_id[episode_id]]))
                for episode_id in bucket.episode_ids
            }
            if len(shapes) != 1:
                raise ValueError("A5 manifest bucket mixes exact segment lengths or Query counts")
        self._global_size = sum(len(bucket.episode_ids) for bucket in self._buckets)

    def set_epoch(self, epoch: int) -> None:
        if type(epoch) is not int or epoch < 0:
            raise ValueError("sampler epoch must be a non-negative integer")
        self.runtime_cost_ema.advance_epoch(epoch)
        self.epoch = epoch

    def observe_runtime_cost(self, record_id: str, seconds: float) -> None:
        self.runtime_cost_ema.observe(record_id, seconds)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        global_batches: list[tuple[int, ...]] = []
        records_by_id = {
            _manifest_record_id(record): _require_a5(record) for record in self.dataset.records
        }
        for bucket in self._buckets:
            real_ids = [
                episode_id
                for episode_id, loss_weight in zip(
                    bucket.episode_ids,
                    bucket.loss_weights,
                    strict=True,
                )
                if loss_weight == 1.0
            ]
            padding_ids = [
                episode_id
                for episode_id, loss_weight in zip(
                    bucket.episode_ids,
                    bucket.loss_weights,
                    strict=True,
                )
                if loss_weight == 0.0
            ]
            weights = [records_by_id[episode_id].sampling_weight for episode_id in real_ids]
            sampled = rng.choices(real_ids, weights=weights, k=len(real_ids))
            scheduled_ids = sampled + padding_ids
            scheduled_ids.sort(key=self._cost_key)
            if len(scheduled_ids) % self.world_size:
                raise RuntimeError("A5 segment bucket lost its rank-aligned padding")
            for start in range(0, len(scheduled_ids), self.world_size):
                group = scheduled_ids[start : start + self.world_size]
                indices = tuple(self.dataset.index_by_id[episode_id] for episode_id in group)
                shapes = {
                    _a5_alignment_shape(_require_a5(self.dataset[index])) for index in indices
                }
                if len(shapes) != 1:
                    raise RuntimeError("A5 sampler mixed segment lengths or Query counts")
                global_batches.append(indices)
        rng.shuffle(global_batches)
        global_indices = [index for batch in global_batches for index in batch]
        if len(global_indices) != self._global_size:
            raise RuntimeError("rank-aligned A5 global sampler length drifted")
        return iter(global_indices)

    def _cost_key(self, episode_id: str) -> tuple[float, int, int, int]:
        record = _require_a5(self.dataset[self.dataset.index_by_id[episode_id]])
        sidecar = self.visual_cost_index.get(episode_id)
        if sidecar is not None:
            if sidecar.support_count != record.support_count:
                raise ValueError("A5 visual cost Support count disagrees with manifest")
            if sidecar.segment_lengths != _a5_segment_lengths(record):
                raise ValueError("A5 visual cost segment lengths disagree with manifest")
            if sidecar.query_count != record.query_count:
                raise ValueError("A5 visual cost Query count disagrees with manifest")
            return (
                self.runtime_cost_ema.value(
                    episode_id,
                    sidecar.predicted_total_seconds,
                ),
                sidecar.history_write_units,
                sidecar.total_visual_tokens,
                sidecar.maximum_visual_tokens,
            )
        support_frames = record.prewarm.maximum_frames + sum(
            chunk.maximum_frames for chunk in record.supports
        )
        query_roles = (
            (self.state_query_visual_mode, self.state_query_max_frames),
            (self.answer_query_visual_mode, self.answer_query_max_frames),
        )
        query_frames = tuple(
            sum(
                _query_visual_frame_budget(
                    query.runtime.query_time,
                    mode=mode,
                    maximum=maximum,
                    sample_fps=self.query_sample_fps,
                )
                for mode, maximum in query_roles
            )
            for query in record.queries
        )
        proxy = support_frames + sum(query_frames)
        history_write_units = record.support_count * 4
        return float(proxy), history_write_units, proxy, max(query_frames)

    def __len__(self) -> int:
        return self._global_size


def build_production_train_sampler(
    dataset: object,
    rank: int,
    world_size: int,
    *,
    visual_cost_index: Mapping[str, VisualCostRecord] | None = None,
    query_sample_fps: float = 2.0,
    state_query_visual_mode: str = "recent_chunk",
    state_query_max_frames: int = 16,
    answer_query_visual_mode: str = "causal_prefix",
    answer_query_max_frames: int = 256,
) -> Sampler[int]:
    """Shared runtime-factory hook for A2 task balance and A5 segment parity."""

    if not isinstance(dataset, ProductionManifestDataset):
        raise TypeError("production sampler requires ProductionManifestDataset")
    if dataset.stage is ManifestStage.A2:
        return BalancedA2DistributedSampler(
            dataset,
            rank=rank,
            world_size=world_size,
            seed=dataset.manifest.seed,
            visual_cost_index=visual_cost_index,
            query_sample_fps=query_sample_fps,
            state_query_visual_mode=state_query_visual_mode,
            state_query_max_frames=state_query_max_frames,
            answer_query_visual_mode=answer_query_visual_mode,
            answer_query_max_frames=answer_query_max_frames,
        )
    return RankAlignedA5SegmentSampler(
        dataset,
        rank=rank,
        world_size=world_size,
        seed=dataset.manifest.seed,
        visual_cost_index=visual_cost_index,
        query_sample_fps=query_sample_fps,
        state_query_visual_mode=state_query_visual_mode,
        state_query_max_frames=state_query_max_frames,
        answer_query_visual_mode=answer_query_visual_mode,
        answer_query_max_frames=answer_query_max_frames,
    )


def _query_visual_frame_budget(
    query_time: float,
    *,
    mode: str,
    maximum: int,
    sample_fps: float,
) -> int:
    if mode not in {"recent_chunk", "causal_prefix"}:
        raise ValueError("Query visual mode is invalid")
    start = 0.0 if mode == "causal_prefix" else max(0.0, query_time - 8.0)
    desired = min(maximum, max(2, int(math.floor((query_time - start) * sample_fps))))
    return max(2, desired - desired % 2)


def _a5_segment_lengths(record: A5EpisodeRecord) -> tuple[int, ...]:
    remaining = record.support_count
    lengths: list[int] = []
    while remaining:
        length = min(record.truncation_horizon, remaining)
        lengths.append(length)
        remaining -= length
    return tuple(lengths)


def _a5_alignment_shape(record: A5EpisodeRecord) -> tuple[tuple[int, ...], int]:
    return _a5_segment_lengths(record), record.query_count


def official_operator(counting_type: str, counting_subtype: str) -> Operator:
    """Map official type/subtype spelling to the exact eight-way operator surface."""

    normalized_type = _normalize_label(counting_type)
    normalized_subtype = _normalize_label(counting_subtype)
    mapping = {
        "o1-snap": Operator.O1_SNAP,
        "o1-delta": Operator.O1_DELTA,
        "o2-unique": Operator.O2_UNIQUE,
        "o2-gain": Operator.O2_GAIN,
        "e1-action": Operator.E1_ACTION,
        "e1-transit": Operator.E1_TRANSIT,
        "e2-periodic": Operator.E2_PERIODIC,
        "e2-episode": Operator.E2_EPISODE,
    }
    operator = mapping.get(normalized_subtype)
    if operator is None:
        raise ValueError(f"unsupported official counting subtype: {counting_subtype!r}")
    if operator.value.split("-", maxsplit=1)[0] != normalized_type:
        raise ValueError("official counting type/subtype disagree")
    return operator


def official_time_mode(record: SVCBenchRecord, operator: Operator) -> TimeWindowMode:
    question = record.question.casefold()
    if any(token in question for token in ("between", " from ", " to ", "从", "到", "至")):
        return TimeWindowMode.EXPLICIT_RANGE
    if any(token in question for token in ("last ", "past ", "recent", "最近", "过去")):
        return TimeWindowMode.RECENT
    if operator is Operator.O1_SNAP or any(
        token in question for token in ("now", "moment", "currently", "此刻", "现在")
    ):
        return TimeWindowMode.NOW
    return TimeWindowMode.HISTORY


def greedy_nonoverlap_query_groups(
    records: Sequence[SVCBenchRecord],
    *,
    maximum_span_seconds: float = 64.0,
    minimum_query_points: int = 2,
) -> tuple[tuple[SVCBenchRecord, ...], ...]:
    """Greedily consume disjoint, maximal Query groups bounded by first-to-last span."""

    if not math.isfinite(maximum_span_seconds) or maximum_span_seconds <= 0.0:
        raise ValueError("maximum Query span must be positive and finite")
    if minimum_query_points < 2:
        raise ValueError("A5 Query groups require at least two points")
    ordered = tuple(sorted(records, key=lambda item: (item.query_time, item.identity.query_index)))
    if not ordered:
        return ()
    owners = {(item.identity.video_id, item.identity.trajectory_id) for item in ordered}
    if len(owners) != 1:
        raise ValueError("greedy Query grouping requires one video trajectory")
    groups: list[tuple[SVCBenchRecord, ...]] = []
    start = 0
    while start < len(ordered):
        stop = start + 1
        while (
            stop < len(ordered)
            and ordered[stop].query_time - ordered[start].query_time <= maximum_span_seconds
        ):
            stop += 1
        group = ordered[start:stop]
        if len(group) >= minimum_query_points:
            groups.append(group)
        start = stop
    return tuple(groups)


def adaptive_support_schedule(
    first_query_time: float,
    *,
    recent_seconds: float = 40.0,
    recent_window_seconds: float = 8.0,
    recent_stride_seconds: float = 4.0,
    overlap_seconds: float = 4.0,
) -> tuple[AdaptiveChunkSpec, tuple[AdaptiveChunkSpec, ...]]:
    """Cover [0, first Query] using recent fine windows and older geometric windows."""

    if not math.isfinite(first_query_time) or first_query_time <= 0.0:
        raise ValueError("the first A5 Query must occur after video time zero")
    if (recent_seconds, recent_window_seconds, recent_stride_seconds, overlap_seconds) != (
        40.0,
        8.0,
        4.0,
        4.0,
    ):
        raise ValueError("production adaptive schedule is frozen at 40/8/4 with 4s overlap")

    # Reserve a causal current-Query observation after the final Support.  That Query chunk is
    # evaluated with W_after and is deliberately not an Inner-SGD Support update.  Its recent
    # window overlaps the final Support by four seconds, so the union still covers the complete
    # prefix while MetaTTTEpisode can enforce Query.end > final Support.end.
    query_observation_gap = min(4.0, first_query_time / 2.0)
    causal_end = math.nextafter(first_query_time - query_observation_gap, 0.0)
    if causal_end <= 0.0:
        raise ValueError("the first A5 Query is too early to form Support and Query chunks")
    recent_start = max(0.0, causal_end - recent_seconds)
    recent: list[tuple[float, float]] = []
    if causal_end - recent_start <= recent_window_seconds:
        recent.append((recent_start, causal_end))
    else:
        start = recent_start
        while start + recent_window_seconds < causal_end:
            recent.append((start, start + recent_window_seconds))
            start += recent_stride_seconds
        final = (causal_end - recent_window_seconds, causal_end)
        if not recent or final != recent[-1]:
            recent.append(final)

    older_reverse: list[tuple[float, float]] = []
    boundary = recent_start
    width = 16.0
    while boundary > 0.0:
        end = min(causal_end, boundary + overlap_seconds)
        start = max(0.0, end - width)
        older_reverse.append((start, end))
        if start == 0.0:
            break
        boundary = start
        width *= 2.0
    intervals = tuple(reversed(older_reverse)) + tuple(recent)
    supports = tuple(
        AdaptiveChunkSpec(
            ChunkRole.SUPPORT,
            start,
            end,
            maximum_frames=_adaptive_support_frame_cap(start, end),
        )
        for start, end in intervals
    )
    if not supports:
        raise RuntimeError("adaptive support schedule unexpectedly produced no chunks")
    for left, right in pairwise(supports):
        if left.end_time - right.start_time + 1.0e-9 < overlap_seconds:
            raise ValueError("adjacent adaptive Support chunks must overlap by at least 4 seconds")
    first = supports[0]
    prewarm_end = min(first.start_time + 8.0, first.end_time - 1.0e-6)
    if prewarm_end <= first.start_time:
        prewarm_end = first.start_time + (first.end_time - first.start_time) / 2.0
    prewarm = AdaptiveChunkSpec(ChunkRole.PREWARM, first.start_time, prewarm_end)
    return prewarm, supports


def _adaptive_support_frame_cap(start_time: float, end_time: float) -> int:
    """Keep 2 FPS detail near the Query and halve sparse geometric-history work."""

    return 16 if end_time - start_time <= 8.0 + 1.0e-6 else 8


def build_production_episode_manifest(
    annotations: LoadedAnnotations,
    *,
    video_durations: Mapping[str, float],
    runtime_video_paths: Mapping[str, str] | None = None,
    fold_index: int = 0,
    seed: int = 42,
    n_splits: int = 5,
    truncation_horizon: int = 8,
    world_size: int = 4,
    video_duration_tolerance: float = 1.0,
) -> ProductionEpisodeManifest:
    """Build the fold0 A2/A5 sidecar and deterministic ZeRO-2 segment buckets."""

    if fold_index != 0 or seed != 42 or n_splits != 5:
        raise ValueError("production split is frozen to fold0, seed=42, five folds (80/20)")
    if truncation_horizon <= 0 or world_size <= 0:
        raise ValueError("truncation horizon/world size must be positive")
    if not math.isfinite(video_duration_tolerance) or video_duration_tolerance < 0.0:
        raise ValueError("video duration tolerance must be finite and non-negative")
    if runtime_video_paths is not None:
        expected_query_ids = {record.identity.query_id for record in annotations.records}
        if set(runtime_video_paths) != expected_query_ids:
            missing = tuple(sorted(expected_query_ids - set(runtime_video_paths)))
            unexpected = tuple(sorted(set(runtime_video_paths) - expected_query_ids))
            raise ValueError(
                "runtime video mapping must cover every annotation Query exactly once: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        for query_id, relative_path in runtime_video_paths.items():
            path = PurePosixPath(relative_path)
            if not query_id or not relative_path or path.is_absolute() or ".." in path.parts:
                raise ValueError("runtime video paths must be safe non-empty relative paths")
    fold_manifest = create_group_kfold_manifest(annotations, n_splits=n_splits, seed=seed)
    split_by_video = _split_map(fold_manifest, fold_index)
    valid_records: list[SVCBenchRecord] = []
    failures: list[EpisodeFailure] = []
    for record in annotations.records:
        duration = _duration_for(record, video_durations)
        if duration is None:
            raise ValueError(f"missing video duration for {record.identity.video_id}")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(f"invalid video duration for {record.identity.video_id}")
        if record.query_time > duration + video_duration_tolerance:
            failures.append(
                EpisodeFailure(
                    query_id=record.identity.query_id,
                    video_id=record.identity.video_id,
                    source_dataset=record.source_dataset,
                    query_time=record.query_time,
                    video_duration=duration,
                    reason="query_time_exceeds_video_duration",
                )
            )
        else:
            valid_records.append(record)

    by_trajectory: dict[tuple[str, str], list[SVCBenchRecord]] = defaultdict(list)
    for record in valid_records:
        by_trajectory[(record.identity.video_id, record.identity.trajectory_id)].append(record)
    raw_episodes: list[A5EpisodeRecord] = []
    for key in sorted(by_trajectory):
        groups = greedy_nonoverlap_query_groups(by_trajectory[key])
        for group_index, group in enumerate(groups):
            raw_episodes.append(
                _episode_from_group(
                    group,
                    group_index=group_index,
                    split=split_by_video[group[0].identity.video_id],
                    truncation_horizon=truncation_horizon,
                    runtime_video_paths=runtime_video_paths,
                )
            )

    task_counts = Counter(record.labels.counting_type.upper() for record in valid_records)
    a2_queries = tuple(
        _a2_query_from_record(
            record,
            split=split_by_video[record.identity.video_id],
            task_query_count=task_counts[record.labels.counting_type.upper()],
            runtime_video_path=(
                None
                if runtime_video_paths is None
                else runtime_video_paths[record.identity.query_id]
            ),
        )
        for record in valid_records
    )
    episodes = tuple(
        _with_sampling_weight(episode, task_counts[episode.task_class]) for episode in raw_episodes
    )
    buckets, padding = _build_segment_buckets(episodes, world_size=world_size)
    all_episodes = episodes + padding
    return ProductionEpisodeManifest(
        schema_version="svcbench_a2_a5_v2",
        dataset_name=annotations.source.name,
        dataset_revision=annotations.source.revision,
        annotation_sha256=annotations.annotation_sha256,
        fold_index=fold_index,
        seed=seed,
        train_fraction=0.8,
        group_key="source_dataset/video_path",
        maximum_query_span_seconds=64.0,
        minimum_query_points=2,
        truncation_horizon=truncation_horizon,
        a2_queries=a2_queries,
        episodes=all_episodes,
        buckets=buckets,
        task_query_counts=tuple(sorted(task_counts.items())),
        failures=tuple(failures),
    )


def write_production_episode_manifest(
    manifest: ProductionEpisodeManifest,
    *,
    manifest_path: str | Path,
    failed_path: str | Path,
) -> None:
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    failure_destination = Path(failed_path)
    failure_destination.parent.mkdir(parents=True, exist_ok=True)
    failure_destination.write_text(
        "".join(
            json.dumps(asdict(failure), ensure_ascii=False) + "\n" for failure in manifest.failures
        ),
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"production manifest cannot serialize {type(value).__name__}")


def load_production_episode_manifest(path: str | Path) -> ProductionEpisodeManifest:
    """Load and fully revalidate a serialized production manifest."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("production episode manifest must contain one JSON object")
    values = raw
    _require_exact_keys(
        values,
        {
            "schema_version",
            "dataset_name",
            "dataset_revision",
            "annotation_sha256",
            "fold_index",
            "seed",
            "train_fraction",
            "group_key",
            "maximum_query_span_seconds",
            "minimum_query_points",
            "truncation_horizon",
            "a2_queries",
            "episodes",
            "buckets",
            "task_query_counts",
            "failures",
        },
        "production episode manifest",
    )
    a2_queries = tuple(_parse_a2_query(item) for item in _object_list(values, "a2_queries"))
    episodes = tuple(_parse_a5_episode(item) for item in _object_list(values, "episodes"))
    buckets = tuple(_parse_segment_bucket(item) for item in _object_list(values, "buckets"))
    failures = tuple(_parse_failure(item) for item in _object_list(values, "failures"))
    task_counts = tuple(
        _task_count_pair(item) for item in _sequence_list(values, "task_query_counts")
    )
    return ProductionEpisodeManifest(
        schema_version=string_value(values, "schema_version"),
        dataset_name=string_value(values, "dataset_name"),
        dataset_revision=string_value(values, "dataset_revision"),
        annotation_sha256=string_value(values, "annotation_sha256"),
        fold_index=integer_value(values, "fold_index"),
        seed=integer_value(values, "seed"),
        train_fraction=_float_value(values, "train_fraction"),
        group_key=string_value(values, "group_key"),
        maximum_query_span_seconds=_float_value(values, "maximum_query_span_seconds"),
        minimum_query_points=integer_value(values, "minimum_query_points"),
        truncation_horizon=integer_value(values, "truncation_horizon"),
        a2_queries=a2_queries,
        episodes=episodes,
        buckets=buckets,
        task_query_counts=task_counts,
        failures=failures,
    )


def _episode_from_group(
    group: tuple[SVCBenchRecord, ...],
    *,
    group_index: int,
    split: EpisodeSplit,
    truncation_horizon: int,
    runtime_video_paths: Mapping[str, str] | None,
) -> A5EpisodeRecord:
    first = group[0]
    operator = official_operator(first.labels.counting_type, first.labels.counting_subtype)
    if any(
        official_operator(item.labels.counting_type, item.labels.counting_subtype) is not operator
        for item in group
    ):
        raise ValueError("one trajectory Query group cannot mix official operators")
    prewarm, supports = adaptive_support_schedule(first.query_time)
    query_records = tuple(_production_query(item, operator) for item in group)
    digest = hashlib.sha256(
        "|".join(item.identity.query_id for item in group).encode("utf-8")
    ).hexdigest()[:12]
    episode_id = f"{first.identity.trajectory_id}-g{group_index:03d}-{digest}"
    return A5EpisodeRecord(
        episode_id=episode_id,
        source_dataset=first.source_dataset,
        # The remote SVCBench conversion stores one causal video per Query.  The
        # final Query clip is the only one guaranteed to contain every earlier
        # Support and every Query point in this episode.
        relative_video_path=(
            first.relative_video_path
            if runtime_video_paths is None
            else runtime_video_paths[group[-1].identity.query_id]
        ),
        video_id=first.identity.video_id,
        trajectory_id=first.identity.trajectory_id,
        split=split,
        task_class=first.labels.counting_type.upper(),
        operator=operator.value,
        prewarm=prewarm,
        supports=supports,
        queries=query_records,
        support_count=len(supports),
        query_count=len(query_records),
        truncation_horizon=truncation_horizon,
        tbptt_segment_count=math.ceil(len(supports) / truncation_horizon),
        sampling_weight=1.0,
    )


def _production_query(record: SVCBenchRecord, operator: Operator) -> ProductionQueryRecord:
    runtime = RuntimeQueryInput(
        video_id=record.identity.video_id,
        trajectory_id=record.identity.trajectory_id,
        query_id=record.identity.query_id,
        query_index=record.identity.query_index,
        video=Path(record.relative_video_path),
        question=record.question,
        query_time=record.query_time,
        explicit_time_values=extract_explicit_time_values(record.question),
    )
    answer = AnswerSupervisionSidecar(
        query_id=record.identity.query_id,
        answer=record.labels.answer,
        provenance=("official_explicit" if record.labels.answer is not None else "missing"),
    )
    occurrence = record.labels.occurrence_times
    weak = WeakQuerySidecar(
        query_id=record.identity.query_id,
        query_index=record.identity.query_index,
        query_time=record.query_time,
        count=record.labels.count,
        counting_type=record.labels.counting_type,
        counting_subtype=record.labels.counting_subtype,
        operator=operator.value,
        time_mode=official_time_mode(record, operator).value,
        occurrence_points=occurrence.points,
        occurrence_intervals=tuple(zip(occurrence.starts, occurrence.ends, strict=True)),
    )
    return ProductionQueryRecord(runtime=runtime, answer=answer, weak=weak)


def _a2_query_from_record(
    record: SVCBenchRecord,
    *,
    split: EpisodeSplit,
    task_query_count: int,
    runtime_video_path: str | None,
) -> A2QueryRecord:
    if task_query_count <= 0:
        raise ValueError("A2 task query count must be positive")
    operator = official_operator(record.labels.counting_type, record.labels.counting_subtype)
    return A2QueryRecord(
        source_dataset=record.source_dataset,
        relative_video_path=runtime_video_path or record.relative_video_path,
        video_id=record.identity.video_id,
        trajectory_id=record.identity.trajectory_id,
        split=split,
        task_class=record.labels.counting_type.upper(),
        query=_production_query(record, operator),
        sampling_weight=1.0 / task_query_count,
    )


def _with_sampling_weight(episode: A5EpisodeRecord, task_query_count: int) -> A5EpisodeRecord:
    if task_query_count <= 0:
        raise ValueError("A5 task query count must be positive")
    return replace(
        episode,
        sampling_weight=episode.query_count / task_query_count,
    )


def _build_segment_buckets(
    episodes: tuple[A5EpisodeRecord, ...],
    *,
    world_size: int,
) -> tuple[tuple[SegmentBucket, ...], tuple[A5EpisodeRecord, ...]]:
    grouped: dict[
        tuple[EpisodeSplit, tuple[int, ...], int],
        list[A5EpisodeRecord],
    ] = defaultdict(list)
    for episode in episodes:
        grouped[(episode.split, _a5_segment_lengths(episode), episode.query_count)].append(episode)
    buckets: list[SegmentBucket] = []
    padding_records: list[A5EpisodeRecord] = []
    for key in sorted(grouped, key=lambda item: (item[0].value, item[1], item[2])):
        rows = sorted(grouped[key], key=lambda item: item.episode_id)
        remainder = len(rows) % world_size
        if remainder:
            source = rows[-1]
            for padding_index in range(world_size - remainder):
                padded = A5EpisodeRecord(
                    episode_id=f"{source.episode_id}-pad{padding_index:02d}",
                    source_dataset=source.source_dataset,
                    relative_video_path=source.relative_video_path,
                    video_id=source.video_id,
                    trajectory_id=source.trajectory_id,
                    split=source.split,
                    task_class=source.task_class,
                    operator=source.operator,
                    prewarm=source.prewarm,
                    supports=source.supports,
                    queries=source.queries,
                    support_count=source.support_count,
                    query_count=source.query_count,
                    truncation_horizon=source.truncation_horizon,
                    tbptt_segment_count=source.tbptt_segment_count,
                    sampling_weight=source.sampling_weight,
                    loss_weight=0.0,
                    padding_source_episode_id=source.episode_id,
                )
                rows.append(padded)
                padding_records.append(padded)
        buckets.append(
            SegmentBucket(
                split=key[0],
                tbptt_segment_count=len(key[1]),
                episode_ids=tuple(row.episode_id for row in rows),
                loss_weights=tuple(row.loss_weight for row in rows),
                world_size=world_size,
            )
        )
    return tuple(buckets), tuple(padding_records)


def _split_map(folds: FoldManifest, fold_index: int) -> dict[str, EpisodeSplit]:
    fold = folds.folds[fold_index]
    result = {video_id: EpisodeSplit.TRAIN for video_id in fold.train_video_ids}
    result.update({video_id: EpisodeSplit.VALIDATION for video_id in fold.validation_video_ids})
    return result


def _duration_for(record: SVCBenchRecord, durations: Mapping[str, float]) -> float | None:
    for key in (
        record.identity.video_id,
        f"{record.source_dataset}/{record.relative_video_path}",
        record.relative_video_path,
    ):
        value = durations.get(key)
        if value is not None:
            return float(value)
    return None


def _normalize_label(value: str) -> str:
    return "-".join(value.strip().casefold().replace("_", "-").split())


def _parse_chunk(value: object) -> AdaptiveChunkSpec:
    row = object_value(value, "adaptive chunk")
    _require_exact_keys(
        row,
        {"role", "start_time", "end_time", "maximum_frames", "frame_sampling"},
        "adaptive chunk",
    )
    return AdaptiveChunkSpec(
        role=ChunkRole(string_value(row, "role")),
        start_time=_float_value(row, "start_time"),
        end_time=_float_value(row, "end_time"),
        maximum_frames=integer_value(row, "maximum_frames"),
        frame_sampling=string_value(row, "frame_sampling"),
    )


def _parse_runtime_query(value: object) -> RuntimeQueryInput:
    row = object_value(value, "runtime Query")
    _require_exact_keys(
        row,
        {
            "video_id",
            "trajectory_id",
            "query_id",
            "query_index",
            "video",
            "question",
            "query_time",
            "explicit_time_values",
            "episode_nonce",
        },
        "runtime Query",
    )
    explicit = _number_list(row, "explicit_time_values")
    return RuntimeQueryInput(
        video_id=string_value(row, "video_id"),
        trajectory_id=string_value(row, "trajectory_id"),
        query_id=string_value(row, "query_id"),
        query_index=integer_value(row, "query_index"),
        video=Path(string_value(row, "video")),
        question=string_value(row, "question"),
        query_time=_float_value(row, "query_time"),
        explicit_time_values=explicit,
        episode_nonce=integer_value(row, "episode_nonce"),
    )


def _parse_answer_sidecar(value: object) -> AnswerSupervisionSidecar:
    row = object_value(value, "answer sidecar")
    _require_exact_keys(row, {"query_id", "answer", "provenance"}, "answer sidecar")
    answer = row.get("answer")
    if answer is not None and not isinstance(answer, str):
        raise ValueError("answer sidecar answer must be string or null")
    return AnswerSupervisionSidecar(
        query_id=string_value(row, "query_id"),
        answer=answer,
        provenance=string_value(row, "provenance"),
    )


def _parse_weak_sidecar(value: object) -> WeakQuerySidecar:
    row = object_value(value, "weak sidecar")
    _require_exact_keys(
        row,
        {
            "query_id",
            "query_index",
            "query_time",
            "count",
            "counting_type",
            "counting_subtype",
            "operator",
            "time_mode",
            "occurrence_points",
            "occurrence_intervals",
            "provenance",
        },
        "weak sidecar",
    )
    intervals = tuple(
        (_pair[0], _pair[1])
        for _pair in (
            _number_pair(item, "weak occurrence interval")
            for item in _sequence_list(row, "occurrence_intervals")
        )
    )
    return WeakQuerySidecar(
        query_id=string_value(row, "query_id"),
        query_index=integer_value(row, "query_index"),
        query_time=_float_value(row, "query_time"),
        count=integer_value(row, "count"),
        counting_type=string_value(row, "counting_type"),
        counting_subtype=string_value(row, "counting_subtype"),
        operator=string_value(row, "operator"),
        time_mode=string_value(row, "time_mode"),
        occurrence_points=_number_list(row, "occurrence_points"),
        occurrence_intervals=intervals,
        provenance=string_value(row, "provenance"),
    )


def _parse_production_query(value: object) -> ProductionQueryRecord:
    row = object_value(value, "production Query")
    _require_exact_keys(row, {"runtime", "answer", "weak"}, "production Query")
    return ProductionQueryRecord(
        runtime=_parse_runtime_query(row["runtime"]),
        answer=_parse_answer_sidecar(row["answer"]),
        weak=_parse_weak_sidecar(row["weak"]),
    )


def _parse_a2_query(value: object) -> A2QueryRecord:
    row = object_value(value, "A2 Query")
    required = {
            "source_dataset",
            "relative_video_path",
            "video_id",
            "trajectory_id",
            "split",
            "task_class",
            "query",
            "sampling_weight",
    }
    _require_exact_keys(row, required, "A2 Query")
    return A2QueryRecord(
        source_dataset=string_value(row, "source_dataset"),
        relative_video_path=string_value(row, "relative_video_path"),
        video_id=string_value(row, "video_id"),
        trajectory_id=string_value(row, "trajectory_id"),
        split=EpisodeSplit(string_value(row, "split")),
        task_class=string_value(row, "task_class"),
        query=_parse_production_query(row["query"]),
        sampling_weight=_float_value(row, "sampling_weight"),
    )


def _parse_a5_episode(value: object) -> A5EpisodeRecord:
    row = object_value(value, "A5 episode")
    required = {
        "episode_id",
        "source_dataset",
        "relative_video_path",
        "video_id",
        "trajectory_id",
        "split",
        "task_class",
        "operator",
        "prewarm",
        "supports",
        "queries",
        "support_count",
        "query_count",
        "truncation_horizon",
        "tbptt_segment_count",
        "sampling_weight",
        "loss_weight",
        "padding_source_episode_id",
    }
    _require_exact_keys(row, required, "A5 episode")
    padding_source = row.get("padding_source_episode_id")
    if padding_source is not None and not isinstance(padding_source, str):
        raise ValueError("A5 padding source must be string or null")
    return A5EpisodeRecord(
        episode_id=string_value(row, "episode_id"),
        source_dataset=string_value(row, "source_dataset"),
        relative_video_path=string_value(row, "relative_video_path"),
        video_id=string_value(row, "video_id"),
        trajectory_id=string_value(row, "trajectory_id"),
        split=EpisodeSplit(string_value(row, "split")),
        task_class=string_value(row, "task_class"),
        operator=string_value(row, "operator"),
        prewarm=_parse_chunk(row["prewarm"]),
        supports=tuple(_parse_chunk(item) for item in _object_list(row, "supports")),
        queries=tuple(_parse_production_query(item) for item in _object_list(row, "queries")),
        support_count=integer_value(row, "support_count"),
        query_count=integer_value(row, "query_count"),
        truncation_horizon=integer_value(row, "truncation_horizon"),
        tbptt_segment_count=integer_value(row, "tbptt_segment_count"),
        sampling_weight=_float_value(row, "sampling_weight"),
        loss_weight=_float_value(row, "loss_weight"),
        padding_source_episode_id=padding_source,
    )


def _parse_segment_bucket(value: object) -> SegmentBucket:
    row = object_value(value, "segment bucket")
    _require_exact_keys(
        row,
        {"split", "tbptt_segment_count", "episode_ids", "loss_weights", "world_size"},
        "segment bucket",
    )
    episode_ids_raw = row.get("episode_ids")
    if not isinstance(episode_ids_raw, list) or not all(
        isinstance(item, str) and item for item in episode_ids_raw
    ):
        raise ValueError("segment bucket episode_ids must be non-empty strings")
    return SegmentBucket(
        split=EpisodeSplit(string_value(row, "split")),
        tbptt_segment_count=integer_value(row, "tbptt_segment_count"),
        episode_ids=tuple(episode_ids_raw),
        loss_weights=_number_list(row, "loss_weights"),
        world_size=integer_value(row, "world_size"),
    )


def _parse_failure(value: object) -> EpisodeFailure:
    row = object_value(value, "episode failure")
    _require_exact_keys(
        row,
        {"query_id", "video_id", "source_dataset", "query_time", "video_duration", "reason"},
        "episode failure",
    )
    raw_duration = row.get("video_duration")
    duration = None if raw_duration is None else _number_value(raw_duration, "video_duration")
    return EpisodeFailure(
        query_id=string_value(row, "query_id"),
        video_id=string_value(row, "video_id"),
        source_dataset=string_value(row, "source_dataset"),
        query_time=_float_value(row, "query_time"),
        video_duration=duration,
        reason=string_value(row, "reason"),
    )


def _manifest_record_id(record: ManifestRecord) -> str:
    return record.query.runtime.query_id if isinstance(record, A2QueryRecord) else record.episode_id


def _require_a2(record: ManifestRecord) -> A2QueryRecord:
    if not isinstance(record, A2QueryRecord):
        raise TypeError("A2 sampler received an A5 episode")
    return record


def _require_a5(record: ManifestRecord) -> A5EpisodeRecord:
    if not isinstance(record, A5EpisodeRecord):
        raise TypeError("A5 sampler received an A2 Query")
    return record


def _require_exact_keys(row: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(row)
    if actual != expected:
        raise ValueError(
            f"{name} keys drifted; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _number_value(value: object, name: str) -> float:
    result = number_value(value, name)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float_value(row: Mapping[str, object], key: str) -> float:
    return _number_value(row.get(key), key)


def _object_list(row: Mapping[str, object], key: str) -> list[object]:
    value = row.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{key} must be a list of objects")
    return value


def _sequence_list(row: Mapping[str, object], key: str) -> list[object]:
    value = row.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _number_list(row: Mapping[str, object], key: str) -> tuple[float, ...]:
    return tuple(_number_value(value, key) for value in _sequence_list(row, key))


def _number_pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    return (_number_value(value[0], name), _number_value(value[1], name))


def _task_count_pair(value: object) -> tuple[str, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("task query count entries must be [task, count]")
    task, count = value
    if not isinstance(task, str) or not task:
        raise ValueError("task query count name must be non-empty")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("task query count must be a positive integer")
    return task, count


__all__ = [
    "A2QueryRecord",
    "A5EpisodeRecord",
    "AdaptiveChunkSpec",
    "AnswerSupervisionSidecar",
    "BalancedA2DistributedSampler",
    "ChunkRole",
    "EpisodeFailure",
    "EpisodeSplit",
    "ManifestStage",
    "ProductionEpisodeManifest",
    "ProductionQueryRecord",
    "ProductionManifestDataset",
    "RankAlignedA5SegmentSampler",
    "SegmentBucket",
    "WeakQuerySidecar",
    "adaptive_support_schedule",
    "build_production_episode_manifest",
    "build_production_train_sampler",
    "greedy_nonoverlap_query_groups",
    "load_production_episode_manifest",
    "load_visual_cost_index",
    "official_operator",
    "official_time_mode",
    "write_production_episode_manifest",
]
