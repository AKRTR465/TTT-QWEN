"""Build production A2/A5 manifests.

The manifest is a training sidecar: labels in this module are consumed only by the
post-forward loss builder, never by a runtime model payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

from torch.utils.data import Dataset, Sampler

from ttt_svcbench_qwen.data import (
    FoldManifest,
    LoadedAnnotations,
    RuntimeQueryInput,
    SVCBenchRecord,
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


class EpisodeSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"


class ChunkRole(StrEnum):
    PREWARM = "prewarm"
    SUPPORT = "support"


class A5QueryRole(StrEnum):
    INTERMEDIATE = "intermediate"
    FINAL = "final"


@dataclass(frozen=True, slots=True)
class AdaptiveChunkSpec:
    role: ChunkRole
    start_time: float
    end_time: float
    maximum_frames: int = 16
    frame_sampling: str = "uniform"


@dataclass(frozen=True, slots=True)
class AnswerSupervisionSidecar:
    """Answer-only training label kept outside :class:`RuntimeQueryInput`."""

    query_id: str
    answer: str | None
    provenance: str


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


@dataclass(frozen=True, slots=True)
class ProductionQueryRecord:
    """One runtime Query plus two loss-only sidecars with aligned identity."""

    runtime: RuntimeQueryInput
    answer: AnswerSupervisionSidecar
    weak: WeakQuerySidecar


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


@dataclass(frozen=True, slots=True)
class A5SupervisedSegmentRecord:
    role: A5QueryRole
    supports: tuple[AdaptiveChunkSpec, ...]
    meta_query: ProductionQueryRecord
    query_weight: float = 1.0
    additional_meta_queries: tuple[ProductionQueryRecord, ...] = ()

    @property
    def queries(self) -> tuple[ProductionQueryRecord, ...]:
        """All official Query points supervised by this post-Support fast state."""

        return (*self.additional_meta_queries, self.meta_query)

    @property
    def query_roles(self) -> tuple[A5QueryRole, ...]:
        if self.role is A5QueryRole.FINAL:
            return (
                *(A5QueryRole.INTERMEDIATE for _ in self.additional_meta_queries),
                A5QueryRole.FINAL,
            )
        return (A5QueryRole.INTERMEDIATE,) * len(self.queries)

    @property
    def query_weights(self) -> tuple[float, ...]:
        return (self.query_weight,) * len(self.queries)


@dataclass(frozen=True, slots=True)
class A5EpisodeRecord:
    """One K=8 support-aligned episode: a prewarm chunk plus one or two supervised
    segments, each segment being up-to-K Support chunks followed by its Query bundle."""

    episode_id: str
    source_dataset: str
    relative_video_path: str
    video_id: str
    trajectory_id: str
    split: EpisodeSplit
    task_class: str
    operator: str
    prewarm: AdaptiveChunkSpec
    supervised_segments: tuple[A5SupervisedSegmentRecord, ...]
    support_count: int
    meta_query_count: int
    truncation_horizon: int
    tbptt_segment_count: int
    sampling_weight: float
    loss_weight: float = 1.0

    @property
    def supports(self) -> tuple[AdaptiveChunkSpec, ...]:
        return tuple(
            chunk for segment in self.supervised_segments for chunk in segment.supports
        )

    @property
    def queries(self) -> tuple[ProductionQueryRecord, ...]:
        return tuple(
            query for segment in self.supervised_segments for query in segment.queries
        )

    @property
    def segment_lengths(self) -> tuple[int, ...]:
        return tuple(len(segment.supports) for segment in self.supervised_segments)

    @property
    def segment_query_counts(self) -> tuple[int, ...]:
        return tuple(len(segment.queries) for segment in self.supervised_segments)


@dataclass(frozen=True, slots=True)
class SegmentBucket:
    split: EpisodeSplit
    tbptt_segment_count: int
    episode_ids: tuple[str, ...]
    loss_weights: tuple[float, ...]
    world_size: int


@dataclass(frozen=True, slots=True)
class ProductionEpisodeManifest:
    dataset_name: str
    dataset_revision: str
    annotation_sha256: str
    fold_index: int
    seed: int
    truncation_horizon: int
    a2_queries: tuple[A2QueryRecord, ...]
    episodes: tuple[A5EpisodeRecord, ...]
    buckets: tuple[SegmentBucket, ...]

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
        if stage is ManifestStage.A2:
            records: tuple[ManifestRecord, ...] = tuple(
                record for record in manifest.a2_queries if record.split is split
            )
        else:
            records = tuple(record for record in manifest.episodes if record.split is split)
        self.manifest = manifest
        self.stage = stage
        self.split = split
        self.records = records
        self.index_by_id = {
            _manifest_record_id(record): index for index, record in enumerate(records)
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ManifestRecord:
        return self.records[index]


def load_production_manifest_views(
    manifest_path: str | Path,
    *,
    stage: ManifestStage,
) -> tuple[ProductionManifestDataset, ProductionManifestDataset]:
    """Load the one authoritative manifest and expose immutable train/validation views."""

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
) -> tuple[int, int]:
    """Return a cheap deterministic proxy for visual tokens and decode work.

    The first component is the exact configured upper bound on causal frames at 2 FPS across
    every Support plus the configured Query observation.
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
    # Support count is already a hard sampler bucket.  Within it, frame budget predicts ViT
    # work and history write units break ties.
    return frame_budget, history_write_units


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
        query_sample_fps: float = 2.0,
        state_query_visual_mode: str = "recent_chunk",
        state_query_max_frames: int = 16,
        answer_query_visual_mode: str = "causal_prefix",
        answer_query_max_frames: int = 256,
    ) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.query_sample_fps = query_sample_fps
        self.state_query_visual_mode = state_query_visual_mode
        self.state_query_max_frames = state_query_max_frames
        self.answer_query_visual_mode = answer_query_visual_mode
        self.answer_query_max_frames = answer_query_max_frames
        buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
        records: tuple[A2QueryRecord, ...] = self.dataset.records  # type: ignore[assignment]
        for index, record in enumerate(records):
            support_count = len(adaptive_support_schedule(record.query.runtime.query_time)[1])
            buckets[(record.task_class, support_count)].append(index)
        self._buckets = {name: tuple(values) for name, values in buckets.items()}
        self._visual_lengths: dict[int, tuple[int, int]] = {
            index: _a2_visual_length_key(
                record,
                query_sample_fps=self.query_sample_fps,
                state_query_visual_mode=self.state_query_visual_mode,
                state_query_max_frames=self.state_query_max_frames,
                answer_query_visual_mode=self.answer_query_visual_mode,
                answer_query_max_frames=self.answer_query_max_frames,
            )
            for index, record in enumerate(records)
        }
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

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

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
        rng.shuffle(global_batches)
        return iter([index for batch, _ in global_batches for index in batch])

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
        query_sample_fps: float = 2.0,
        state_query_visual_mode: str = "recent_chunk",
        state_query_max_frames: int = 16,
        answer_query_visual_mode: str = "causal_prefix",
        answer_query_max_frames: int = 256,
    ) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.query_sample_fps = query_sample_fps
        self.state_query_visual_mode = state_query_visual_mode
        self.state_query_max_frames = state_query_max_frames
        self.answer_query_visual_mode = answer_query_visual_mode
        self.answer_query_max_frames = answer_query_max_frames
        self._buckets = tuple(
            bucket for bucket in dataset.manifest.buckets if bucket.split is EpisodeSplit.TRAIN
        )
        self._global_size = sum(len(bucket.episode_ids) for bucket in self._buckets)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        global_batches: list[tuple[int, ...]] = []
        records_by_id: dict[str, A5EpisodeRecord] = {
            _manifest_record_id(record): record  # type: ignore[misc]
            for record in self.dataset.records
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
            for start in range(0, len(scheduled_ids), self.world_size):
                group = scheduled_ids[start : start + self.world_size]
                global_batches.append(
                    tuple(self.dataset.index_by_id[episode_id] for episode_id in group)
                )
        rng.shuffle(global_batches)
        return iter([index for batch in global_batches for index in batch])

    def _cost_key(self, episode_id: str) -> tuple[float, int, int, int]:
        record: A5EpisodeRecord = self.dataset[  # type: ignore[assignment]
            self.dataset.index_by_id[episode_id]
        ]
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
    dataset: ProductionManifestDataset,
    rank: int,
    world_size: int,
    *,
    query_sample_fps: float = 2.0,
    state_query_visual_mode: str = "recent_chunk",
    state_query_max_frames: int = 16,
    answer_query_visual_mode: str = "causal_prefix",
    answer_query_max_frames: int = 256,
) -> Sampler[int]:
    """Shared runtime-factory hook for A2 task balance and A5 segment parity."""

    if dataset.stage is ManifestStage.A2:
        return BalancedA2DistributedSampler(
            dataset,
            rank=rank,
            world_size=world_size,
            seed=dataset.manifest.seed,
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
    start = 0.0 if mode == "causal_prefix" else max(0.0, query_time - 8.0)
    desired = min(maximum, max(2, int(math.floor((query_time - start) * sample_fps))))
    return max(2, desired - desired % 2)


def _a5_alignment_shape(record: A5EpisodeRecord) -> tuple[int, ...]:
    return tuple(
        value
        for pair in zip(
            record.segment_lengths,
            record.segment_query_counts,
            strict=True,
        )
        for value in pair
    )


def official_operator(counting_type: str, counting_subtype: str) -> Operator:
    """Map official type/subtype spelling to the exact eight-way operator surface."""

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
    return mapping[normalized_subtype]


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

    ordered = tuple(sorted(records, key=lambda item: (item.query_time, item.identity.query_index)))
    if not ordered:
        return ()
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

    # Reserve a causal current-Query observation after the final Support.  That Query chunk is
    # evaluated with M_after and is deliberately not a Support memory write.
    query_observation_gap = min(4.0, first_query_time / 2.0)
    causal_end = math.nextafter(first_query_time - query_observation_gap, 0.0)
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
    first = supports[0]
    # Prewarm establishes the initial detached online state, but must leave a real
    # unseen tail for the first Support memory write.  Consuming an entire
    # short first Support here made W_after identical to W0 for many segments.
    first_duration = first.end_time - first.start_time
    prewarm_end = first.start_time + min(4.0, first_duration / 2.0)
    prewarm_end = min(prewarm_end, first.end_time - 1.0e-6)
    if prewarm_end <= first.start_time:
        prewarm_end = first.start_time + (first.end_time - first.start_time) / 2.0
    prewarm = AdaptiveChunkSpec(ChunkRole.PREWARM, first.start_time, prewarm_end)
    return prewarm, supports


def _compress_support_schedule(
    supports: tuple[AdaptiveChunkSpec, ...],
    maximum: int,
) -> tuple[AdaptiveChunkSpec, ...]:
    """Uniformly retain temporal coverage, including the earliest and latest Support."""

    if len(supports) <= maximum:
        return supports
    if maximum == 1:
        return (supports[-1],)
    last = len(supports) - 1
    indices = tuple(index * last // (maximum - 1) for index in range(maximum))
    return tuple(supports[index] for index in indices)


def _shift_support_schedule(
    supports: tuple[AdaptiveChunkSpec, ...],
    offset: float,
) -> tuple[AdaptiveChunkSpec, ...]:
    return tuple(
        replace(
            chunk,
            start_time=chunk.start_time + offset,
            end_time=chunk.end_time + offset,
        )
        for chunk in supports
    )


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

    fold_manifest = create_group_kfold_manifest(annotations, n_splits=n_splits, seed=seed)
    split_by_video = _split_map(fold_manifest, fold_index)
    valid_records: list[SVCBenchRecord] = []
    for record in annotations.records:
        duration = _duration_for(record, video_durations)
        if duration is None or record.query_time <= duration + video_duration_tolerance:
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
    task_episode_counts = Counter(episode.task_class for episode in raw_episodes)
    episodes = tuple(
        _with_sampling_weight(episode, task_episode_counts[episode.task_class])
        for episode in raw_episodes
    )
    buckets, padding = _build_segment_buckets(episodes, world_size=world_size)
    all_episodes = episodes + padding
    return ProductionEpisodeManifest(
        dataset_name=annotations.source.name,
        dataset_revision=annotations.source.revision,
        annotation_sha256=annotations.annotation_sha256,
        fold_index=fold_index,
        seed=seed,
        truncation_horizon=truncation_horizon,
        a2_queries=a2_queries,
        episodes=all_episodes,
        buckets=buckets,
    )


def write_production_episode_manifest(
    manifest: ProductionEpisodeManifest,
    *,
    manifest_path: str | Path,
    failed_path: str | Path | None = None,
) -> None:
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    return str(value) if isinstance(value, Path) else value


def load_production_episode_manifest(path: str | Path) -> ProductionEpisodeManifest:
    """Load a serialized production manifest."""

    values = json.loads(Path(path).read_text(encoding="utf-8"))
    a2_queries = tuple(_parse_a2_query(item) for item in _object_list(values, "a2_queries"))
    episodes = tuple(_parse_a5_episode(item) for item in _object_list(values, "episodes"))
    buckets = tuple(_parse_segment_bucket(item) for item in _object_list(values, "buckets"))
    return ProductionEpisodeManifest(
        dataset_name=string_value(values, "dataset_name"),
        dataset_revision=string_value(values, "dataset_revision"),
        annotation_sha256=string_value(values, "annotation_sha256"),
        fold_index=integer_value(values, "fold_index"),
        seed=integer_value(values, "seed"),
        truncation_horizon=integer_value(values, "truncation_horizon"),
        a2_queries=a2_queries,
        episodes=episodes,
        buckets=buckets,
    )


def _episode_from_group(
    group: tuple[SVCBenchRecord, ...],
    *,
    group_index: int,
    split: EpisodeSplit,
    truncation_horizon: int,
    runtime_video_paths: Mapping[str, str] | None,
) -> A5EpisodeRecord:
    group = tuple(
        sorted(
            group,
            key=lambda item: (
                item.query_time,
                item.identity.query_index,
                item.identity.query_id,
            ),
        )
    )
    first = group[0]
    operator = official_operator(first.labels.counting_type, first.labels.counting_subtype)
    prewarm, original_supports = adaptive_support_schedule(first.query_time)
    query_records = tuple(_production_query(item, operator) for item in group)
    final_query = query_records[-1]
    split_candidates: list[tuple[int, tuple[AdaptiveChunkSpec, ...]]] = []
    for index, query in enumerate(query_records[:-1]):
        query_gap = final_query.runtime.query_time - query.runtime.query_time
        if query_gap < 4.0:
            continue
        _unused_prewarm, relative_supports = adaptive_support_schedule(query_gap)
        candidate_supports = _compress_support_schedule(
            _shift_support_schedule(
                relative_supports,
                query.runtime.query_time,
            ),
            truncation_horizon,
        )
        if candidate_supports[-1].end_time < query_records[index + 1].runtime.query_time:
            split_candidates.append((index, candidate_supports))
    split_index, split_supports = (
        split_candidates[-1] if split_candidates else (None, ())
    )
    split_episode = split_index is not None
    first_supports = _compress_support_schedule(
        original_supports,
        truncation_horizon,
    )
    supervised_segments: tuple[A5SupervisedSegmentRecord, ...]
    if split_episode:
        assert split_index is not None
        pivot_query = query_records[split_index]
        supervised_segments = (
            A5SupervisedSegmentRecord(
                role=A5QueryRole.INTERMEDIATE,
                supports=first_supports,
                meta_query=pivot_query,
                additional_meta_queries=query_records[:split_index],
            ),
            A5SupervisedSegmentRecord(
                role=A5QueryRole.FINAL,
                supports=split_supports,
                meta_query=final_query,
                additional_meta_queries=query_records[split_index + 1 : -1],
            ),
        )
    else:
        supervised_segments = (
            A5SupervisedSegmentRecord(
                role=A5QueryRole.FINAL,
                supports=first_supports,
                meta_query=final_query,
                additional_meta_queries=query_records[:-1],
            ),
        )
    digest = hashlib.sha256(
        "|".join(item.identity.query_id for item in group).encode("utf-8")
    ).hexdigest()[:12]
    episode_id = f"{first.identity.trajectory_id}-g{group_index:03d}-{digest}"
    return A5EpisodeRecord(
        episode_id=episode_id,
        source_dataset=first.source_dataset,
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
        supervised_segments=supervised_segments,
        support_count=sum(len(segment.supports) for segment in supervised_segments),
        meta_query_count=sum(len(segment.queries) for segment in supervised_segments),
        truncation_horizon=truncation_horizon,
        tbptt_segment_count=len(supervised_segments),
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


def _with_sampling_weight(episode: A5EpisodeRecord, task_episode_count: int) -> A5EpisodeRecord:
    return replace(
        episode,
        sampling_weight=1.0 / task_episode_count,
    )


def _build_segment_buckets(
    episodes: tuple[A5EpisodeRecord, ...],
    *,
    world_size: int,
) -> tuple[tuple[SegmentBucket, ...], tuple[A5EpisodeRecord, ...]]:
    grouped: dict[tuple[EpisodeSplit, tuple[int, ...]], list[A5EpisodeRecord]] = defaultdict(list)
    for episode in episodes:
        grouped[(episode.split, _a5_alignment_shape(episode))].append(episode)
    buckets: list[SegmentBucket] = []
    padding_records: list[A5EpisodeRecord] = []
    for key in sorted(grouped, key=lambda item: (item[0].value, item[1])):
        rows = sorted(grouped[key], key=lambda item: item.episode_id)
        remainder = len(rows) % world_size
        if remainder:
            source = rows[-1]
            for padding_index in range(world_size - remainder):
                # Zero-weight clone: every rank must still run the same number of
                # backward collectives for this bucket.
                padded = replace(
                    source,
                    episode_id=f"{source.episode_id}-pad{padding_index:02d}",
                    loss_weight=0.0,
                )
                rows.append(padded)
                padding_records.append(padded)
        buckets.append(
            SegmentBucket(
                split=key[0],
                tbptt_segment_count=rows[0].tbptt_segment_count,
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
    return AdaptiveChunkSpec(
        role=ChunkRole(string_value(row, "role")),
        start_time=_float_value(row, "start_time"),
        end_time=_float_value(row, "end_time"),
        maximum_frames=integer_value(row, "maximum_frames"),
        frame_sampling=string_value(row, "frame_sampling"),
    )


def _parse_runtime_query(value: object) -> RuntimeQueryInput:
    row = object_value(value, "runtime Query")
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
    answer = row.get("answer")
    return AnswerSupervisionSidecar(
        query_id=string_value(row, "query_id"),
        answer=None if answer is None else str(answer),
        provenance=string_value(row, "provenance"),
    )


def _parse_weak_sidecar(value: object) -> WeakQuerySidecar:
    row = object_value(value, "weak sidecar")
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
    return ProductionQueryRecord(
        runtime=_parse_runtime_query(row["runtime"]),
        answer=_parse_answer_sidecar(row["answer"]),
        weak=_parse_weak_sidecar(row["weak"]),
    )


def _parse_a2_query(value: object) -> A2QueryRecord:
    row = object_value(value, "A2 Query")
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
        supervised_segments=tuple(
            _parse_a5_supervised_segment(item)
            for item in _object_list(row, "supervised_segments")
        ),
        support_count=integer_value(row, "support_count"),
        meta_query_count=integer_value(row, "meta_query_count"),
        truncation_horizon=integer_value(row, "truncation_horizon"),
        tbptt_segment_count=integer_value(row, "tbptt_segment_count"),
        sampling_weight=_float_value(row, "sampling_weight"),
        loss_weight=_float_value(row, "loss_weight"),
    )


def _parse_a5_supervised_segment(value: object) -> A5SupervisedSegmentRecord:
    row = object_value(value, "A5 supervised segment")
    return A5SupervisedSegmentRecord(
        role=A5QueryRole(string_value(row, "role")),
        supports=tuple(_parse_chunk(item) for item in _object_list(row, "supports")),
        meta_query=_parse_production_query(row["meta_query"]),
        query_weight=_float_value(row, "query_weight"),
        additional_meta_queries=tuple(
            _parse_production_query(item)
            for item in _object_list(row, "additional_meta_queries")
        ),
    )


def _parse_segment_bucket(value: object) -> SegmentBucket:
    row = object_value(value, "segment bucket")
    return SegmentBucket(
        split=EpisodeSplit(string_value(row, "split")),
        tbptt_segment_count=integer_value(row, "tbptt_segment_count"),
        episode_ids=tuple(str(item) for item in _sequence_list(row, "episode_ids")),
        loss_weights=_number_list(row, "loss_weights"),
        world_size=integer_value(row, "world_size"),
    )


def _manifest_record_id(record: ManifestRecord) -> str:
    return record.query.runtime.query_id if isinstance(record, A2QueryRecord) else record.episode_id


def _float_value(row: Mapping[str, object], key: str) -> float:
    return number_value(row.get(key), key)


def _object_list(row: Mapping[str, object], key: str) -> list[object]:
    return list(row[key])  # type: ignore[call-overload]


def _sequence_list(row: Mapping[str, object], key: str) -> list[object]:
    return list(row[key])  # type: ignore[call-overload]


def _number_list(row: Mapping[str, object], key: str) -> tuple[float, ...]:
    return tuple(number_value(value, key) for value in _sequence_list(row, key))


def _number_pair(value: object, name: str) -> tuple[float, float]:
    left, right = cast(tuple[object, object], value)
    return (number_value(left, name), number_value(right, name))


__all__ = [
    "A2QueryRecord",
    "A5EpisodeRecord",
    "A5QueryRole",
    "A5SupervisedSegmentRecord",
    "AdaptiveChunkSpec",
    "AnswerSupervisionSidecar",
    "BalancedA2DistributedSampler",
    "ChunkRole",
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
    "official_operator",
    "official_time_mode",
    "write_production_episode_manifest",
]
