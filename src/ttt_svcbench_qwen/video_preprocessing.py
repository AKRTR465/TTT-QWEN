"""Perform causal video cutoff/chunking and the pinned Qwen3-VL video transform.

Inputs: decoded frames with timestamps, legal query_time, and validated preprocessing config.
Outputs: audited overlapping chunks and project-normalized Qwen tubelet tensors/grid metadata.
Forbidden: future frames, full-video preprocessing before cutoff, hidden padding, or hardcoded
grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

import av
import torch
from torch import Tensor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

from ttt_svcbench_qwen.config import ProjectConfig, VideoPreprocessingConfig


class CausalBoundary(StrEnum):
    RIGHT_CLOSED = "right_closed"


@dataclass(frozen=True, slots=True)
class DecodedVideo:
    frames: Tensor
    timestamps: Tensor
    source_fps: float

    def __post_init__(self) -> None:
        _validate_frames_and_timestamps(self.frames, self.timestamps)
        if self.source_fps <= 0.0:
            raise ValueError("source_fps must be positive")


@dataclass(frozen=True, slots=True)
class CausalCut:
    frames: Tensor
    timestamps: Tensor
    original_frame_indices: Tensor
    query_time: float
    boundary: CausalBoundary
    max_visible_time: float | None

    def __post_init__(self) -> None:
        _validate_frames_and_timestamps(self.frames, self.timestamps)
        if self.original_frame_indices.shape != self.timestamps.shape:
            raise ValueError("original_frame_indices must be [F_visible]")
        if self.original_frame_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("original_frame_indices must use an integer dtype")
        if self.query_time < 0.0:
            raise ValueError("query_time must be non-negative")
        if self.timestamps.numel() and bool(torch.any(self.timestamps > self.query_time)):
            raise ValueError("causal cut contains a frame after query_time")
        expected_max = float(self.timestamps[-1].item()) if self.timestamps.numel() else None
        if self.max_visible_time != expected_max:
            raise ValueError("max_visible_time must equal the final retained timestamp")


@dataclass(frozen=True, slots=True)
class CausalChunk:
    chunk_index: int
    frames: Tensor
    frame_timestamps: Tensor
    frame_valid_mask: Tensor
    original_frame_indices: Tensor
    chunk_start_time: float | None
    chunk_end_time: float | None
    tubelet_timestamps: Tensor
    tubelet_valid_mask: Tensor
    tubelet_frame_counts: Tensor
    tubelet_source_indices: Tensor
    overlap_with_previous: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.chunk_index < 0 or self.frames.ndim != 4:
            raise ValueError("chunk_index/frames are invalid")
        frame_count = self.frames.shape[0]
        if self.frame_timestamps.shape != (frame_count,) or not torch.is_floating_point(
            self.frame_timestamps
        ):
            raise ValueError("frame_timestamps must be floating [frames_per_chunk]")
        if (
            self.frame_valid_mask.shape != (frame_count,)
            or self.frame_valid_mask.dtype != torch.bool
        ):
            raise ValueError("frame_valid_mask must be bool [frames_per_chunk]")
        if self.original_frame_indices.shape != (frame_count,):
            raise ValueError("original_frame_indices must be [frames_per_chunk]")
        tubelet_count = self.tubelet_timestamps.shape[0]
        if self.tubelet_valid_mask.shape != (tubelet_count,):
            raise ValueError("tubelet_valid_mask must be [T]")
        if self.tubelet_frame_counts.shape != (tubelet_count,):
            raise ValueError("tubelet_frame_counts must be [T]")
        if self.tubelet_source_indices.shape[0] != tubelet_count:
            raise ValueError("tubelet_source_indices must have one row per tubelet")
        if self.frame_valid_mask.any():
            valid_times = self.frame_timestamps[self.frame_valid_mask]
            if self.chunk_start_time != float(valid_times[0].item()):
                raise ValueError("chunk_start_time does not match the first valid frame")
            if self.chunk_end_time != float(valid_times[-1].item()):
                raise ValueError("chunk_end_time does not match the final valid frame")
        elif self.chunk_start_time is not None or self.chunk_end_time is not None:
            raise ValueError("an empty chunk must use null start/end times")


@dataclass(frozen=True, slots=True)
class CausalChunks:
    query_time: float
    boundary: CausalBoundary
    max_visible_time: float | None
    chunks: tuple[CausalChunk, ...]

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("CausalChunks must contain an auditable empty or non-empty chunk")
        if self.max_visible_time is not None and self.max_visible_time > self.query_time:
            raise ValueError("CausalChunks contains future-visible time")


@dataclass(frozen=True, slots=True)
class QwenProcessedVideo:
    pixel_values_videos: Tensor
    video_grid_thw: Tensor

    def __post_init__(self) -> None:
        pixels = self.pixel_values_videos
        if pixels.ndim != 3 or pixels.shape[0] != 1 or not torch.is_floating_point(pixels):
            raise ValueError("pixel_values_videos must be project-normalized floating [1, N, D]")
        if self.video_grid_thw.shape != (1, 3):
            raise ValueError("video_grid_thw must be integer [1, 3]")
        if self.video_grid_thw.dtype not in (torch.int32, torch.int64):
            raise TypeError("video_grid_thw must use an integer dtype")
        expected_tokens = int(torch.prod(self.video_grid_thw[0]).item())
        if pixels.shape[1] != expected_tokens:
            raise ValueError("pixel token count must equal product(video_grid_thw)")

    def flatten_for_qwen(self) -> Tensor:
        """Return the raw 2-D layout expected by the upstream Qwen vision tower."""

        return self.pixel_values_videos.squeeze(0)


class QwenVideoPreprocessor:
    """Pinned wrapper around the checkpoint's real Transformers video processor."""

    def __init__(self, config: ProjectConfig) -> None:
        video = config.video_preprocessing
        self._config = video
        self._processor = Qwen3VLVideoProcessor(
            size={
                "shortest_edge": video.processor_shortest_edge,
                "longest_edge": video.processor_longest_edge,
            },
            patch_size=video.patch_size,
            temporal_patch_size=video.temporal_patch_size,
            merge_size=video.spatial_merge_size,
        )

    def process(self, frames: Tensor) -> QwenProcessedVideo:
        if frames.ndim == 5:
            if frames.shape[0] != 1:
                raise ValueError("P2 processor wrapper accepts one video at a time")
            frames = frames[0]
        if frames.ndim != 4 or frames.shape[0] < self._config.temporal_patch_size:
            raise ValueError("frames must be [F, C, H, W] with at least one full tubelet")
        if frames.shape[1] != 3:
            raise ValueError("Qwen video processor requires RGB frames")
        output = self._processor(
            videos=[frames],
            do_sample_frames=False,
            return_tensors="pt",
        )
        raw_pixels = cast(Tensor, output["pixel_values_videos"])
        grid = cast(Tensor, output["video_grid_thw"])
        expected_patch_dim = 3 * self._config.temporal_patch_size * self._config.patch_size**2
        if raw_pixels.ndim != 2 or raw_pixels.shape[-1] != expected_patch_dim:
            raise ValueError("upstream Qwen processor returned an unexpected patch dimension")
        return QwenProcessedVideo(pixel_values_videos=raw_pixels.unsqueeze(0), video_grid_thw=grid)


def causal_right_cut(frames: Tensor, timestamps: Tensor, query_time: float) -> CausalCut:
    _validate_frames_and_timestamps(frames, timestamps)
    if query_time < 0.0:
        raise ValueError("query_time must be non-negative")
    if timestamps.numel() > 1 and bool(torch.any(timestamps[1:] <= timestamps[:-1])):
        raise ValueError("frame timestamps must be strictly increasing")
    visible_mask = timestamps <= query_time
    if visible_mask.numel() and bool(torch.any(visible_mask[1:] & ~visible_mask[:-1])):
        raise ValueError("visible frames must form a causal prefix")
    visible_count = int(visible_mask.sum().item())
    visible_frames = frames[:visible_count]
    visible_timestamps = timestamps[:visible_count]
    original_indices = torch.arange(visible_count, dtype=torch.int64, device=timestamps.device)
    max_visible = float(visible_timestamps[-1].item()) if visible_count else None
    return CausalCut(
        frames=visible_frames,
        timestamps=visible_timestamps,
        original_frame_indices=original_indices,
        query_time=query_time,
        boundary=CausalBoundary.RIGHT_CLOSED,
        max_visible_time=max_visible,
    )


def chunk_causal_cut(cut: CausalCut, config: VideoPreprocessingConfig) -> CausalChunks:
    frame_count = cut.frames.shape[0]
    starts = [0]
    while starts[-1] + config.frames_per_chunk < frame_count:
        starts.append(starts[-1] + config.stride_frames)

    chunks: list[CausalChunk] = []
    previous_pairs: dict[tuple[int, ...], int] = {}
    for chunk_index, start in enumerate(starts):
        end = min(start + config.frames_per_chunk, frame_count)
        real_count = max(0, end - start)
        shape = (config.frames_per_chunk, *cut.frames.shape[1:])
        padded_frames = torch.full(
            shape,
            config.pad_value,
            dtype=cut.frames.dtype,
            device=cut.frames.device,
        )
        frame_valid_mask = torch.zeros(
            config.frames_per_chunk, dtype=torch.bool, device=cut.frames.device
        )
        frame_timestamps = torch.full(
            (config.frames_per_chunk,),
            -1.0,
            dtype=cut.timestamps.dtype,
            device=cut.timestamps.device,
        )
        original_indices = torch.full(
            (config.frames_per_chunk,), -1, dtype=torch.int64, device=cut.timestamps.device
        )
        if real_count:
            padded_frames[:real_count] = cut.frames[start:end]
            frame_valid_mask[:real_count] = True
            frame_timestamps[:real_count] = cut.timestamps[start:end]
            original_indices[:real_count] = cut.original_frame_indices[start:end]

        (
            tubelet_timestamps,
            tubelet_valid_mask,
            tubelet_frame_counts,
            tubelet_source_indices,
        ) = _build_tubelet_audit(
            frame_timestamps,
            frame_valid_mask,
            original_indices,
            config.temporal_patch_size,
            config.full_tubelet_required_for_state,
        )
        current_pairs = {
            tuple(int(value) for value in tubelet_source_indices[index].tolist()): index
            for index in range(tubelet_source_indices.shape[0])
            if bool(tubelet_valid_mask[index])
        }
        overlap = tuple(
            (previous_pairs[pair], current_index)
            for pair, current_index in current_pairs.items()
            if pair in previous_pairs
        )
        valid_times = frame_timestamps[frame_valid_mask]
        chunks.append(
            CausalChunk(
                chunk_index=chunk_index,
                frames=padded_frames,
                frame_timestamps=frame_timestamps,
                frame_valid_mask=frame_valid_mask,
                original_frame_indices=original_indices,
                chunk_start_time=float(valid_times[0].item()) if real_count else None,
                chunk_end_time=float(valid_times[-1].item()) if real_count else None,
                tubelet_timestamps=tubelet_timestamps,
                tubelet_valid_mask=tubelet_valid_mask,
                tubelet_frame_counts=tubelet_frame_counts,
                tubelet_source_indices=tubelet_source_indices,
                overlap_with_previous=overlap,
            )
        )
        previous_pairs = current_pairs
    return CausalChunks(
        query_time=cut.query_time,
        boundary=cut.boundary,
        max_visible_time=cut.max_visible_time,
        chunks=tuple(chunks),
    )


def decode_video_causally(
    path: str | Path,
    *,
    query_time: float,
    sample_fps: float,
) -> DecodedVideo:
    """Decode/sample only the right-closed causal prefix of one video."""

    if query_time < 0.0 or sample_fps <= 0.0:
        raise ValueError("query_time and sample_fps must be non-negative/positive")
    frames: list[Tensor] = []
    timestamps: list[float] = []
    next_sample_time = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate is not None else sample_fps
        for frame in container.decode(stream):
            timestamp = av_frame_timestamp(frame)
            if timestamp > query_time:
                break
            if timestamp + 1.0e-9 < next_sample_time:
                continue
            array = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
            timestamps.append(timestamp)
            while next_sample_time <= timestamp + 1.0e-9:
                next_sample_time += 1.0 / sample_fps
    frame_tensor = torch.stack(frames) if frames else torch.empty((0, 3, 0, 0), dtype=torch.uint8)
    return DecodedVideo(
        frames=frame_tensor,
        timestamps=torch.tensor(timestamps, dtype=torch.float64),
        source_fps=source_fps,
    )


def _build_tubelet_audit(
    frame_timestamps: Tensor,
    frame_valid_mask: Tensor,
    original_indices: Tensor,
    temporal_patch_size: int,
    require_full: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if frame_timestamps.shape[0] % temporal_patch_size:
        raise ValueError("chunk length must be divisible by temporal_patch_size")
    tubelet_count = frame_timestamps.shape[0] // temporal_patch_size
    grouped_times = frame_timestamps.reshape(tubelet_count, temporal_patch_size)
    grouped_valid = frame_valid_mask.reshape(tubelet_count, temporal_patch_size)
    grouped_indices = original_indices.reshape(tubelet_count, temporal_patch_size)
    counts = grouped_valid.sum(dim=1).to(torch.int64)
    valid = counts == temporal_patch_size if require_full else counts > 0
    timestamps = torch.full(
        (tubelet_count,), -1.0, dtype=frame_timestamps.dtype, device=frame_timestamps.device
    )
    for index in range(tubelet_count):
        if counts[index] > 0:
            timestamps[index] = grouped_times[index][grouped_valid[index]].max()
    return timestamps, valid, counts, grouped_indices


def av_frame_timestamp(frame: av.VideoFrame) -> float:
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is None or frame.time_base is None:
        raise ValueError("decoded frame has no auditable timestamp")
    return float(frame.pts * frame.time_base)


def _validate_frames_and_timestamps(frames: Tensor, timestamps: Tensor) -> None:
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must be [F, 3, H, W]")
    if timestamps.shape != (frames.shape[0],) or not torch.is_floating_point(timestamps):
        raise ValueError("timestamps must be floating [F]")
    if timestamps.numel() and not bool(torch.all(torch.isfinite(timestamps))):
        raise ValueError("timestamps must be finite")
