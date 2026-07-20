"""Disk and process-local cache for causal video preprocessing.

The cache stops at the Qwen video-processor boundary.  It intentionally contains no labels,
model parameters, State Bank values, or Fast-TTT runtime state.  A cache entry is therefore safe
to share between A2/A5 workers and across epochs as long as its preprocessing fingerprint still
matches the current video and configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

CACHE_SCHEMA_VERSION = 1


class PreprocessCacheMode(StrEnum):
    DISABLED = "disabled"
    READ_WRITE = "read_write"
    READONLY = "readonly"


class PreprocessCacheMissPolicy(StrEnum):
    DECODE = "decode"
    ERROR = "error"


class PreprocessCacheMissError(RuntimeError):
    """A strict cache run encountered an absent, stale, or corrupt entry."""


@dataclass(frozen=True, slots=True)
class PreprocessFingerprint:
    """All inputs that can change a cached causal chunk."""

    source_dataset: str
    relative_video_path: str
    video_file_size: int
    video_file_mtime_ns: int
    start_time: float
    end_time: float
    maximum_frames: int
    sample_fps: float
    minimum_pixels: int
    maximum_pixels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    transformers_version: str
    observation_role: str = "support"
    frame_sampling: str = "uniform"
    cache_schema_version: int = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.source_dataset or not self.relative_video_path:
            raise ValueError("preprocess fingerprint dataset/path must be non-empty")
        if self.video_file_size < 0 or self.video_file_mtime_ns < 0:
            raise ValueError("preprocess fingerprint video stat must be non-negative")
        if (
            not math.isfinite(self.start_time)
            or not math.isfinite(self.end_time)
            or self.start_time < 0.0
            or self.end_time <= self.start_time
        ):
            raise ValueError("preprocess fingerprint interval is invalid")
        if self.maximum_frames < 2 or self.sample_fps <= 0.0:
            raise ValueError("preprocess fingerprint frame settings are invalid")
        if self.observation_role not in {"support", "query"} or not self.frame_sampling:
            raise ValueError("preprocess fingerprint observation role/policy is invalid")

    def canonical_json(self) -> str:
        values = asdict(self)
        # Preserve the digest of the already-warmed Support cache. Query observations always
        # retain the explicit role/policy fields, so a causal-prefix Query can never reuse a
        # legacy 16-frame current-chunk entry.
        if self.observation_role == "support" and self.frame_sampling == "uniform":
            values.pop("observation_role")
            values.pop("frame_sampling")
        return json.dumps(values, sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CachedChunk:
    """Tensor payload reconstructed from one cache entry."""

    frames: Tensor
    frame_timestamps: Tensor
    pixel_values_videos: Tensor
    video_grid_thw: Tensor
    tubelet_timestamps: Tensor
    tubelet_valid_mask: Tensor
    tubelet_position_ids: Tensor


def build_fingerprint(
    *,
    source_dataset: str,
    relative_video_path: str,
    video_path: Path,
    start_time: float,
    end_time: float,
    maximum_frames: int,
    sample_fps: float,
    minimum_pixels: int,
    maximum_pixels: int,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    transformers_version: str,
    observation_role: str = "support",
    frame_sampling: str = "uniform",
) -> PreprocessFingerprint:
    stat = video_path.stat()
    return PreprocessFingerprint(
        source_dataset=source_dataset,
        relative_video_path=relative_video_path,
        video_file_size=int(stat.st_size),
        video_file_mtime_ns=int(stat.st_mtime_ns),
        start_time=round(float(start_time), 6),
        end_time=round(float(end_time), 6),
        maximum_frames=int(maximum_frames),
        sample_fps=float(sample_fps),
        minimum_pixels=int(minimum_pixels),
        maximum_pixels=int(maximum_pixels),
        patch_size=int(patch_size),
        temporal_patch_size=int(temporal_patch_size),
        spatial_merge_size=int(spatial_merge_size),
        transformers_version=str(transformers_version),
        observation_role=observation_role,
        frame_sampling=frame_sampling,
    )


class PreprocessCache:
    """Bounded process-local cache backed by optional atomic safetensors files."""

    def __init__(
        self,
        root: str | Path | None,
        *,
        max_bytes: int = 200 * 1024**3,
        memory_entries: int = 2,
        mode: PreprocessCacheMode | str = PreprocessCacheMode.READ_WRITE,
        miss_policy: PreprocessCacheMissPolicy | str = PreprocessCacheMissPolicy.DECODE,
        namespace: str | None = None,
    ) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("preprocess cache max_bytes must be a positive integer")
        if type(memory_entries) is not int or memory_entries < 0:
            raise ValueError("preprocess cache memory_entries must be non-negative")
        self.mode = PreprocessCacheMode(mode)
        self.miss_policy = PreprocessCacheMissPolicy(miss_policy)
        if self.mode is not PreprocessCacheMode.DISABLED and root is None:
            raise ValueError("enabled preprocess cache requires a root directory")
        if (
            self.mode is PreprocessCacheMode.DISABLED
            and self.miss_policy is PreprocessCacheMissPolicy.ERROR
        ):
            raise ValueError("disabled preprocess cache cannot use miss_policy=error")
        self.root = None if root is None else Path(root).expanduser().resolve()
        self.max_bytes = max_bytes
        self.memory_entries = memory_entries
        if namespace is not None:
            namespace = namespace.strip().replace("\\", "/").strip("/")
            if not namespace or any(part in {".", ".."} for part in namespace.split("/")):
                raise ValueError("preprocess cache namespace must be a safe non-empty path")
        self.namespace = namespace
        self._memory: OrderedDict[str, CachedChunk] = OrderedDict()
        self._memory_sizes: dict[str, int] = {}
        self.hit_count = 0
        self.miss_count = 0
        if self.enabled and self.root is not None:
            if self.mode is PreprocessCacheMode.READ_WRITE:
                self.root.mkdir(parents=True, exist_ok=True)
            elif not self.root.is_dir():
                raise FileNotFoundError(f"readonly preprocess cache does not exist: {self.root}")

    @property
    def enabled(self) -> bool:
        return self.mode is not PreprocessCacheMode.DISABLED

    @property
    def writable(self) -> bool:
        return self.mode is PreprocessCacheMode.READ_WRITE

    def payload_size(self, fingerprint: PreprocessFingerprint) -> int:
        """Return the bytes read for one cached tensor payload, or zero when absent."""

        key = fingerprint.digest
        memory_size = self._memory_sizes.get(key)
        if memory_size is not None:
            return int(memory_size)
        path = self._path_for(fingerprint)
        if path is None:
            return 0
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def get(self, fingerprint: PreprocessFingerprint) -> CachedChunk | None:
        if not self.enabled:
            return None
        key = fingerprint.digest
        cached = self._memory.get(key)
        if cached is not None:
            self._memory.move_to_end(key)
            self.hit_count += 1
            return _clone_cached_chunk(cached)
        path = self._path_for(fingerprint)
        if path is None or not path.is_file():
            return self._miss(fingerprint, "entry_missing")
        try:
            tensors = load_file(str(path), device="cpu")
            embedded_metadata = _read_metadata(tensors)
            sidecar_metadata = _read_sidecar_metadata(path)
            if sidecar_metadata is not None and sidecar_metadata != embedded_metadata:
                return self._miss(fingerprint, "sidecar_mismatch")
            metadata = sidecar_metadata or embedded_metadata
            if metadata != fingerprint.canonical_json():
                return self._miss(fingerprint, "fingerprint_mismatch")
            cached = _cached_chunk_from_tensors(tensors)
        except PreprocessCacheMissError:
            raise
        except Exception:  # corrupt/partially replaced entries follow the configured miss policy
            return self._miss(fingerprint, "entry_corrupt")
        if self.mode is PreprocessCacheMode.READ_WRITE:
            with suppress(OSError):
                os.utime(path, None)
        try:
            size = path.stat().st_size
        except OSError:
            return self._miss(fingerprint, "entry_stat_failed")
        self._remember(key, cached, size)
        self.hit_count += 1
        return _clone_cached_chunk(cached)

    def put(self, fingerprint: PreprocessFingerprint, chunk: CachedChunk) -> None:
        if not self.enabled:
            return
        if not self.writable:
            raise PermissionError("readonly preprocess cache forbids put()")
        _validate_cached_chunk(chunk)
        key = fingerprint.digest
        self._remember(key, chunk, _tensor_bytes(chunk))
        path = self._path_for(fingerprint)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        tensors = _cached_chunk_tensors(chunk, fingerprint.canonical_json())
        try:
            save_file(
                tensors,
                str(temporary),
                metadata={"fingerprint": fingerprint.canonical_json()},
            )
            _replace_idempotent(temporary, path)
            metadata_temporary = path.with_name(
                f".{path.stem}.{os.getpid()}.{time.time_ns()}.json.tmp"
            )
            metadata_temporary.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint.canonical_json(),
                        "cache_schema_version": CACHE_SCHEMA_VERSION,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            _replace_idempotent(metadata_temporary, self._metadata_path(path))
        finally:
            if temporary.exists():
                temporary.unlink()
            if "metadata_temporary" in locals() and metadata_temporary.exists():
                metadata_temporary.unlink()

    def clear_memory(self) -> None:
        self._memory.clear()
        self._memory_sizes.clear()

    def stats(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "miss_policy": self.miss_policy.value,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "memory_entries": len(self._memory),
        }

    def disk_size_bytes(self) -> int:
        """Return the current on-disk payload size, ignoring temporary files."""

        if self.root is None or not self.root.exists():
            return 0
        total = 0
        for path in self.root.rglob("*.safetensors"):
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def prune(self) -> int:
        """Evict least-recently-accessed entries until ``max_bytes`` is respected.

        The operation is deliberately best-effort: another rank may replace or remove a file
        between the scan and unlink.  Such races are harmless because cache entries are
        recomputable and writes are atomic.
        """

        if not self.writable:
            raise PermissionError("cache prune requires read_write mode")
        if self.root is None or not self.root.exists():
            return 0
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for path in self.root.rglob("*.safetensors"):
            try:
                stat = path.stat()
            except OSError:
                continue
            total += stat.st_size
            entries.append((stat.st_atime_ns / 1.0e9, stat.st_size, path))
        removed = 0
        if total <= self.max_bytes:
            return removed
        for _, size, path in sorted(entries, key=lambda item: item[0]):
            if total <= self.max_bytes:
                break
            try:
                path.unlink()
                metadata = self._metadata_path(path)
                if metadata.exists():
                    metadata.unlink()
            except OSError:
                continue
            total -= size
            removed += 1
        return removed

    def _miss(
        self,
        fingerprint: PreprocessFingerprint,
        reason: str,
    ) -> CachedChunk | None:
        self.miss_count += 1
        if self.miss_policy is PreprocessCacheMissPolicy.ERROR:
            raise PreprocessCacheMissError(
                f"strict preprocess cache miss for {fingerprint.digest}: {reason}"
            )
        return None

    def _path_for(self, fingerprint: PreprocessFingerprint) -> Path | None:
        if self.root is None:
            return None
        # Two digest levels avoid directories with millions of entries on shared filesystems.
        digest = fingerprint.digest
        root = self.root if self.namespace is None else self.root / self.namespace
        return root / digest[:2] / digest[2:4] / f"{digest}.safetensors"

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        return path.with_suffix(".json")

    def _remember(self, key: str, chunk: CachedChunk, size: int) -> None:
        if self.memory_entries == 0:
            return
        self._memory[key] = _clone_cached_chunk(chunk)
        self._memory_sizes[key] = max(0, int(size))
        self._memory.move_to_end(key)
        while len(self._memory) > self.memory_entries:
            old_key, _ = self._memory.popitem(last=False)
            self._memory_sizes.pop(old_key, None)


def _cached_chunk_tensors(chunk: CachedChunk, fingerprint_json: str) -> dict[str, Tensor]:
    # safetensors metadata is written separately; storing the JSON as a tiny tensor also lets
    # load_file() validate it without opening a second sidecar file.
    metadata_tensor = torch.tensor(list(fingerprint_json.encode("utf-8")), dtype=torch.uint8)
    return {
        "frames": chunk.frames.detach().cpu().contiguous(),
        "frame_timestamps": chunk.frame_timestamps.detach().cpu().contiguous(),
        "pixel_values_videos": (
            chunk.pixel_values_videos.detach().cpu().to(torch.float32).contiguous()
        ),
        "video_grid_thw": chunk.video_grid_thw.detach().cpu().to(torch.int64).contiguous(),
        "tubelet_timestamps": chunk.tubelet_timestamps.detach().cpu().contiguous(),
        "tubelet_valid_mask": chunk.tubelet_valid_mask.detach().cpu().contiguous(),
        "tubelet_position_ids": chunk.tubelet_position_ids.detach().cpu().contiguous(),
        "__fingerprint_json": metadata_tensor,
    }


def _cached_chunk_from_tensors(tensors: Mapping[str, Tensor]) -> CachedChunk:
    required = {
        "frames",
        "frame_timestamps",
        "pixel_values_videos",
        "video_grid_thw",
        "tubelet_timestamps",
        "tubelet_valid_mask",
        "tubelet_position_ids",
    }
    missing = required.difference(tensors)
    if missing:
        raise KeyError(f"cache entry is missing tensors: {sorted(missing)}")
    chunk = CachedChunk(
        frames=tensors["frames"],
        frame_timestamps=tensors["frame_timestamps"],
        pixel_values_videos=tensors["pixel_values_videos"],
        video_grid_thw=tensors["video_grid_thw"],
        tubelet_timestamps=tensors["tubelet_timestamps"],
        tubelet_valid_mask=tensors["tubelet_valid_mask"],
        tubelet_position_ids=tensors["tubelet_position_ids"],
    )
    _validate_cached_chunk(chunk)
    return chunk


def _read_metadata(tensors: Mapping[str, Tensor]) -> str:
    raw = tensors.get("__fingerprint_json")
    if raw is None or raw.dtype != torch.uint8 or raw.ndim != 1:
        raise ValueError("cache entry has no valid fingerprint metadata")
    return bytes(int(value) for value in raw.tolist()).decode("utf-8")


def _read_sidecar_metadata(path: Path) -> str | None:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    value = raw.get("fingerprint") if isinstance(raw, Mapping) else None
    return value if isinstance(value, str) else None


def _replace_idempotent(source: Path, target: Path) -> None:
    """Publish one entry atomically while tolerating duplicate writers."""

    try:
        os.replace(source, target)
    except OSError:
        # A second rank/worker may have published the same fingerprint between our write and
        # replace (Windows can report this while the target is briefly mapped by a reader).  The
        # cache entry is immutable by fingerprint, so an existing target is an acceptable winner.
        if not target.is_file():
            raise


def _validate_cached_chunk(chunk: CachedChunk) -> None:
    if chunk.frames.ndim != 4 or chunk.frames.shape[1] != 3:
        raise ValueError("cached frames must be [F, 3, H, W]")
    if chunk.frames.dtype != torch.uint8:
        raise TypeError("cached frames must use uint8")
    if chunk.frames.shape[0] < 2 or chunk.frames.shape[0] % 2:
        raise ValueError("cached frames must contain an even number of tubelet frames")
    if (
        chunk.frame_timestamps.shape != (chunk.frames.shape[0],)
        or chunk.frame_timestamps.dtype != torch.float64
    ):
        raise ValueError("cached frame timestamps must align with frames")
    if (
        chunk.pixel_values_videos.ndim != 2
        or chunk.pixel_values_videos.shape[1] != 1536
        or chunk.pixel_values_videos.dtype != torch.float32
    ):
        raise ValueError("cached Qwen pixels must be float32 [N, 1536]")
    if chunk.video_grid_thw.shape != (1, 3) or chunk.video_grid_thw.dtype != torch.int64:
        raise ValueError("cached video grid must be integer [1, 3]")
    if (
        chunk.tubelet_timestamps.ndim != 2
        or chunk.tubelet_timestamps.shape[0] != 1
        or chunk.tubelet_timestamps.dtype != torch.float64
    ):
        raise ValueError("cached tubelet timestamps must be [1, T]")
    if chunk.tubelet_valid_mask.shape != chunk.tubelet_timestamps.shape:
        raise ValueError("cached tubelet validity must align with timestamps")
    if chunk.tubelet_valid_mask.dtype != torch.bool:
        raise TypeError("cached tubelet validity must use bool")
    if (
        chunk.tubelet_position_ids.shape != chunk.tubelet_timestamps.shape
        or chunk.tubelet_position_ids.dtype != torch.int64
    ):
        raise ValueError("cached tubelet positions must align with timestamps")
    if int(chunk.video_grid_thw[0, 0].item()) != chunk.frames.shape[0] // 2:
        raise ValueError("cached video grid temporal dimension must equal tubelets")


def _tensor_bytes(chunk: CachedChunk) -> int:
    return sum(value.numel() * value.element_size() for value in _chunk_tensors(chunk))


def _chunk_tensors(chunk: CachedChunk) -> tuple[Tensor, ...]:
    return (
        chunk.frames,
        chunk.frame_timestamps,
        chunk.pixel_values_videos,
        chunk.video_grid_thw,
        chunk.tubelet_timestamps,
        chunk.tubelet_valid_mask,
        chunk.tubelet_position_ids,
    )


def _clone_cached_chunk(chunk: CachedChunk) -> CachedChunk:
    return CachedChunk(*(value.clone() for value in _chunk_tensors(chunk)))


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CachedChunk",
    "PreprocessCache",
    "PreprocessCacheMissError",
    "PreprocessCacheMissPolicy",
    "PreprocessCacheMode",
    "PreprocessFingerprint",
    "build_fingerprint",
]
