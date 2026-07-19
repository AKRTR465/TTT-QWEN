"""Inspect, prewarm, verify, or prune the State-TTT preprocessing cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

from safetensors.torch import load_file

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    A5EpisodeRecord,
    ManifestStage,
    load_production_manifest_views,
)
from ttt_svcbench_qwen.preprocess_cache import PreprocessCache
from ttt_svcbench_qwen.production_runtime import (
    CurrentChunkSpec,
    VideoChunkMaterializer,
    _a2_support_chunk_specs,
    _query_chunk_spec,
    _resolve_video_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "verify", "prune"):
        child = subparsers.add_parser(name)
        _add_cache_arguments(child)
    prewarm = subparsers.add_parser("prewarm")
    _add_cache_arguments(prewarm)
    prewarm.add_argument("--manifest", required=True, type=Path)
    prewarm.add_argument("--project-config", required=True, type=Path)
    prewarm.add_argument("--video-root", required=True, type=Path)
    prewarm.add_argument("--stage", choices=("a2", "a5"), required=True)
    prewarm.add_argument("--minimum-pixels", type=int, required=True)
    prewarm.add_argument("--maximum-pixels", type=int, required=True)
    prewarm.add_argument("--shard-index", type=int, default=0)
    prewarm.add_argument("--shard-count", type=int, default=1)
    prewarm.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    if args.max_gb <= 0.0:
        parser.error("--max-gb must be positive")
    if args.command == "prewarm":
        return _prewarm(args, parser)
    cache = _cache(args)
    if args.command == "prune":
        payload = _inspect(cache)
        payload["removed_entries"] = cache.prune()
        payload["size_bytes_after"] = cache.disk_size_bytes()
    elif args.command == "verify":
        payload = {**_inspect(cache), **_verify(cache)}
        if payload["corrupt_entries"]:
            print(json.dumps(payload, sort_keys=True))
            return 1
    else:
        payload = _inspect(cache)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--max-gb", type=float, default=200.0)
    parser.add_argument("--namespace", required=True)


def _cache(args: argparse.Namespace) -> PreprocessCache:
    mode = "readonly" if args.command in {"inspect", "verify"} else "read_write"
    return PreprocessCache(
        args.root,
        max_bytes=int(args.max_gb * 1024**3),
        memory_entries=0,
        mode=mode,
        namespace=args.namespace,
    )


def _inspect(cache: PreprocessCache) -> dict[str, object]:
    root = cache.root if cache.namespace is None else cache.root / cache.namespace  # type: ignore[operator]
    usage = shutil.disk_usage(cache.root)  # type: ignore[arg-type]
    return {
        "root": str(cache.root),
        "namespace": cache.namespace,
        "namespace_root": str(root),
        "entry_count": len(tuple(root.rglob("*.safetensors"))),
        "size_bytes": cache.disk_size_bytes(),
        "max_bytes": cache.max_bytes,
        "free_bytes": usage.free,
    }


def _verify(cache: PreprocessCache) -> dict[str, int]:
    root = cache.root if cache.namespace is None else cache.root / cache.namespace  # type: ignore[operator]
    valid = corrupt = 0
    for path in root.rglob("*.safetensors"):
        try:
            tensors = load_file(str(path), device="cpu")
            if "__fingerprint_json__" not in tensors:
                raise ValueError("missing embedded fingerprint")
            metadata = path.with_suffix(".json")
            if not metadata.is_file() or not isinstance(
                json.loads(metadata.read_text(encoding="utf-8")), dict
            ):
                raise ValueError("missing cache metadata sidecar")
        except (OSError, ValueError, json.JSONDecodeError):
            corrupt += 1
        else:
            valid += 1
    return {"valid_entries": valid, "corrupt_entries": corrupt}


def _prewarm(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("prewarm shard must satisfy 0 <= index < count")
    if args.minimum_pixels <= 0 or args.maximum_pixels < args.minimum_pixels:
        parser.error("pixel limits must satisfy 0 < minimum <= maximum")
    os.environ["SVCBENCH_VIDEO_ROOT"] = str(args.video_root.resolve())
    config = load_config(args.project_config)
    cache = _cache(args)
    materializer = VideoChunkMaterializer(
        config,
        minimum_pixels=args.minimum_pixels,
        maximum_pixels=args.maximum_pixels,
        preprocess_cache=cache,
        prefetch_depth=1,
        decode_coalesce=False,
    )
    train, evaluation = load_production_manifest_views(
        args.manifest,
        stage=ManifestStage(args.stage),
    )
    specs = tuple(_iter_specs((*train.records, *evaluation.records)))
    selected = tuple(
        spec for spec in specs if _owns_shard(spec, args.shard_index, args.shard_count)
    )
    before = cache.disk_size_bytes()
    for spec, source_dataset in selected:
        materializer.set_source_dataset(source_dataset)
        materializer(spec)
    after = cache.disk_size_bytes()
    payload = {
        **_inspect(cache),
        "stage": args.stage,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "unique_chunk_count": len(specs),
        "selected_chunk_count": len(selected),
        "written_bytes": max(0, after - before),
        "average_entry_bytes": 0 if not selected else max(0, after - before) // len(selected),
    }
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


def _iter_specs(
    records: Iterable[A2QueryRecord | A5EpisodeRecord],
) -> Iterable[tuple[CurrentChunkSpec, str]]:
    seen: set[str] = set()
    for record in records:
        path = _resolve_video_path(record.source_dataset, record.relative_video_path)
        if isinstance(record, A2QueryRecord):
            specs = (
                *_a2_support_chunk_specs(record, path),
                _query_chunk_spec(
                    f"{record.query.runtime.query_id}:query",
                    path,
                    record.query.runtime.query_time,
                    reset_soft_state=False,
                ),
            )
        else:
            query_time = record.queries[0].runtime.query_time
            chunks = (record.prewarm, *record.supports)
            specs = tuple(
                CurrentChunkSpec(
                    chunk_id=f"{record.episode_id}:prewarm"
                    if index == 0
                    else f"{record.episode_id}:s{index}",
                    video_path=path,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    maximum_frames=chunk.maximum_frames,
                    query_time=query_time,
                )
                for index, chunk in enumerate(chunks)
            ) + tuple(
                _query_chunk_spec(
                    f"{record.episode_id}:q{index}",
                    path,
                    query.runtime.query_time,
                    reset_soft_state=index > 0,
                )
                for index, query in enumerate(record.queries)
            )
        for spec in specs:
            key = f"{record.source_dataset}:{spec.video_path}:{spec.start_time}:{spec.end_time}"
            if key not in seen:
                seen.add(key)
                yield spec, record.source_dataset


def _owns_shard(
    item: tuple[CurrentChunkSpec, str],
    shard_index: int,
    shard_count: int,
) -> bool:
    spec, source_dataset = item
    digest = hashlib.sha256(f"{source_dataset}:{spec.chunk_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % shard_count == shard_index


if __name__ == "__main__":
    raise SystemExit(main())
