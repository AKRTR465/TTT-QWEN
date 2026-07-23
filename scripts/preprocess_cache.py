"""Inspect, prewarm, verify, or prune the State-TTT preprocessing cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file

from ttt_svcbench_qwen.config import ProjectConfig, load_config
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    A5EpisodeRecord,
    ManifestStage,
    load_production_manifest_views,
)
from ttt_svcbench_qwen.preprocess_cache import PreprocessCache, PreprocessFingerprint
from ttt_svcbench_qwen.production_factory import ProductionTTTConfig
from ttt_svcbench_qwen.production_runtime import (
    QueryObservationSpec,
    SupportChunkSpec,
    VideoChunkMaterializer,
    _a2_support_chunk_specs,
    _build_preprocess_fingerprint,
    _query_chunk_spec,
    _resolve_video_path,
)

ObservationSpec = SupportChunkSpec | QueryObservationSpec
FingerprintedSpec = tuple[ObservationSpec, str, PreprocessFingerprint]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "verify", "prune"):
        child = subparsers.add_parser(name)
        _add_cache_arguments(child)
    prewarm = subparsers.add_parser("prewarm")
    _add_cache_arguments(prewarm)
    _add_input_arguments(prewarm)
    prewarm.add_argument("--shard-index", type=int, default=0)
    prewarm.add_argument("--shard-count", type=int, default=1)
    prewarm.add_argument("--summary", type=Path, default=None)
    verify_inputs = subparsers.add_parser("verify-inputs")
    _add_cache_arguments(verify_inputs)
    _add_input_arguments(verify_inputs)
    args = parser.parse_args()
    if args.max_gb <= 0.0:
        parser.error("--max-gb must be positive")
    if args.command == "prewarm":
        return _prewarm(args, parser)
    if args.command == "verify-inputs":
        return _verify_inputs(args, parser)
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


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--project-config", required=True, type=Path)
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--stage", choices=("a2", "a5"), required=True)
    parser.add_argument("--minimum-pixels", type=int, required=True)
    parser.add_argument("--maximum-pixels", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=("support", "state_query", "answer_query"),
        default=("support", "state_query", "answer_query"),
    )


def _cache(args: argparse.Namespace) -> PreprocessCache:
    mode = "readonly" if args.command in {"inspect", "verify", "verify-inputs"} else "read_write"
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
            embedded = tensors.get("__fingerprint_json")
            if embedded is None or embedded.dtype != torch.uint8 or embedded.ndim != 1:
                raise ValueError("missing embedded fingerprint")
            embedded_fingerprint = bytes(
                int(value) for value in embedded.tolist()
            ).decode("utf-8")
            metadata = path.with_suffix(".json")
            sidecar = (
                json.loads(metadata.read_text(encoding="utf-8"))
                if metadata.is_file()
                else None
            )
            if not isinstance(sidecar, dict) or sidecar.get("fingerprint") != embedded_fingerprint:
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
    ttt_config = _load_training_config(args.training_config)
    if ttt_config.stage != args.stage:
        parser.error(
            f"training config stage {ttt_config.stage!r} does not match --stage {args.stage!r}"
        )
    roles = frozenset(args.roles)
    for role in ("state_query", "answer_query"):
        if role in roles and not ttt_config.query_cache_enabled(role):
            parser.error(f"{role} prewarm requires its cache mode to be inherit")
    cache = _cache(args)
    materializer = VideoChunkMaterializer(
        config,
        minimum_pixels=args.minimum_pixels,
        maximum_pixels=args.maximum_pixels,
        preprocess_cache=cache,
        cache_support_visuals="support" in roles,
        cache_query_roles=frozenset(roles & {"state_query", "answer_query"}),
        prefetch_depth=1,
        decode_coalesce=False,
    )
    records = _load_input_records(args)
    candidates = tuple(_iter_specs(records, ttt_config, roles=roles))
    specs = _fingerprinted_specs(
        candidates,
        config=config,
        minimum_pixels=args.minimum_pixels,
        maximum_pixels=args.maximum_pixels,
    )
    selected = tuple(
        item for item in specs if _owns_shard(item[2], args.shard_index, args.shard_count)
    )
    before = cache.disk_size_bytes()
    for spec, source_dataset, _fingerprint in selected:
        materializer.set_source_dataset(source_dataset)
        materializer(spec)
    after = cache.disk_size_bytes()
    payload = {
        **_inspect(cache),
        "stage": args.stage,
        "split": args.split,
        "roles": sorted(roles),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "candidate_chunk_count": len(candidates),
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


def _load_input_records(
    args: argparse.Namespace,
) -> tuple[A2QueryRecord | A5EpisodeRecord, ...]:
    train, evaluation = load_production_manifest_views(
        args.manifest,
        stage=ManifestStage(args.stage),
    )
    return {
        "train": train.records,
        "validation": evaluation.records,
        "all": (*train.records, *evaluation.records),
    }[args.split]


def _verify_inputs(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.minimum_pixels <= 0 or args.maximum_pixels < args.minimum_pixels:
        parser.error("pixel limits must satisfy 0 < minimum <= maximum")
    os.environ["SVCBENCH_VIDEO_ROOT"] = str(args.video_root.resolve())
    config = load_config(args.project_config)
    ttt_config = _load_training_config(args.training_config)
    if ttt_config.stage != args.stage:
        parser.error(
            f"training config stage {ttt_config.stage!r} does not match --stage {args.stage!r}"
        )
    roles = frozenset(args.roles)
    for role in ("state_query", "answer_query"):
        if role in roles and not ttt_config.query_cache_enabled(role):
            parser.error(f"{role} verification requires its cache mode to be inherit")
    cache = _cache(args)
    records = _load_input_records(args)
    candidates = tuple(_iter_specs(records, ttt_config, roles=roles))
    specs = _fingerprinted_specs(
        candidates,
        config=config,
        minimum_pixels=args.minimum_pixels,
        maximum_pixels=args.maximum_pixels,
    )
    missing = tuple(item[2].digest for item in specs if cache.payload_size(item[2]) <= 0)
    integrity = _verify(cache)
    payload = {
        **_inspect(cache),
        **integrity,
        "stage": args.stage,
        "split": args.split,
        "roles": sorted(roles),
        "candidate_chunk_count": len(candidates),
        "unique_chunk_count": len(specs),
        "missing_input_count": len(missing),
        "missing_input_digests": list(missing[:32]),
    }
    print(json.dumps(payload, sort_keys=True))
    return 1 if missing or integrity["corrupt_entries"] else 0


def _iter_specs(
    records: Iterable[A2QueryRecord | A5EpisodeRecord],
    config: ProductionTTTConfig,
    *,
    roles: frozenset[str] = frozenset(("support", "state_query", "answer_query")),
) -> Iterable[tuple[ObservationSpec, str]]:
    for record in records:
        path = _resolve_video_path(record.source_dataset, record.relative_video_path)
        if isinstance(record, A2QueryRecord):
            specs: tuple[ObservationSpec, ...] = ()
            if "support" in roles:
                specs += _a2_support_chunk_specs(record, path)
            if "state_query" in roles:
                specs += (
                    _query_chunk_spec(
                        f"{record.query.runtime.query_id}:state_query",
                        path,
                        record.query.runtime.query_time,
                        reset_soft_state=False,
                        config=config,
                        role="state_query",
                    ),
                )
            if "answer_query" in roles:
                specs += (
                    _query_chunk_spec(
                        f"{record.query.runtime.query_id}:answer_query",
                        path,
                        record.query.runtime.query_time,
                        reset_soft_state=False,
                        config=config,
                        role="answer_query",
                    ),
                )
        else:
            query_time = record.queries[0].runtime.query_time
            chunks = (record.prewarm, *record.supports)
            specs = (
                tuple(
                    SupportChunkSpec(
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
                )
                if "support" in roles
                else ()
            )
            specs += tuple(
                spec
                for index, query in enumerate(record.queries)
                for spec in (
                    _query_chunk_spec(
                        f"{record.episode_id}:q{index}:state_query",
                        path,
                        query.runtime.query_time,
                        reset_soft_state=index > 0,
                        config=config,
                        role="state_query",
                    ),
                    _query_chunk_spec(
                        f"{record.episode_id}:q{index}:answer_query",
                        path,
                        query.runtime.query_time,
                        reset_soft_state=index > 0,
                        config=config,
                        role="answer_query",
                    ),
                )
                if spec.observation_role in roles
            )
        for spec in specs:
            yield spec, record.source_dataset


def _fingerprinted_specs(
    candidates: Iterable[tuple[ObservationSpec, str]],
    *,
    config: ProjectConfig,
    minimum_pixels: int,
    maximum_pixels: int,
) -> tuple[FingerprintedSpec, ...]:
    """Deduplicate exactly as the runtime cache does, including both Query roles."""

    unique: dict[str, FingerprintedSpec] = {}
    for spec, source_dataset in candidates:
        fingerprint = _build_preprocess_fingerprint(
            spec,
            config=config,
            minimum_pixels=minimum_pixels,
            maximum_pixels=maximum_pixels,
            source_dataset=source_dataset,
        )
        unique.setdefault(fingerprint.digest, (spec, source_dataset, fingerprint))
    return tuple(unique.values())


def _owns_shard(
    fingerprint: PreprocessFingerprint,
    shard_index: int,
    shard_count: int,
) -> bool:
    digest = bytes.fromhex(fingerprint.digest)
    return int.from_bytes(digest[:8], "little") % shard_count == shard_index


def _load_training_config(path: Path) -> ProductionTTTConfig:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("ttt_qwen"), dict):
        raise ValueError("training config must contain a ttt_qwen mapping")
    return ProductionTTTConfig.model_validate(raw["ttt_qwen"])


if __name__ == "__main__":
    raise SystemExit(main())
