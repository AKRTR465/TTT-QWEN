"""Create a deterministic random subset of the production A5 train episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ttt_svcbench_qwen.episode_data import (
    EpisodeSplit,
    load_production_episode_manifest,
    sample_a5_training_manifest,
    write_production_episode_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source = load_production_episode_manifest(args.source)
    subset = sample_a5_training_manifest(
        source,
        fraction=args.fraction,
        seed=args.seed,
        world_size=args.world_size,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output / "dataset_manifest.json"
    failed_path = args.output / "failed.jsonl"
    write_production_episode_manifest(
        subset,
        manifest_path=manifest_path,
        failed_path=failed_path,
    )

    real_train = tuple(
        episode
        for episode in subset.episodes
        if episode.split is EpisodeSplit.TRAIN and episode.loss_weight == 1.0
    )
    padding_train = tuple(
        episode
        for episode in subset.episodes
        if episode.split is EpisodeSplit.TRAIN and episode.loss_weight == 0.0
    )
    task_counts = Counter(episode.task_class for episode in real_train)
    operator_counts = Counter(episode.operator for episode in real_train)
    summary = {
        "status": "complete",
        "source_manifest": str(args.source.resolve()),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_manifest": str(manifest_path.resolve()),
        "output_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "fraction": args.fraction,
        "seed": args.seed,
        "world_size": args.world_size,
        "real_train_episode_count": len(real_train),
        "train_padding_episode_count": len(padding_train),
        "train_support_count": sum(episode.support_count for episode in real_train),
        "train_query_count": sum(episode.query_count for episode in real_train),
        "train_unique_video_count": len(
            {episode.relative_video_path for episode in real_train}
        ),
        "task_episode_counts": dict(sorted(task_counts.items())),
        "operator_episode_counts": dict(sorted(operator_counts.items())),
        "rank_aligned_bucket_count": sum(
            bucket.split is EpisodeSplit.TRAIN for bucket in subset.buckets
        ),
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
