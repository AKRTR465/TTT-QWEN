"""Build an advisory A2 visual-cost sidecar from a production manifest.

The index never changes record ordering or balancing.  It only moves local media-header probes out
of the sampler's first epoch, so each rank can sort by the same immutable estimates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import av

from ttt_svcbench_qwen.episode_data import (
    ManifestStage,
    _a2_visual_length_key,
    load_production_manifest_views,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--video-root", type=Path, default=None)
    args = parser.parse_args()
    if args.video_root is not None:
        os.environ["SVCBENCH_VIDEO_ROOT"] = str(args.video_root.resolve())
    train, _ = load_production_manifest_views(args.manifest, stage=ManifestStage.A2)
    rows: list[dict[str, object]] = []
    root = Path(os.environ.get("SVCBENCH_VIDEO_ROOT", ".")).resolve()
    for record in train.records:
        key = _a2_visual_length_key(record)
        path = (root / record.relative_video_path).resolve()
        if not path.is_file():
            path = (root / record.source_dataset / record.relative_video_path).resolve()
        pixel_rate = encoded_rate = 0
        if path.is_file():
            try:
                with av.open(str(path)) as container:
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
                    pixel_rate = max(0, int(width * height * fps))
                    if duration > 0.0 and math.isfinite(duration):
                        encoded_rate = max(0, int(path.stat().st_size / duration))
            except (OSError, ValueError, TypeError, IndexError, av.error.FFmpegError):
                pass
        rows.append(
            {
                "record_id": record.query.runtime.query_id,
                "frame_budget": key[0],
                "pixel_rate": pixel_rate or key[1],
                "encoded_bytes_per_second": encoded_rate or key[2],
                "estimated_visual_tokens": key[0],
                "estimated_decode_seconds": 0.0,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"records": rows}, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {len(rows)} visual-cost rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
