"""Select A2 2x2 or 4x1 DataLoader settings from measured H200 trial summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttt_svcbench_qwen.loader_tuning import parse_loader_trials, select_loader_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", required=True, type=Path)
    args = parser.parse_args()
    trials = parse_loader_trials(json.loads(args.trials.read_text(encoding="utf-8")))
    workers, prefetch_factor = select_loader_profile(trials)
    print(
        json.dumps(
            {
                "dataloader_num_workers": workers,
                "dataloader_prefetch_factor": prefetch_factor,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
