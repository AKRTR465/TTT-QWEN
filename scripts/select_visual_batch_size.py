"""Select the largest H200 visual batch that passes memory, parity and p50 gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttt_svcbench_qwen.batch_tuning import (
    parse_visual_batch_trials,
    select_visual_batch_size,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("a2", "a5"))
    parser.add_argument("--trials", required=True, type=Path)
    args = parser.parse_args()
    trials = parse_visual_batch_trials(
        json.loads(args.trials.read_text(encoding="utf-8"))
    )
    selected = select_visual_batch_size(args.stage, trials)
    print(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
