#!/usr/bin/env python3
"""Prepare an isolated 3,706-row train-set evaluation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ttt_svcbench_qwen.svcbench_train_eval import prepare_train_eval_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--raw-annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dataset-name", default="svcbench_qwen3vl_sft")
    parser.add_argument("--output-dataset-name", default="svcbench_train3706")
    parser.add_argument("--expected-sft-sha256")
    args = parser.parse_args()
    prepared = prepare_train_eval_dataset(
        sft_data=args.sft_data,
        raw_annotations=args.raw_annotations,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        source_dataset_name=args.source_dataset_name,
        output_dataset_name=args.output_dataset_name,
        expected_sft_sha256=args.expected_sft_sha256,
    )
    print(
        json.dumps(
            {
                "output_dir": str(prepared.output_dir),
                "sft_data": str(prepared.sft_data),
                "selection": str(prepared.selection),
                "dataset_name": prepared.dataset_name,
                "row_count": prepared.row_count,
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
