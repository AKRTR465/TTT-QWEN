#!/usr/bin/env python3
"""Attach comparable SVCBench train3706 provenance to one scorer result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

CONTRACT_VERSION = "svcbench_train3706_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("baseline", "a2_static"))
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--prepared-sft", required=True, type=Path)
    parser.add_argument("--score-annotations", required=True, type=Path)
    parser.add_argument("--scorer", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evaluation-config", required=True, type=Path)
    parser.add_argument("--model-identity", required=True)
    args = parser.parse_args()

    raw = cast(object, json.loads(args.metrics.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError("SVCBench metrics must contain one JSON object")
    payload = cast(dict[str, Any], raw)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("evaluation_scope") != "train_set":
        raise ValueError("SVCBench comparison provenance requires train_set metrics")

    provenance = {
        "contract_version": CONTRACT_VERSION,
        "method": args.method,
        "selection_sha256": _sha256(args.selection),
        "prepared_sft_sha256": _sha256(args.prepared_sft),
        "score_annotations_sha256": _sha256(args.score_annotations),
        "scorer_sha256": _sha256(args.scorer),
        "manifest_sha256": _sha256(args.manifest),
        "evaluation_config_sha256": _sha256(args.evaluation_config),
        "model_identity": args.model_identity,
    }
    metadata["comparison_provenance"] = provenance
    temporary = args.metrics.with_name(f".{args.metrics.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.metrics)
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
