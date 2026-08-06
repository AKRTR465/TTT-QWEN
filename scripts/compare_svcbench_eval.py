#!/usr/bin/env python3
"""Compare two result files emitted by the shared SVCBench scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

METRICS = ("gpa", "moc", "uda", "exact_match", "exact_match_macro")
CONTRACT_VERSION = "svcbench_train3706_v1"
MATCHED_PROVENANCE_FIELDS = (
    "contract_version",
    "selection_sha256",
    "prepared_sft_sha256",
    "score_annotations_sha256",
    "scorer_sha256",
    "manifest_sha256",
)


def _read(path: Path) -> dict[str, Any]:
    value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict) or not isinstance(value.get("overall"), dict):
        raise ValueError(f"invalid SVCBench metrics file: {path}")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("evaluation_scope") != "train_set":
        raise ValueError(f"comparison requires train_set metrics: {path}")
    return cast(dict[str, Any], value)


def _provenance(
    metrics: dict[str, Any],
    path: Path,
    *,
    expected_method: str,
) -> dict[str, str]:
    metadata = cast(dict[str, Any], metrics["metadata"])
    raw = metadata.get("comparison_provenance")
    if not isinstance(raw, dict):
        raise ValueError(f"metrics lack comparison provenance: {path}")
    required = {
        *MATCHED_PROVENANCE_FIELDS,
        "method",
        "evaluation_config_sha256",
        "model_identity",
    }
    values = {key: raw.get(key) for key in required}
    if any(not isinstance(value, str) or not value for value in values.values()):
        raise ValueError(f"metrics contain incomplete comparison provenance: {path}")
    provenance = cast(dict[str, str], values)
    if provenance["contract_version"] != CONTRACT_VERSION:
        raise ValueError(f"unsupported comparison contract: {path}")
    if provenance["method"] != expected_method:
        raise ValueError(
            f"expected {expected_method} metrics, got {provenance['method']}: {path}"
        )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--a2-static", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = _read(args.baseline)
    a2_static = _read(args.a2_static)
    baseline_provenance = _provenance(
        baseline,
        args.baseline,
        expected_method="baseline",
    )
    a2_provenance = _provenance(
        a2_static,
        args.a2_static,
        expected_method="a2_static",
    )
    for field in MATCHED_PROVENANCE_FIELDS:
        if baseline_provenance[field] != a2_provenance[field]:
            raise ValueError(f"comparison provenance mismatch for {field}")
    base_overall = cast(dict[str, Any], baseline["overall"])
    a2_overall = cast(dict[str, Any], a2_static["overall"])
    for name, value in (("baseline", base_overall), ("a2_static", a2_overall)):
        if int(value.get("n_query_points", -1)) != 3_706:
            raise ValueError(f"{name} did not score exactly 3,706 Query points")

    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        baseline_value = base_overall.get(metric)
        a2_value = a2_overall.get(metric)
        delta = (
            None
            if baseline_value is None or a2_value is None
            else float(a2_value) - float(baseline_value)
        )
        rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "a2_static": a2_value,
                "delta": delta,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_scope": "train_set",
        "row_count": 3_706,
        "baseline_metrics": str(args.baseline.resolve()),
        "a2_static_metrics": str(args.a2_static.resolve()),
        "comparison_provenance": {
            "matched": {
                field: baseline_provenance[field] for field in MATCHED_PROVENANCE_FIELDS
            },
            "baseline": baseline_provenance,
            "a2_static": a2_provenance,
        },
        "rows": rows,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# SVCBench train3706: Baseline vs A2-static",
        "",
        "这是训练集收敛/过拟合评测，不代表 held-out 泛化能力。",
        "",
        "| Metric | Qwen3-VL baseline | TTT A2-static | Delta |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        values = [
            row["metric"],
            _format(row["baseline"]),
            _format(row["a2_static"]),
            _format(row["delta"], signed=True),
        ]
        lines.append(f"| {' | '.join(values)} |")
    (args.output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "comparison.md")
    return 0


def _format(value: object, *, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    return f"{number:+.6f}" if signed else f"{number:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
