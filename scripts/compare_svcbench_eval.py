#!/usr/bin/env python3
"""Compare two result files emitted by the shared SVCBench scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

METRICS = ("gpa", "moc", "uda", "exact_match", "exact_match_macro")


def _read(path: Path) -> dict[str, Any]:
    value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(value, dict) or not isinstance(value.get("overall"), dict):
        raise ValueError(f"invalid SVCBench metrics file: {path}")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("evaluation_scope") != "train_set":
        raise ValueError(f"comparison requires train_set metrics: {path}")
    return cast(dict[str, Any], value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--a2-static", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline = _read(args.baseline)
    a2_static = _read(args.a2_static)
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
