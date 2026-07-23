from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_operator_diagnostics_script_renders_latest_jsonl_metrics(tmp_path: Path) -> None:
    source = tmp_path / "metrics.jsonl"
    source.write_text(
        json.dumps(
            {
                "metrics": {
                    "operator/support/o1-snap": 4.0,
                    "operator/recall/o1-snap": 0.75,
                    "operator/raw_loss/o1-snap": 1.25,
                    "operator/effective_confusion/o1-snap/unsupported": 1.0,
                    "operator/micro_accuracy": 0.5,
                    "operator/macro_recall": 0.6,
                    "operator/predicted_unsupported_rate": 0.1,
                    "operator/temperature": 0.9,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    script = Path(__file__).parents[1] / "scripts" / "summarize_operator_diagnostics.py"
    result = subprocess.run(
        [sys.executable, str(script), str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "| o1-snap | 4 | 0.750000 | 1.250000 | 1 |" in result.stdout
    assert "macro_recall: 0.600000" in result.stdout
