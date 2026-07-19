from __future__ import annotations

import json
from pathlib import Path

from ttt_svcbench_qwen.episode_data import load_visual_cost_index


def test_visual_cost_index_accepts_records_and_ignores_bad_rows(tmp_path: Path) -> None:
    path = tmp_path / "visual_cost_index.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "q1",
                        "frame_budget": 32,
                        "pixel_rate": 100,
                        "encoded_bytes_per_second": 200,
                    },
                    {"record_id": "missing"},
                    {"record_id": 3, "frame_budget": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_visual_cost_index(path) == {"q1": (32, 100, 200)}
