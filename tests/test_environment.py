from __future__ import annotations

import torch
import transformers

from ttt_svcbench_qwen import __version__
from ttt_svcbench_qwen.config import environment_summary


def test_runtime_versions() -> None:
    summary = environment_summary()
    assert summary["python"].startswith("3.12")
    assert torch.__version__.startswith("2.9.0")
    assert transformers.__version__ == "4.57.1"
    assert __version__ == "0.1.0"
