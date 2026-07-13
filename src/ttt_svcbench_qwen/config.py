"""Environment smoke check for the project."""

from __future__ import annotations

import platform

import torch
import transformers


def environment_summary() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> None:
    for key, value in environment_summary().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
