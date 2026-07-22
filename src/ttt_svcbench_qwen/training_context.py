"""Shared activation-lifetime controls for A2 and A5 training."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import cast

import torch


def query_activation_context(enabled: bool) -> AbstractContextManager[object]:
    if not enabled or not torch.cuda.is_available():
        return nullcontext()
    return cast(
        AbstractContextManager[object],
        torch.autograd.graph.save_on_cpu(pin_memory=True),
    )
