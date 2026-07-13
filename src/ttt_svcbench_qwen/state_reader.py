"""Define deterministic exact-count and number-token result contracts.

Inputs: hard operator, resolved TimeWindow, and complete retrieved typed records.
Outputs: status, exact integer, tokenizer IDs, selected IDs, and audit fields.
Forbidden: neural count regression, ground-truth substitution, retrieval, or natural-language
generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn

from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.query_encoder import Operator, TimeWindow


class ReaderStatus(StrEnum):
    OK = "ok"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


type AuditValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class ReaderResult:
    status: ReaderStatus
    exact_count: int | None
    number_token_ids: tuple[int, ...]
    selected_record_ids: tuple[str, ...]
    operator: Operator
    time_window: TimeWindow
    audit_fields: tuple[tuple[str, AuditValue], ...]

    def __post_init__(self) -> None:
        if self.status is ReaderStatus.OK and (self.exact_count is None or self.exact_count < 0):
            raise ValueError("Reader status ok requires a non-negative exact_count")
        if (
            self.status in (ReaderStatus.UNSUPPORTED, ReaderStatus.INVALID)
            and self.exact_count is not None
        ):
            raise ValueError("unsupported/invalid Reader results cannot contain an exact_count")
        if any(token_id < 0 for token_id in self.number_token_ids):
            raise ValueError("number_token_ids must be non-negative")
        if self.exact_count is None and self.number_token_ids:
            raise ValueError("number tokens cannot be emitted without an exact_count")


def build_state_reader(_config: ProjectConfig | None = None) -> NoReturn:
    """P12 owns operator-specific arithmetic and tokenizer serialization."""

    raise NotImplementedError("Deterministic State Reader implementation is deferred to P12")
