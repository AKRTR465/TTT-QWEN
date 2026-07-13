"""Define top-level State-TTT model composition and unified outputs.

Inputs: validated component instances, VideoBatch, question embeddings, and runtime state.
Outputs: component outputs assembled without duplicating their internal algorithms.
Forbidden: local copies of Adapter, Retriever, Reader, Bank, loss, FSM, or training logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

from ttt_svcbench_qwen.config import ProjectConfig
from ttt_svcbench_qwen.observation_heads import ObservationOutputs
from ttt_svcbench_qwen.query_encoder import QueryEncoderOutput
from ttt_svcbench_qwen.qwen_adapter import QwenVisualOutput
from ttt_svcbench_qwen.state_encoder import SpatialEncoderOutput, TemporalEncoderOutput
from ttt_svcbench_qwen.state_reader import ReaderResult
from ttt_svcbench_qwen.state_retriever import RetrieverOutput


@dataclass(frozen=True, slots=True)
class StateTTTModelOutput:
    visual: QwenVisualOutput
    query: QueryEncoderOutput
    spatial: SpatialEncoderOutput
    temporal: TemporalEncoderOutput
    observations: ObservationOutputs
    retrieval: RetrieverOutput
    reader: ReaderResult


def build_model(_config: ProjectConfig | None = None) -> NoReturn:
    """P13 owns dependency injection and the unified forward orchestration."""

    raise NotImplementedError("Top-level model orchestration is deferred to P13")
