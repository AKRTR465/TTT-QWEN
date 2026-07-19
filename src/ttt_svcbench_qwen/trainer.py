"""P15 Stage A training, optimizer ownership, metrics, and compact checkpoints.

Inputs: label-free runtime payloads, separate supervision, a forward adapter, and config.
Outputs: A1 Answer or A2 State+Answer losses, step audits, metrics, and trainable-only state.
Forbidden: Inner SGD, Predictor/TTT loss, transient W_t, hard runtime checkpoints, or label leakage.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import random
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from ttt_svcbench_qwen.config import ProjectConfig, StageAVariant
from ttt_svcbench_qwen.data import assert_runtime_payload_safe
from ttt_svcbench_qwen.input_composer import ComposedInput, map_teacher_forced_targets
from ttt_svcbench_qwen.losses import (
    AnswerLossInput,
    AnswerLossOutput,
    LossTerm,
    ReaderCountMetricInput,
    StateLossInput,
    StateLossOutput,
    TrainingLossOutput,
    compute_answer_loss,
    compute_state_loss,
)
from ttt_svcbench_qwen.model import (
    AnswerQueryRequest,
    ObservationChunkOutput,
    ObservationChunkRequest,
    PrefillLifecycle,
    RuntimeOwner,
    StateTTTModel,
    StateTTTModelOutput,
)
from ttt_svcbench_qwen.observation_heads import ObservationOutputs
from ttt_svcbench_qwen.query_encoder import QueryEncoderOutput
from ttt_svcbench_qwen.stage_a_targets import (
    AnswerTargetLabels,
    OfficialWeakSupervision,
    StageATargetBatch,
    StageATargetBuilder,
    TargetProvenance,
)
from ttt_svcbench_qwen.state_retriever import RetrieverOutput


class TrainingStage(StrEnum):
    A = "stage_a"
    B = "stage_b"
    C = "stage_c"
    D = "stage_d"


REQUIRED_A2_METRICS: tuple[str, ...] = (
    "o1/soft_count_mae",
    "o1/hard_count_accuracy",
    "o2/duplicate_rate",
    "o2/missed_new_rate",
    "e1/duplicate_rate",
    "e1/miss_rate",
    "e2/duplicate_rate",
    "e2/miss_rate",
    "operator/macro_accuracy",
    "operator/unsupported_rate",
    "retrieval/precision",
    "retrieval/recall",
    "time/mode_accuracy",
    "time/span_exact",
    "reader/llm_number_disagreement_rate",
)


@dataclass(frozen=True, slots=True)
class StageALossOutput:
    variant: StageAVariant
    state: StateLossOutput | None
    answer: AnswerLossOutput
    total: Tensor

    def __post_init__(self) -> None:
        if self.total.ndim != 0 or self.total.dtype != torch.float32:
            raise ValueError("Stage A total must be an FP32 scalar")
        if not bool(torch.isfinite(self.total).item()):
            raise ValueError("Stage A total must be finite")
        if self.variant is StageAVariant.A1 and self.state is not None:
            raise ValueError("A1 is Answer-only and cannot carry State loss")
        if self.variant is StageAVariant.A2 and self.state is None:
            raise ValueError("A2 requires explicit State loss")
        expected = self.answer.loss.value
        if self.state is not None:
            expected = expected + self.state.total
        if not torch.allclose(
            self.total.detach(),
            expected.detach(),
            atol=1.0e-7,
            rtol=1.0e-7,
        ):
            raise ValueError("Stage A total must equal Answer or State+Answer exactly")


def compute_stage_a_losses(
    variant: StageAVariant,
    *,
    answer: AnswerLossInput,
    state: StateLossInput | None,
) -> StageALossOutput:
    """Compute P15 losses without making the P14 TTT path reachable."""

    if not isinstance(variant, StageAVariant):
        raise TypeError("Stage A variant must be a StageAVariant")
    if variant is StageAVariant.A1 and state is not None:
        raise ValueError("A1 cannot receive State supervision")
    if variant is StageAVariant.A2 and state is None:
        raise ValueError("A2 requires State supervision")
    answer_output = compute_answer_loss(answer)
    state_output = None if state is None else compute_state_loss(state)
    total = answer_output.loss.value
    if state_output is not None:
        total = total + state_output.total
    return StageALossOutput(
        variant=variant,
        state=state_output,
        answer=answer_output,
        total=total,
    )


@dataclass(frozen=True, slots=True)
class StageAExecutionAudit:
    """Facts emitted by the forward adapter and checked before every optimizer step."""

    row_count: int
    observed_chunk_count: int
    hard_state_row_count: int
    query_router_row_count: int
    time_resolver_row_count: int
    retrieval_row_count: int
    reader_result_count: int
    bank_reset_count: int
    bank_write_count: int
    cache_advance_count: int
    fsm_rollout_count: int
    decode_step_count: int = 0
    ground_truth_reader_input_count: int = 0
    inner_sgd_attempted: int = 0
    inner_sgd_updated: int = 0
    inner_sgd_skipped: int = 0

    def validate_for(self, variant: StageAVariant) -> None:
        values = (
            tuple(self.__dict__.values())
            if hasattr(self, "__dict__")
            else (
                self.row_count,
                self.observed_chunk_count,
                self.hard_state_row_count,
                self.query_router_row_count,
                self.time_resolver_row_count,
                self.retrieval_row_count,
                self.reader_result_count,
                self.bank_reset_count,
                self.bank_write_count,
                self.cache_advance_count,
                self.fsm_rollout_count,
                self.decode_step_count,
                self.ground_truth_reader_input_count,
                self.inner_sgd_attempted,
                self.inner_sgd_updated,
                self.inner_sgd_skipped,
            )
        )
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("Stage A execution counters must be non-negative integers")
        if self.row_count <= 0:
            raise ValueError("Stage A execution requires at least one row")
        if self.decode_step_count:
            raise ValueError("Stage A teacher forcing must use prefill only, never decode")
        if self.ground_truth_reader_input_count:
            raise ValueError("Reader exact count cannot consume ground-truth labels")
        if any((self.inner_sgd_attempted, self.inner_sgd_updated, self.inner_sgd_skipped)):
            raise ValueError("Stage A cannot attempt, update, or skip Inner SGD")
        state_counts = (
            self.hard_state_row_count,
            self.query_router_row_count,
            self.time_resolver_row_count,
            self.retrieval_row_count,
            self.reader_result_count,
            self.bank_reset_count,
            self.bank_write_count,
            self.cache_advance_count,
            self.fsm_rollout_count,
        )
        if variant is StageAVariant.A1:
            if any(state_counts):
                raise ValueError("A1 must keep every explicit-state component disabled")
        else:
            if self.observed_chunk_count <= 0 or self.hard_state_row_count <= 0:
                raise ValueError("A2 requires causal observation and hard-state rollout")
            if self.hard_state_row_count != self.row_count:
                raise ValueError("A2 hard-state rollout must cover every supported task row")
            row_aligned = (
                self.query_router_row_count,
                self.time_resolver_row_count,
                self.retrieval_row_count,
                self.reader_result_count,
                self.bank_reset_count,
            )
            if any(value != self.row_count for value in row_aligned):
                raise ValueError("A2 router/time/retrieval/Reader/reset must cover every row")
            if self.cache_advance_count <= 0:
                raise ValueError("A2 requires temporal cache advancement")
            # A randomly initialized label-free router may legitimately choose UNSUPPORTED for
            # every row in an early batch.  The hard writer still ran (the episode runner checks
            # its typed audit), but there is intentionally no Bank write to commit.  Requiring a
            # write here would force official operator labels into the runtime path and leak
            # supervision before the loss builder.
            # O1/O2 counting rows legitimately exercise Bank state without an event FSM.
            # Dataset-level task balancing guarantees E1/E2 coverage; a per-batch FSM
            # requirement would reject every valid one-row O1/O2 production batch.


@dataclass(frozen=True, slots=True)
class StageASupervisionBatch:
    """Answer targets plus optional state-only labels, kept outside runtime payloads."""

    answer: AnswerTargetLabels
    state: StageATargetBatch | None
    official_weak: tuple[OfficialWeakSupervision, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.answer, AnswerTargetLabels):
            raise TypeError("Stage A Answer supervision has the wrong type")
        if self.state is not None and not isinstance(self.state, StageATargetBatch):
            raise TypeError("Stage A State supervision has the wrong type")
        if any(not isinstance(value, OfficialWeakSupervision) for value in self.official_weak):
            raise TypeError("Stage A official-weak supervision has the wrong type")
        if self.official_weak and len(self.official_weak) != self.answer.batch_size:
            raise ValueError("Stage A official-weak supervision must align to Answer rows")
        if self.state is not None and self.official_weak:
            raise ValueError("one Stage A batch cannot mix dense and official-weak State labels")


@dataclass(frozen=True, slots=True)
class StageATrainingBatch:
    runtime_payloads: tuple[Mapping[str, object], ...]
    model_inputs: object
    supervision: StageASupervisionBatch

    def __post_init__(self) -> None:
        if not self.runtime_payloads:
            raise ValueError("Stage A batch requires at least one runtime payload")
        if not isinstance(self.supervision, StageASupervisionBatch):
            raise TypeError("Stage A supervision must use StageASupervisionBatch")
        if self.supervision.answer.batch_size != len(self.runtime_payloads):
            raise ValueError("Stage A Answer labels must align to runtime payload rows")


@dataclass(frozen=True, slots=True)
class StageAForwardOutput:
    answer_loss_input: AnswerLossInput
    state_loss_input: StateLossInput | None
    audit: StageAExecutionAudit
    metrics: tuple[tuple[str, float | None], ...] = ()
    failure_cases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.metrics)
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Stage A forward metric names must be unique and non-empty")
        if any(not case for case in self.failure_cases):
            raise ValueError("Stage A failure-case identifiers must be non-empty")
        if any(value is not None and not math.isfinite(value) for _, value in self.metrics):
            raise ValueError("Stage A forward metrics must be finite or N/A")


class StageAForward(Protocol):
    def __call__(
        self,
        batch: StageATrainingBatch,
        *,
        training: bool,
    ) -> StageAForwardOutput: ...


@dataclass(frozen=True, slots=True)
class StageAModelForwardOutput:
    """Typed model-side values before training-only labels are joined."""

    answer_logits: Tensor
    composed_input: ComposedInput
    source_input_ids: Tensor
    source_attention_mask: Tensor
    reader_counts: Tensor
    reader_count_valid_mask: Tensor
    audit: StageAExecutionAudit
    observations: ObservationOutputs | None = None
    query: QueryEncoderOutput | None = None
    retrieval: RetrieverOutput | None = None
    metrics: tuple[tuple[str, float | None], ...] = ()
    failure_cases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        batch_size, sequence_length = self.composed_input.input_ids.shape
        if (
            self.answer_logits.ndim != 3
            or self.answer_logits.shape[:2] != (batch_size, sequence_length)
            or not torch.is_floating_point(self.answer_logits)
        ):
            raise ValueError("Stage A model answer logits must align [B, L_composed, V]")
        if self.source_input_ids.shape != self.source_attention_mask.shape:
            raise ValueError("Stage A source IDs/attention must align")
        if self.source_input_ids.shape[0] != batch_size:
            raise ValueError("Stage A source/composed batches must align")
        if self.source_input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("Stage A source IDs must be integer")
        if self.source_attention_mask.dtype not in (torch.bool, torch.int32, torch.int64):
            raise TypeError("Stage A source attention must be bool/integer")
        if self.reader_counts.shape != (batch_size,) or self.reader_counts.dtype != torch.int64:
            raise ValueError("Stage A Reader counts must be int64 [B]")
        if (
            self.reader_count_valid_mask.shape != (batch_size,)
            or self.reader_count_valid_mask.dtype != torch.bool
        ):
            raise ValueError("Stage A Reader count validity must be bool [B]")
        tensors = (self.answer_logits, self.reader_counts, self.reader_count_valid_mask)
        if any(tensor.device != self.answer_logits.device for tensor in tensors):
            raise ValueError("Stage A model forward tensors must share one device")


class StageAModelForward(Protocol):
    def __call__(
        self,
        batch: StageATrainingBatch,
        *,
        training: bool,
    ) -> StageAModelForwardOutput: ...


@dataclass(frozen=True, slots=True)
class StageAEpisodeAnswerInputs:
    base_input_ids: Tensor
    base_attention_mask: Tensor
    pixel_values_videos: object
    video_grid_thw: object
    tokenizer: object
    embedding_owner: object
    rope_indexer: object
    qwen_kwargs: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.base_input_ids.ndim != 2 or self.base_input_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Stage A answer base IDs must be integer [B, L]")
        if self.base_attention_mask.shape != self.base_input_ids.shape:
            raise ValueError("Stage A answer attention must match base IDs")


@dataclass(frozen=True, slots=True)
class StageAEpisodeInputs:
    owner: RuntimeOwner
    observation_requests: tuple[ObservationChunkRequest, ...]
    answer: StageAEpisodeAnswerInputs

    def __post_init__(self) -> None:
        if not self.observation_requests:
            raise ValueError("Stage A episode requires at least one causal observation chunk")
        if any(request.owner != self.owner for request in self.observation_requests):
            raise ValueError("Stage A episode observation owners must align")
        if self.answer.base_input_ids.shape[0] != len(self.owner.video_ids):
            raise ValueError("Stage A answer rows must align to the runtime owner")


class StageAEpisodeMetricBuilder(Protocol):
    def __call__(
        self,
        output: StateTTTModelOutput,
        supervision: StageASupervisionBatch,
    ) -> tuple[tuple[tuple[str, float | None], ...], tuple[str, ...]]: ...


class StageAEpisodeRunner:
    """Run causal observe chunks and exactly one teacher-forced Qwen prefill."""

    def __init__(
        self,
        *,
        model: StateTTTModel,
        variant: StageAVariant,
        metric_builder: StageAEpisodeMetricBuilder,
    ) -> None:
        self.model = model
        self.variant = variant
        self.metric_builder = metric_builder

    def __call__(
        self,
        batch: StageATrainingBatch,
        *,
        training: bool,
    ) -> StageAModelForwardOutput:
        del training
        episode = batch.model_inputs
        if not isinstance(episode, StageAEpisodeInputs):
            raise TypeError("Stage A episode runner requires StageAEpisodeInputs")
        from ttt_svcbench_qwen.stage_a_runtime import (
            StageABatchRuntime,
            StageAWriteAudit,
        )

        initial = episode.observation_requests[0].runtime_state
        if not isinstance(initial, StageABatchRuntime):
            raise TypeError("Stage A episode must begin from a reset StageABatchRuntime")
        if initial.next_chunk_index != 0 or any(
            state.version != 0 for state in initial.state_bank_states
        ):
            raise ValueError("Stage A episode must reset every owner before the batch")
        lifecycle = PrefillLifecycle(episode.owner)
        observations: list[ObservationChunkOutput] = []
        runtime: object = initial
        bank_states: tuple[object, ...] = tuple(initial.state_bank_states)
        bank_write_count = fsm_rollout_count = cache_advance_count = 0
        for chunk_index, template in enumerate(episode.observation_requests):
            request = replace(template, runtime_state=runtime, bank_states=bank_states)
            is_current_query_chunk = chunk_index + 1 == len(episode.observation_requests)
            if self.variant is StageAVariant.A2 and not is_current_query_chunk:
                # A2's loss is defined on the current Query chunk.  Earlier Support chunks
                # only causally commit detached Bank/FSM/temporal state, so retaining their
                # Qwen activation graphs both violates the bounded-current-token design and
                # lets variable Support counts change the distributed autograd hook schedule.
                # Keep their numerical state transition exactly the same, but do not retain
                # activations.  A5 deliberately does not take this path: its supports carry
                # the differentiable Inner-SGD computation.
                with torch.no_grad():
                    observed = self.model.observe_chunk(request, lifecycle)
            else:
                observed = self.model.observe_chunk(request, lifecycle)
            observations.append(observed)
            runtime = observed.runtime_state
            bank_states = observed.bank_states
            if isinstance(runtime, StageABatchRuntime):
                if runtime.next_chunk_index != chunk_index + 1:
                    raise ValueError("Stage A runtime chunk index did not advance causally")
                cache_advance_count += len(episode.owner.video_ids)
            audit = observed.state_audit
            if self.variant is StageAVariant.A2 and not isinstance(audit, StageAWriteAudit):
                raise TypeError("A2 observation must execute the typed hard-state writer")
            if isinstance(audit, StageAWriteAudit):
                bank_write_count += len(audit.head_types) - len(audit.skipped_rows)
                fsm_rollout_count += sum(
                    head is not None and head.value in {"e1", "e2"} for head in audit.head_types
                )
        final_observation = observations[-1]
        answer_inputs = episode.answer
        answer_request = AnswerQueryRequest(
            owner=episode.owner,
            observation=final_observation,
            base_input_ids=answer_inputs.base_input_ids,
            base_attention_mask=answer_inputs.base_attention_mask,
            pixel_values_videos=answer_inputs.pixel_values_videos,
            video_grid_thw=answer_inputs.video_grid_thw,
            tokenizer=answer_inputs.tokenizer,
            embedding_owner=answer_inputs.embedding_owner,
            rope_indexer=answer_inputs.rope_indexer,
            qwen_kwargs=answer_inputs.qwen_kwargs,
        )
        output = self.model.answer_query(answer_request, lifecycle)
        if not isinstance(output.composed, ComposedInput):
            raise TypeError("Stage A episode Composer must return ComposedInput")
        if not isinstance(output.answer_logits, Tensor):
            raise TypeError("Stage A Qwen prefill must return Tensor logits")
        row_count = len(episode.owner.video_ids)
        reader_counts = torch.full(
            (row_count,),
            -100,
            dtype=torch.int64,
            device=output.answer_logits.device,
        )
        reader_valid = torch.zeros(
            (row_count,),
            dtype=torch.bool,
            device=output.answer_logits.device,
        )
        for row, result in enumerate(output.reader):
            exact_count = getattr(result, "exact_count", None)
            if type(exact_count) is int:
                reader_counts[row] = exact_count
                reader_valid[row] = True
        metrics, failure_cases = self.metric_builder(output, batch.supervision)
        if self.variant is StageAVariant.A2:
            observations_output = output.observations
            query_output = output.query
            retrieval_output = output.retrieval
            if not isinstance(observations_output, ObservationOutputs):
                raise TypeError("A2 episode must expose ObservationOutputs")
            if not isinstance(query_output, QueryEncoderOutput):
                raise TypeError("A2 episode must expose QueryEncoderOutput")
            if not isinstance(retrieval_output, RetrieverOutput):
                raise TypeError("A2 episode must expose RetrieverOutput")
            # This is execution coverage, not the number of currently supported predictions.
            # UNSUPPORTED is a valid pre-training model decision; the official weak operator
            # target is joined only after this label-free runtime forward has completed.
            hard_state_rows = row_count
            router_rows = time_rows = retrieval_rows = row_count
            reader_rows = len(output.reader)
        else:
            observations_output = None
            query_output = None
            retrieval_output = None
            hard_state_rows = router_rows = time_rows = retrieval_rows = reader_rows = 0
        return StageAModelForwardOutput(
            answer_logits=output.answer_logits,
            composed_input=output.composed,
            source_input_ids=answer_inputs.base_input_ids,
            source_attention_mask=answer_inputs.base_attention_mask,
            reader_counts=reader_counts,
            reader_count_valid_mask=reader_valid,
            audit=StageAExecutionAudit(
                row_count=row_count,
                observed_chunk_count=len(observations) * row_count,
                hard_state_row_count=hard_state_rows,
                query_router_row_count=router_rows,
                time_resolver_row_count=time_rows,
                retrieval_row_count=retrieval_rows,
                reader_result_count=reader_rows,
                bank_reset_count=row_count,
                bank_write_count=bank_write_count,
                cache_advance_count=cache_advance_count,
                fsm_rollout_count=fsm_rollout_count,
            ),
            observations=observations_output,
            query=query_output,
            retrieval=retrieval_output,
            metrics=metrics,
            failure_cases=failure_cases,
        )


class StageATypedForwardAdapter:
    """Join model predictions to label-only targets at the last possible boundary."""

    def __init__(
        self,
        variant: StageAVariant,
        model_forward: StageAModelForward,
        *,
        target_builder: StageATargetBuilder | None = None,
    ) -> None:
        self.variant = variant
        self.model_forward = model_forward
        self.target_builder = target_builder or StageATargetBuilder()

    def __call__(
        self,
        batch: StageATrainingBatch,
        *,
        training: bool,
    ) -> StageAForwardOutput:
        raw = self.model_forward(batch, training=training)
        if not isinstance(raw, StageAModelForwardOutput):
            raise TypeError("Stage A model forward must return StageAModelForwardOutput")
        answer_labels = batch.supervision.answer
        mapped = map_teacher_forced_targets(
            composed_input=raw.composed_input,
            source_input_ids=raw.source_input_ids,
            source_attention_mask=raw.source_attention_mask,
            source_labels=answer_labels.base_labels,
            source_number_token_mask=answer_labels.base_number_token_mask,
        )
        count_label_valid = torch.tensor(
            [
                provenance is not TargetProvenance.MISSING
                for provenance in answer_labels.count_provenance
            ],
            dtype=torch.bool,
            device=raw.answer_logits.device,
        )
        count_valid = raw.reader_count_valid_mask & count_label_valid
        answer = AnswerLossInput(
            logits=raw.answer_logits,
            labels=mapped.labels,
            number_token_mask=mapped.number_token_mask,
            reader_counts=ReaderCountMetricInput(
                predicted_counts=raw.reader_counts,
                target_counts=answer_labels.target_counts.to(raw.answer_logits.device),
                valid_mask=count_valid,
            ),
        )
        state: StateLossInput | None = None
        if self.variant is StageAVariant.A2:
            if raw.observations is None or raw.query is None or raw.retrieval is None:
                raise ValueError("A2 model forward must expose Observation/Query/Retrieval outputs")
            if batch.supervision.state is None:
                raise ValueError("A2 requires typed State supervision labels")
            state = self.target_builder(
                raw.observations,
                raw.query,
                raw.retrieval,
                batch.supervision.state,
            )
        else:
            if batch.supervision.state is not None:
                raise ValueError("A1 cannot carry State supervision labels")
            if any(value is not None for value in (raw.observations, raw.query, raw.retrieval)):
                raise ValueError("A1 model forward cannot expose explicit-state outputs")
        return StageAForwardOutput(
            answer_loss_input=answer,
            state_loss_input=state,
            audit=raw.audit,
            metrics=raw.metrics,
            failure_cases=raw.failure_cases,
        )


@dataclass(frozen=True, slots=True)
class StageAParameterAudit:
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    trainable_parameter_count: int
    frozen_parameter_count: int
    qwen_strategy: str
    excluded_inner_parameter_count: int

    def __post_init__(self) -> None:
        if not self.trainable_names or len(set(self.trainable_names)) != len(self.trainable_names):
            raise ValueError("Stage A requires unique trainable parameter names")
        if set(self.trainable_names).intersection(self.frozen_names):
            raise ValueError("Stage A trainable/frozen parameter sets must be disjoint")
        counts = (
            self.trainable_parameter_count,
            self.frozen_parameter_count,
            self.excluded_inner_parameter_count,
        )
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Stage A parameter counts must be non-negative integers")


@dataclass(frozen=True, slots=True)
class StageAStepAudit:
    variant: StageAVariant
    optimizer_step_applied: bool
    skip_reason: str | None
    gradient_norm: float | None
    trainable_parameter_count: int
    failure_case_count: int
    inner_sgd_attempted: int = 0
    inner_sgd_updated: int = 0
    inner_sgd_skipped: int = 0

    def __post_init__(self) -> None:
        if self.optimizer_step_applied != (self.skip_reason is None):
            raise ValueError("Stage A applied-step and skip-reason audit disagree")
        if self.gradient_norm is not None and (
            not math.isfinite(self.gradient_norm) or self.gradient_norm < 0.0
        ):
            raise ValueError("Stage A gradient norm must be finite and non-negative")
        if any((self.inner_sgd_attempted, self.inner_sgd_updated, self.inner_sgd_skipped)):
            raise ValueError("Stage A step audit cannot contain Inner SGD activity")


@dataclass(frozen=True, slots=True)
class TrainingStepOutput:
    stage: TrainingStage
    losses: StageALossOutput | TrainingLossOutput
    global_step: int
    metrics: tuple[tuple[str, float | None], ...]
    checkpoint_path: str | None
    audit: StageAStepAudit | None = None

    def __post_init__(self) -> None:
        if self.global_step < 0:
            raise ValueError("global_step must be non-negative")
        names = tuple(name for name, _ in self.metrics)
        if len(names) != len(set(names)):
            raise ValueError("training-step metric names must be unique")
        if self.stage is TrainingStage.A:
            if not isinstance(self.losses, StageALossOutput) or self.audit is None:
                raise ValueError("Stage A output requires Stage A losses and audit")
            if self.audit.variant is not self.losses.variant:
                raise ValueError("Stage A loss/audit variants must match")
        elif isinstance(self.losses, StageALossOutput) or self.audit is not None:
            raise ValueError("Stage A losses/audit cannot be attached to a later stage")


def configure_stage_a_parameters(
    model: nn.Module,
    config: ProjectConfig,
    *,
    variant: StageAVariant | None = None,
) -> StageAParameterAudit:
    """Apply the production A2 full-outer policy or the retained A1 allowlist."""

    active_variant = config.stage_a.variant if variant is None else variant
    if not isinstance(active_variant, StageAVariant):
        raise TypeError("Stage A parameter policy requires a StageAVariant")
    named = tuple(model.named_parameters())
    if not named:
        raise ValueError("Stage A model exposes no parameters")
    selected: list[str] = []
    frozen: list[str] = []
    excluded_inner = 0
    for name, parameter in named:
        inner_owned = "predictor" in name or "functional_sgd" in name or "transient_w_t" in name
        if active_variant is StageAVariant.A2:
            allowed = True
        else:
            allowed = any(
                fnmatch.fnmatchcase(name, pattern)
                for pattern in config.stage_a.qwen_parameter_allowlist
            )
        allowed = allowed and not inner_owned
        parameter.requires_grad_(allowed)
        if allowed:
            selected.append(name)
        else:
            frozen.append(name)
            if inner_owned:
                excluded_inner += parameter.numel()
    if not selected:
        raise ValueError(
            "Stage A parameter allowlist selected nothing; A1 requires an explicit Qwen allowlist"
        )
    selected_parameters = [parameter for name, parameter in named if name in set(selected)]
    if len({id(parameter) for parameter in selected_parameters}) != len(selected_parameters):
        raise ValueError("Stage A optimizer parameters cannot contain aliases")
    if any(
        token in name
        for name in selected
        for token in ("predictor", "functional_sgd", "transient_w_t")
    ):
        raise ValueError("Stage A selected an Inner-SGD/Predictor-owned parameter")
    return StageAParameterAudit(
        trainable_names=tuple(selected),
        frozen_names=tuple(frozen),
        trainable_parameter_count=sum(parameter.numel() for parameter in selected_parameters),
        frozen_parameter_count=sum(
            parameter.numel() for name, parameter in named if name in frozen
        ),
        qwen_strategy=config.stage_a.qwen_strategy,
        excluded_inner_parameter_count=excluded_inner,
    )


def build_stage_a_optimizer(
    model: nn.Module,
    config: ProjectConfig,
    audit: StageAParameterAudit,
) -> torch.optim.Optimizer:
    parameters_by_name = dict(model.named_parameters())
    parameters = [parameters_by_name[name] for name in audit.trainable_names]
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("Stage A optimizer received a frozen allowlisted parameter")
    optimizer_config = config.stage_a.optimizer
    if optimizer_config.name != "adamw":
        raise ValueError("production A2 supports only explicit AdamW")
    qwen_tokens = ("qwen", "vision", "visual_stage", "merger", "decoder", "llm")
    w0_tokens = ("w0_1", "w0_2", "meta_fast")
    named_parameters = [(name, parameters_by_name[name]) for name in audit.trainable_names]
    w0 = [
        parameter
        for name, parameter in named_parameters
        if any(token in name for token in w0_tokens)
    ]
    qwen = [
        parameter
        for name, parameter in named_parameters
        if not any(token in name for token in w0_tokens)
        and any(token in name.casefold() for token in qwen_tokens)
    ]
    state = [
        parameter
        for name, parameter in named_parameters
        if not any(token in name for token in w0_tokens)
        and not any(token in name.casefold() for token in qwen_tokens)
    ]
    groups = [
        {"params": qwen, "lr": optimizer_config.qwen_learning_rate, "group_name": "qwen"},
        {"params": state, "lr": optimizer_config.state_learning_rate, "group_name": "state"},
        {"params": w0, "lr": optimizer_config.w0_learning_rate, "group_name": "w0"},
    ]
    groups = [group for group in groups if group["params"]]
    return torch.optim.AdamW(
        groups,
        betas=optimizer_config.betas,
        eps=optimizer_config.epsilon,
        weight_decay=optimizer_config.weight_decay,
    )


class StageATrainer:
    def __init__(
        self,
        *,
        config: ProjectConfig,
        model: nn.Module,
        forward_step: StageAForward,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        variant: StageAVariant | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.forward_step = forward_step
        self.variant = config.stage_a.variant if variant is None else variant
        if config.stage_a.inner_sgd_enabled:
            raise ValueError("Stage A config must force Inner SGD off")
        self.parameter_audit = configure_stage_a_parameters(
            model,
            config,
            variant=self.variant,
        )
        self.optimizer = optimizer or build_stage_a_optimizer(model, config, self.parameter_audit)
        self.scheduler = scheduler
        self._validate_optimizer_ownership()
        self.global_step = 0
        self.nonfinite_skip_count = 0

    def train_step(self, batch: StageATrainingBatch) -> TrainingStepOutput:
        self.model.train()
        output = self._forward(batch, training=True)
        losses = compute_stage_a_losses(
            self.variant,
            answer=output.answer_loss_input,
            state=output.state_loss_input,
        )
        self.optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        gradients = tuple(
            parameter.grad
            for parameter in self._trainable_parameters()
            if parameter.grad is not None
        )
        finite = bool(gradients) and all(bool(torch.isfinite(value).all()) for value in gradients)
        gradient_norm: float | None = None
        skip_reason: str | None = None
        if finite:
            norm = torch.nn.utils.clip_grad_norm_(
                self._trainable_parameters(),
                self.config.stage_a.optimizer.grad_clip_norm,
                error_if_nonfinite=False,
            )
            gradient_norm = float(norm.detach().float().item())
            if math.isfinite(gradient_norm):
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1
            else:
                finite = False
        if not finite:
            skip_reason = "no_gradient" if not gradients else "nonfinite_gradient"
            self.nonfinite_skip_count += 1
            self.optimizer.zero_grad(set_to_none=True)
        metrics = _stage_a_metrics(losses, gradient_norm, output.metrics)
        return TrainingStepOutput(
            stage=TrainingStage.A,
            losses=losses,
            global_step=self.global_step,
            metrics=metrics,
            checkpoint_path=None,
            audit=StageAStepAudit(
                variant=self.variant,
                optimizer_step_applied=skip_reason is None,
                skip_reason=skip_reason,
                gradient_norm=gradient_norm,
                trainable_parameter_count=self.parameter_audit.trainable_parameter_count,
                failure_case_count=len(output.failure_cases),
            ),
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def validate_step(self, batch: StageATrainingBatch) -> TrainingStepOutput:
        self.model.eval()
        output = self._forward(batch, training=False)
        losses = compute_stage_a_losses(
            self.variant,
            answer=output.answer_loss_input,
            state=output.state_loss_input,
        )
        return TrainingStepOutput(
            stage=TrainingStage.A,
            losses=losses,
            global_step=self.global_step,
            metrics=_stage_a_metrics(losses, None, output.metrics),
            checkpoint_path=None,
            audit=StageAStepAudit(
                variant=self.variant,
                optimizer_step_applied=False,
                skip_reason="validation",
                gradient_norm=None,
                trainable_parameter_count=self.parameter_audit.trainable_parameter_count,
                failure_case_count=len(output.failure_cases),
            ),
        )

    def save_checkpoint(
        self,
        directory: str | Path,
        *,
        metrics: Mapping[str, float | None],
        fold: int,
        dataset_revision: str,
        annotation_sha256: str,
        architecture_sha256: str,
        git_commit: str,
    ) -> Path:
        return save_stage_a_checkpoint(
            directory,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            parameter_audit=self.parameter_audit,
            variant=self.variant,
            global_step=self.global_step,
            metrics=metrics,
            fold=fold,
            dataset_revision=dataset_revision,
            annotation_sha256=annotation_sha256,
            architecture_sha256=architecture_sha256,
            git_commit=git_commit,
        )

    def _forward(self, batch: StageATrainingBatch, *, training: bool) -> StageAForwardOutput:
        for payload in batch.runtime_payloads:
            assert_trainer_runtime_payload(payload)
        output = self.forward_step(batch, training=training)
        if not isinstance(output, StageAForwardOutput):
            raise TypeError("Stage A forward adapter must return StageAForwardOutput")
        output.audit.validate_for(self.variant)
        if output.audit.row_count != len(batch.runtime_payloads):
            raise ValueError("Stage A forward audit must align to runtime payload rows")
        if self.variant is StageAVariant.A1 and output.state_loss_input is not None:
            raise ValueError("A1 forward adapter cannot emit State supervision")
        if self.variant is StageAVariant.A2 and output.state_loss_input is None:
            raise ValueError("A2 forward adapter must emit State supervision")
        if self.variant is StageAVariant.A2:
            available = {name for name, _ in output.metrics}
            missing = tuple(name for name in REQUIRED_A2_METRICS if name not in available)
            if missing:
                raise ValueError(f"A2 forward metrics are incomplete: {missing}")
        return output

    def _trainable_parameters(self) -> list[nn.Parameter]:
        by_name = dict(self.model.named_parameters())
        return [by_name[name] for name in self.parameter_audit.trainable_names]

    def _validate_optimizer_ownership(self) -> None:
        expected = {id(parameter) for parameter in self._trainable_parameters()}
        actual = {
            id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]
        }
        if actual != expected:
            raise ValueError("Stage A optimizer parameters must exactly equal the allowlist")
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise TypeError("Stage A outer optimizer must be AdamW")
        expected_config = self.config.stage_a.optimizer
        expected_lrs = {
            "qwen": expected_config.qwen_learning_rate,
            "state": expected_config.state_learning_rate,
            "w0": expected_config.w0_learning_rate,
        }
        for group in self.optimizer.param_groups:
            group_name = group.get("group_name")
            if not isinstance(group_name, str) or group_name not in expected_lrs:
                raise ValueError("Stage A AdamW groups require qwen/state/w0 ownership labels")
            actual_values = (
                group["lr"],
                group["weight_decay"],
                tuple(group["betas"]),
                group["eps"],
            )
            expected_values = (
                expected_lrs[group_name],
                expected_config.weight_decay,
                expected_config.betas,
                expected_config.epsilon,
            )
            if actual_values != expected_values:
                raise ValueError("Stage A AdamW hyperparameters must match validated config")


def build_trainer(
    *,
    config: ProjectConfig,
    model: nn.Module,
    forward_step: StageAForward,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    variant: StageAVariant | None = None,
) -> StageATrainer:
    return StageATrainer(
        config=config,
        model=model,
        forward_step=forward_step,
        optimizer=optimizer,
        scheduler=scheduler,
        variant=variant,
    )


def build_balanced_stage_a_indices(
    task_names: Sequence[str],
    *,
    seed: int,
    samples_per_task: int | None = None,
) -> tuple[int, ...]:
    """Deterministically oversample the four explicit task families without global RNG drift."""

    if type(seed) is not int or seed < 0:
        raise ValueError("balanced Stage A seed must be non-negative")
    allowed = ("o1", "o2", "e1", "e2")
    buckets: dict[str, list[int]] = {name: [] for name in allowed}
    for index, name in enumerate(task_names):
        if name not in buckets:
            raise ValueError(f"unknown Stage A task family: {name}")
        buckets[name].append(index)
    if any(not values for values in buckets.values()):
        raise ValueError("balanced Stage A sampling requires all four task families")
    target = max(len(values) for values in buckets.values())
    if samples_per_task is not None:
        if type(samples_per_task) is not int or samples_per_task <= 0:
            raise ValueError("samples_per_task must be a positive integer")
        target = samples_per_task
    rng = random.Random(seed)
    balanced: list[int] = []
    for name in allowed:
        source = buckets[name]
        cycles, remainder = divmod(target, len(source))
        selected = source * cycles + rng.sample(source, remainder)
        rng.shuffle(selected)
        balanced.extend(selected)
    rng.shuffle(balanced)
    return tuple(balanced)


def save_stage_a_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: ProjectConfig,
    parameter_audit: StageAParameterAudit,
    variant: StageAVariant,
    global_step: int,
    metrics: Mapping[str, float | None],
    fold: int,
    dataset_revision: str,
    annotation_sha256: str,
    architecture_sha256: str,
    git_commit: str,
) -> Path:
    """Atomically write the configured A2 model and complete resume metadata.

    The production LLaMA-Factory path uses its native distributed save hooks.  This
    standalone serializer mirrors the same ownership policy for CPU/tiny audits:
    checkpointed module state is complete, while transient fast weights and hard
    Bank/FSM/cache runtime are excluded by name and by registration contract.
    """

    checkpoint = config.stage_a.checkpoint
    if checkpoint.trainable_only or not checkpoint.save_full_model:
        raise ValueError("production A2 checkpoints must save the complete module state")
    if checkpoint.save_runtime_state:
        raise ValueError("production A2 checkpoints must exclude hard runtime state")
    if checkpoint.include_scheduler and scheduler is None:
        raise ValueError("production A2 checkpoint requires scheduler state")
    if global_step < 0 or fold < 0:
        raise ValueError("checkpoint global_step/fold must be non-negative")
    required_text = {
        "dataset_revision": dataset_revision,
        "annotation_sha256": annotation_sha256,
        "architecture_sha256": architecture_sha256,
        "git_commit": git_commit,
    }
    if any(not value for value in required_text.values()):
        raise ValueError("checkpoint provenance strings must be non-empty")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    forbidden_tokens = (
        "transient_w_t",
        "state_bank_runtime",
        "identity_bank_runtime",
        "fsm_runtime",
        "temporal_cache",
        "visual_cache",
        "soft_overlap_snapshot",
    )
    full_state = model.state_dict()
    forbidden_registered = tuple(
        name for name in full_state if any(token in name.casefold() for token in forbidden_tokens)
    )
    if forbidden_registered:
        raise ValueError(
            "hard/transient runtime state was registered on the checkpointed model: "
            f"{forbidden_registered}"
        )
    saved_state = {name: value.detach().cpu().contiguous() for name, value in full_state.items()}
    weights_path = root / "model.safetensors"
    state_path = root / "training_state.pt"
    manifest_path = root / "manifest.json"
    nonce = f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary_weights = root / f"model{nonce}.safetensors"
    temporary_state = root / f"training_state{nonce}.pt"
    temporary_manifest = root / f"manifest{nonce}.json"
    save_file(saved_state, str(temporary_weights))
    training_state: dict[str, object] = {
        "global_step": global_step,
        "variant": variant.value,
    }
    if checkpoint.include_optimizer:
        training_state["optimizer"] = optimizer.state_dict()
    if checkpoint.include_scheduler:
        if scheduler is None:  # guarded above; keeps the resume artifact type explicit
            raise RuntimeError("scheduler disappeared during checkpoint serialization")
        training_state["scheduler"] = scheduler.state_dict()
    if checkpoint.include_rng:
        training_state["python_rng"] = random.getstate()
        training_state["torch_rng"] = torch.get_rng_state()
        if torch.cuda.is_available():
            training_state["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(training_state, temporary_state)
    config_json = config.model_dump_json()
    manifest: dict[str, object] = {
        "schema": checkpoint.format,
        "stage": TrainingStage.A.value,
        "variant": variant.value,
        "global_step": global_step,
        "fold": fold,
        "seed": config.stage_a.seed,
        "dataset_revision": dataset_revision,
        "annotation_sha256": annotation_sha256,
        "architecture_sha256": architecture_sha256,
        "git_commit": git_commit,
        "spec_version": config.spec_version,
        "config_sha256": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        "tokenizer_revision": config.input_composer.tokenizer_revision,
        "tokenizer_manifest_sha256": config.state_reader.tokenizer_manifest_sha256,
        "composer_special_token_ids": list(config.input_composer.special_token_ids),
        "qwen_strategy": config.stage_a.qwen_strategy,
        "saved_state_names": list(saved_state),
        "trainable_names": list(parameter_audit.trainable_names),
        "frozen_names": list(parameter_audit.frozen_names),
        "trainable_parameter_count": parameter_audit.trainable_parameter_count,
        "frozen_parameter_count": parameter_audit.frozen_parameter_count,
        "metrics": dict(metrics),
        "excluded_runtime": [
            "transient_W_t",
            "functional_sgd",
            "state_bank_runtime",
            "identity_bank_runtime",
            "fsm_runtime",
            "temporal_cache",
            "soft_overlap_snapshot",
        ],
    }
    temporary_weights.replace(weights_path)
    temporary_state.replace(state_path)
    manifest["artifacts"] = {
        "model.safetensors": _sha256_file(weights_path),
        "training_state.pt": _sha256_file(state_path),
    }
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return root


def load_stage_a_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    config: ProjectConfig,
    parameter_audit: StageAParameterAudit,
    variant: StageAVariant,
    restore_rng: bool = True,
) -> int:
    """Verify and restore complete A2 module/optimizer/scheduler/RNG state."""

    root = Path(directory)
    weights_path = root / "model.safetensors"
    state_path = root / "training_state.pt"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != config.stage_a.checkpoint.format:
        raise ValueError("Stage A checkpoint schema mismatch")
    if (
        manifest.get("variant") != variant.value
        or manifest.get("spec_version") != config.spec_version
    ):
        raise ValueError("Stage A checkpoint variant/spec mismatch")
    config_sha = hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()
    if manifest.get("config_sha256") != config_sha:
        raise ValueError("Stage A checkpoint config hash mismatch")
    expected_trainable_names = list(parameter_audit.trainable_names)
    if manifest.get("trainable_names") != expected_trainable_names:
        raise ValueError("Stage A checkpoint trainable allowlist mismatch")
    current_state = model.state_dict()
    expected_state_names = list(current_state)
    if manifest.get("saved_state_names") != expected_state_names:
        raise ValueError("Stage A checkpoint complete module-state names mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts != {
        "model.safetensors": _sha256_file(weights_path),
        "training_state.pt": _sha256_file(state_path),
    }:
        raise ValueError("Stage A checkpoint artifact hash mismatch")
    saved = load_file(str(weights_path))
    if set(saved) != set(expected_state_names):
        raise ValueError("Stage A checkpoint tensor names mismatch")
    with torch.no_grad():
        for name in expected_state_names:
            target = current_state[name]
            source = saved[name].to(device=target.device, dtype=target.dtype)
            if source.shape != target.shape:
                raise ValueError(f"Stage A checkpoint tensor shape mismatch: {name}")
            target.copy_(source)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("variant") != variant.value:
        raise ValueError("Stage A training-state variant mismatch")
    if config.stage_a.checkpoint.include_optimizer:
        optimizer.load_state_dict(state["optimizer"])
    if config.stage_a.checkpoint.include_scheduler:
        if scheduler is None:
            raise ValueError("restoring production A2 requires a scheduler instance")
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng and config.stage_a.checkpoint.include_rng:
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    global_step = state.get("global_step")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("Stage A checkpoint global_step is invalid")
    return global_step


def assert_trainer_runtime_payload(payload: Mapping[str, object]) -> None:
    """P2 leakage guard applied before any trainer/model handoff."""

    assert_runtime_payload_safe(payload, layer="Trainer")


def _stage_a_metrics(
    losses: StageALossOutput,
    gradient_norm: float | None,
    additional: Sequence[tuple[str, float | None]],
) -> tuple[tuple[str, float | None], ...]:
    values: list[tuple[str, float | None]] = [
        ("loss/total", float(losses.total.detach().item())),
        ("loss/answer", _term_value(losses.answer.loss)),
        ("answer/token_accuracy", _term_value(losses.answer.teacher_forced_token_accuracy)),
        ("answer/number_token_accuracy", _term_value(losses.answer.number_token_accuracy)),
        ("answer/exact_match", _term_value(losses.answer.answer_exact_match)),
        ("reader/exact_count_accuracy", _term_value(losses.answer.reader_exact_count_accuracy)),
        ("outer/gradient_norm", gradient_norm),
    ]
    if losses.state is not None:
        state = losses.state
        values.extend(
            (
                ("loss/state", float(state.total.detach().item())),
                ("state/o1", _term_value(state.o1)),
                ("state/o2", _term_value(state.o2)),
                ("state/e1", _term_value(state.e1)),
                ("state/e2", _term_value(state.e2)),
                ("state/operator", _term_value(state.operator)),
                ("state/retrieval", _term_value(state.retrieval)),
                ("state/time_mode", _term_value(state.time.mode)),
                ("state/time_start", _term_value(state.time.start)),
                ("state/time_end", _term_value(state.time.end)),
            )
        )
    names = {name for name, _ in values}
    for name, value in additional:
        if name in names:
            raise ValueError(f"duplicate Stage A metric: {name}")
        names.add(name)
        values.append((name, value))
    return tuple(values)


def _term_value(term: LossTerm) -> float | None:
    if not bool(term.row_valid_mask.any().item()):
        return None
    return float(term.value.detach().item())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
