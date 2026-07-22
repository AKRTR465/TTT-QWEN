"""Production A2 episode types and causal State-TTT runner.

Inputs: label-free runtime payloads, separate supervision, and one StateTTT model.
Outputs: typed model-forward values used by the formal A2 training runtime.
Forbidden: Inner SGD, transient runtime checkpoints, or label leakage into model inputs.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from typing import Protocol, cast

import torch
from torch import Tensor

from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.input_composer import ComposedInput
from ttt_svcbench_qwen.model import (
    AnswerQueryRequest,
    BatchRuntimeState,
    ObservationChunkOutput,
    ObservationChunkRequest,
    PrefillLifecycle,
    PreparedQueryOutput,
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
)
from ttt_svcbench_qwen.state_retriever import RetrieverOutput


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

    def validate(self) -> None:
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
        # every row in an early batch. The hard writer still ran (the episode runner checks
        # its typed audit), but there is intentionally no Bank write to commit. Requiring a
        # write here would force official operator labels into the runtime path and leak
        # supervision before the loss builder. O1/O2 rows also need no event FSM rollout.


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
    runtime_queries: tuple[RuntimeQueryInput, ...]
    model_inputs: object
    supervision: StageASupervisionBatch

    def __post_init__(self) -> None:
        if not self.runtime_queries:
            raise ValueError("Stage A batch requires at least one runtime Query")
        if any(not isinstance(value, RuntimeQueryInput) for value in self.runtime_queries):
            raise TypeError("Stage A runtime rows must use RuntimeQueryInput")
        if not isinstance(self.supervision, StageASupervisionBatch):
            raise TypeError("Stage A supervision must use StageASupervisionBatch")
        if self.supervision.answer.batch_size != len(self.runtime_queries):
            raise ValueError("Stage A Answer labels must align to runtime Query rows")


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


@dataclass(frozen=True, slots=True)
class StageAEpisodeAnswerInputs:
    base_input_ids: Tensor
    base_attention_mask: Tensor
    pixel_values_videos: Tensor
    video_grid_thw: Tensor
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
        metric_builder: StageAEpisodeMetricBuilder,
        query_encoder_reuse: bool = False,
        query_activation_offload: bool = False,
    ) -> None:
        self.model = model
        self.metric_builder = metric_builder
        if type(query_encoder_reuse) is not bool:
            raise TypeError("query_encoder_reuse must be bool")
        self.query_encoder_reuse = query_encoder_reuse
        if type(query_activation_offload) is not bool:
            raise TypeError("query_activation_offload must be bool")
        self.query_activation_offload = query_activation_offload

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
        from ttt_svcbench_qwen.stage_a_runtime import StageAWriteAudit

        initial = episode.observation_requests[0].runtime_state
        if not isinstance(initial, BatchRuntimeState):
            raise TypeError("Stage A episode must begin from a reset BatchRuntimeState")
        if initial.next_chunk_index != 0 or any(
            state.version != 0 for state in initial.state_bank_states
        ):
            raise ValueError("Stage A episode must reset every owner before the batch")
        lifecycle = PrefillLifecycle(episode.owner)
        observations: list[ObservationChunkOutput] = []
        runtime = initial
        bank_states = initial.state_bank_states
        bank_write_count = fsm_rollout_count = cache_advance_count = 0
        prepared_query: PreparedQueryOutput | None = None
        detached_query: PreparedQueryOutput | None = None
        if self.query_encoder_reuse:
            final_request = episode.observation_requests[-1]
            prepared_query = PreparedQueryOutput.bind(
                final_request.query_input,
                self.model.components.query_encoder(
                    final_request.query_input,
                    inference=final_request.inference,
                ),
            )
            detached_query = prepared_query.detached()
        for chunk_index, template in enumerate(episode.observation_requests):
            is_current_query_chunk = chunk_index + 1 == len(episode.observation_requests)
            request = replace(
                template,
                runtime_state=runtime,
                bank_states=bank_states,
                prepared_query=(prepared_query if is_current_query_chunk else detached_query),
            )
            if not is_current_query_chunk:
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
                with self._query_activation_context():
                    observed = self.model.observe_chunk(request, lifecycle)
            observations.append(observed)
            runtime = observed.runtime_state
            bank_states = observed.bank_states
            if runtime.next_chunk_index != chunk_index + 1:
                raise ValueError("Stage A runtime chunk index did not advance causally")
            cache_advance_count += len(episode.owner.video_ids)
            audit = observed.state_audit
            if not isinstance(audit, StageAWriteAudit):
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
        with self._query_activation_context():
            output = self.model.prefill_answer(
                self.model.prepare_answer(answer_request, lifecycle),
                lifecycle,
            )
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

    def _query_activation_context(self) -> AbstractContextManager[object]:
        if not self.query_activation_offload or not torch.cuda.is_available():
            return nullcontext()
        return cast(
            AbstractContextManager[object],
            torch.autograd.graph.save_on_cpu(pin_memory=True),
        )
