"""Production A2 episode types and causal State-TTT runner.

Inputs: label-free runtime payloads, separate supervision, and one StateTTT model.
Outputs: typed model-forward values used by the formal A2 training runtime.
Forbidden: memory writes, transient runtime checkpoints, or label leakage into model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, cast

import torch
from torch import Tensor

from ttt_svcbench_qwen.data import RuntimeQueryInput
from ttt_svcbench_qwen.input_composer import ComposedInput
from ttt_svcbench_qwen.model import (
    AnswerQueryRequest,
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
    O2DedupContext,
    OfficialWeakSupervision,
    StageATargetBatch,
)
from ttt_svcbench_qwen.state_retriever import RetrieverOutput


@dataclass(frozen=True, slots=True)
class StageASupervisionBatch:
    """Answer targets plus optional state-only labels, kept outside runtime payloads."""

    answer: AnswerTargetLabels
    state: StageATargetBatch | None
    official_weak: tuple[OfficialWeakSupervision, ...] = ()


@dataclass(frozen=True, slots=True)
class StageATrainingBatch:
    runtime_queries: tuple[RuntimeQueryInput, ...]
    model_inputs: object
    supervision: StageASupervisionBatch


@dataclass(frozen=True, slots=True)
class StageAModelForwardOutput:
    """Typed model-side values before training-only labels are joined."""

    answer_logits: Tensor
    composed_input: ComposedInput
    source_input_ids: Tensor
    source_attention_mask: Tensor
    reader_counts: Tensor
    reader_count_valid_mask: Tensor
    observations: ObservationOutputs | None = None
    query: QueryEncoderOutput | None = None
    retrieval: RetrieverOutput | None = None
    metrics: tuple[tuple[str, float | None], ...] = ()
    o2_dedup: O2DedupContext | None = None


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


@dataclass(frozen=True, slots=True)
class StageAEpisodeInputs:
    owner: RuntimeOwner
    observation_requests: tuple[ObservationChunkRequest, ...]
    answer: StageAEpisodeAnswerInputs


def answer_query_request(
    owner: RuntimeOwner,
    observation: ObservationChunkOutput,
    answer: StageAEpisodeAnswerInputs,
) -> AnswerQueryRequest:
    return AnswerQueryRequest(
        owner=owner,
        observation=observation,
        base_input_ids=answer.base_input_ids,
        base_attention_mask=answer.base_attention_mask,
        pixel_values_videos=answer.pixel_values_videos,
        video_grid_thw=answer.video_grid_thw,
        tokenizer=answer.tokenizer,
        embedding_owner=answer.embedding_owner,
        rope_indexer=answer.rope_indexer,
        qwen_kwargs=answer.qwen_kwargs,
    )


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
        self.query_encoder_reuse = query_encoder_reuse
        self.query_activation_offload = query_activation_offload

    def __call__(
        self,
        batch: StageATrainingBatch,
        *,
        training: bool,
    ) -> StageAModelForwardOutput:
        del training
        episode = cast(StageAEpisodeInputs, batch.model_inputs)
        initial = episode.observation_requests[0].runtime_state
        lifecycle = PrefillLifecycle(episode.owner)
        observations: list[ObservationChunkOutput] = []
        runtime = initial
        bank_states = initial.state_bank_states
        prepared_query: PreparedQueryOutput | None = None
        detached_query: PreparedQueryOutput | None = None
        if self.query_encoder_reuse:
            final_request = episode.observation_requests[-1]
            prepared_query = PreparedQueryOutput.bind(
                final_request.query_input,
                self.model.components.query_encoder(
                    final_request.query_input, inference=final_request.inference
                ),
            )
            # Support chunks must consume the DETACHED encoding so their graphs never
            # re-enter the query encoder.
            detached_query = prepared_query.detached()
        pre_query_identity_states = initial.identity_bank_states
        for chunk_index, template in enumerate(episode.observation_requests):
            is_current_query_chunk = chunk_index + 1 == len(episode.observation_requests)
            if is_current_query_chunk:
                # Snapshot the confirmed identities before the query chunk's own hard
                # commit: the O2 soft-dedup base must not contain the current slots.
                pre_query_identity_states = runtime.identity_bank_states
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
                # the differentiable memory-write computation.
                with torch.no_grad():
                    observed = self.model.observe_chunk(request, lifecycle)
            else:
                observed = self.model.observe_chunk(request, lifecycle)
            observations.append(observed)
            runtime = observed.runtime_state
            bank_states = observed.bank_states
        final_observation = observations[-1]
        answer_inputs = episode.answer
        answer_request = answer_query_request(episode.owner, final_observation, answer_inputs)
        output = self.model.prefill_answer(
            self.model.prepare_answer(answer_request, lifecycle),
            lifecycle,
        )
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
        metrics, _ = self.metric_builder(output, batch.supervision)
        observations_output = output.observations
        query_output = output.query
        retrieval_output = output.retrieval
        return StageAModelForwardOutput(
            answer_logits=output.answer_logits,
            composed_input=output.composed,
            source_input_ids=answer_inputs.base_input_ids,
            source_attention_mask=answer_inputs.base_attention_mask,
            reader_counts=reader_counts,
            reader_count_valid_mask=reader_valid,
            observations=observations_output,
            query=query_output,
            retrieval=retrieval_output,
            metrics=metrics,
            o2_dedup=O2DedupContext.from_identity_states(pre_query_identity_states),
        )
