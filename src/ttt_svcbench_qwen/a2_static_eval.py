"""Distributed A2-static generation over a fixed SVCBench train-set selection.

``A2-static`` means the learned slow state ``W0`` is used for every visual pass.  No
``OnlineTTTUpdater`` is constructed, no transient fast matrices exist, and no Inner SGD can
run.  Support and State Query chunks still update the hard Bank/FSM state, and the Answer
Query still consumes Reader/State Tokens exactly as in A2.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn

from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    ManifestStage,
    load_production_manifest_views,
)
from ttt_svcbench_qwen.llamafactory_trainer import ProductionTrainerRuntime
from ttt_svcbench_qwen.model import (
    BatchRuntimeState,
    ObservationChunkOutput,
    PrefillLifecycle,
    PreparedQueryOutput,
    StateTTTGenerationOutput,
    StateTTTModel,
)
from ttt_svcbench_qwen.production_factory import (
    DEFAULT_LLAMFACTORY_ROOT,
    load_llamafactory_backbone,
    load_outer_checkpoint,
)
from ttt_svcbench_qwen.production_runtime import (
    PreparedA2Record,
    ProductionA2LossStep,
    build_runtime,
)
from ttt_svcbench_qwen.svcbench_train_eval import (
    TrainEvalSelection,
    read_selection,
    read_sft_rows,
)
from ttt_svcbench_qwen.trainer import (
    StageAEpisodeAnswerInputs,
    StageAEpisodeInputs,
    StageATrainingBatch,
    answer_query_request,
    prepared_query_pair,
)


@dataclass(frozen=True, slots=True)
class A2StaticGenerationAudit:
    mode: str
    support_count: int
    observed_chunk_count: int
    inner_sgd_attempted: int
    inner_sgd_updated: int
    fast_state_row_count: int
    reader_status: str
    reader_exact_count: int | None
    reader_selected_record_count: int
    lifecycle_observation_count: int
    lifecycle_prefill_count: int
    lifecycle_decode_count: int

    def __post_init__(self) -> None:
        if self.mode != "a2_static":
            raise ValueError("static generation audit has the wrong mode")
        if any(
            value != 0
            for value in (
                self.inner_sgd_attempted,
                self.inner_sgd_updated,
                self.fast_state_row_count,
            )
        ):
            raise ValueError("A2-static cannot expose fast-state or Inner-SGD activity")


def prompt_only_answer_inputs(
    answer: StageAEpisodeAnswerInputs,
    labels: Tensor,
) -> StageAEpisodeAnswerInputs:
    """Remove the teacher-forced answer while preserving the exact native prompt prefix."""

    if labels.shape != answer.base_input_ids.shape or labels.ndim != 2:
        raise ValueError("answer labels must align to one prompt batch")
    if labels.shape[0] != 1:
        raise ValueError("A2-static evaluation requires one row per rank")
    supervised = torch.nonzero(labels[0].ne(-100), as_tuple=False).flatten()
    if supervised.numel() == 0:
        raise ValueError("teacher-forced source has no assistant answer boundary")
    prompt_length = int(supervised[0].item())
    if prompt_length <= 0 or prompt_length >= answer.base_input_ids.shape[1]:
        raise ValueError("derived generation prompt length is invalid")
    if not bool(torch.all(labels[0, :prompt_length].eq(-100))):
        raise ValueError("assistant supervision starts before the derived prompt boundary")
    return replace(
        answer,
        base_input_ids=answer.base_input_ids[:, :prompt_length],
        base_attention_mask=answer.base_attention_mask[:, :prompt_length],
        qwen_kwargs=(("use_cache", True),),
    )


@torch.inference_mode()  # type: ignore[untyped-decorator]
def generate_a2_static(
    batch: StageATrainingBatch,
    *,
    model: StateTTTModel,
    query_encoder_reuse: bool,
    max_new_tokens: int,
) -> tuple[StateTTTGenerationOutput, A2StaticGenerationAudit]:
    episode = batch.model_inputs
    if not isinstance(episode, StageAEpisodeInputs):
        raise TypeError("A2-static requires StageAEpisodeInputs")
    initial = episode.observation_requests[0].runtime_state
    if not isinstance(initial, BatchRuntimeState) or initial.next_chunk_index != 0:
        raise ValueError("A2-static episode must begin from reset runtime state")
    _assert_no_fast_state(initial)

    lifecycle = PrefillLifecycle(episode.owner)
    runtime = initial
    bank_states = initial.state_bank_states
    final_observation: ObservationChunkOutput | None = None
    prepared_query: PreparedQueryOutput | None = None
    detached_query: PreparedQueryOutput | None = None
    if query_encoder_reuse:
        final_template = episode.observation_requests[-1]
        prepared_query, detached_query = prepared_query_pair(model, final_template, inference=True)

    for chunk_index, template in enumerate(episode.observation_requests):
        is_state_query = chunk_index + 1 == len(episode.observation_requests)
        observation_request = replace(
            template,
            inference=True,
            runtime_state=runtime,
            bank_states=bank_states,
            prepared_query=(prepared_query if is_state_query else detached_query),
        )
        observed = model.observe_chunk(observation_request, lifecycle)
        runtime = observed.runtime_state
        bank_states = observed.bank_states
        _assert_no_fast_state(runtime)
        if runtime.next_chunk_index != chunk_index + 1:
            raise ValueError("A2-static runtime did not advance exactly once per chunk")
        final_observation = observed
    if final_observation is None:
        raise RuntimeError("A2-static produced no causal observation")

    answer = prompt_only_answer_inputs(
        episode.answer,
        batch.supervision.answer.base_labels,
    )
    answer_request = answer_query_request(episode.owner, final_observation, answer)
    generated = model.generate_answer(
        model.prepare_answer(answer_request, lifecycle),
        lifecycle,
        max_new_tokens=max_new_tokens,
    )
    _assert_no_fast_state(generated.runtime_state)
    lifecycle_audit = generated.lifecycle
    if (
        lifecycle_audit.observation_count != len(episode.observation_requests)
        or lifecycle_audit.prefill_count != 1
        or lifecycle_audit.decode_count != 0
    ):
        raise ValueError("A2-static lifecycle counts are inconsistent")
    reader = generated.reader[0] if generated.reader else None
    audit = A2StaticGenerationAudit(
        mode="a2_static",
        support_count=len(episode.observation_requests) - 1,
        observed_chunk_count=len(episode.observation_requests),
        inner_sgd_attempted=0,
        inner_sgd_updated=0,
        fast_state_row_count=0,
        reader_status=(
            "disabled" if reader is None else str(getattr(reader.status, "value", reader.status))
        ),
        reader_exact_count=None if reader is None else reader.exact_count,
        reader_selected_record_count=(
            0 if reader is None else len(reader.selected_record_ids)
        ),
        lifecycle_observation_count=lifecycle_audit.observation_count,
        lifecycle_prefill_count=lifecycle_audit.prefill_count,
        lifecycle_decode_count=lifecycle_audit.decode_count,
    )
    return generated, audit


def _assert_no_fast_state(runtime: BatchRuntimeState) -> None:
    if any(row.fast_weights is not None or row.optimizer is not None for row in runtime.rows):
        raise ValueError("A2-static runtime unexpectedly contains transient fast state")


def _prepare_support_batch(
    batch: StageATrainingBatch,
    loss_step: ProductionA2LossStep,
) -> StageATrainingBatch:
    episode = batch.model_inputs
    if not isinstance(episode, StageAEpisodeInputs):
        raise TypeError("materialized A2 batch has the wrong model input type")
    support_requests = episode.observation_requests[:-1]
    if loss_step.support_visual_batch_size <= 1 or len(support_requests) <= 1:
        return batch
    prepared = loss_step.visual_runtime.prepare_support_batch(
        tuple(request.video_input for request in support_requests),
        batch_size=loss_step.support_visual_batch_size,
    )
    return replace(
        batch,
        model_inputs=replace(
            episode,
            observation_requests=tuple(
                replace(request, video_input=value)
                for request, value in zip(support_requests, prepared, strict=True)
            )
            + (episode.observation_requests[-1],),
        ),
    )


def _selection_label(sft_row: dict[str, Any], selection: TrainEvalSelection) -> str:
    messages = sft_row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                isinstance(message, dict)
                and message.get("role") == "assistant"
                and isinstance(message.get("content"), str)
            ):
                value = cast(str, message["content"]).strip()
                if value != selection.label:
                    raise ValueError("filtered SFT label drifted from the fixed selection")
                return value
    raise ValueError("filtered SFT row has no assistant label")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = cast(object, json.loads(line))
                if not isinstance(value, dict):
                    raise ValueError(f"non-object row in {path}")
                rows.append(cast(dict[str, Any], value))
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sft-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--llamafactory-root",
        type=Path,
        default=Path(DEFAULT_LLAMFACTORY_ROOT),
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--allow-failures", action="store_true")
    return parser.parse_args()


def _initialize_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("A2-static production evaluation requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def main() -> int:
    args = _parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    rank, local_rank, world_size, device = _initialize_distributed()
    started = time.monotonic()
    output_dir = args.output_dir.resolve()
    rank_dir = output_dir / "rank_outputs"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        rank_dir.mkdir(parents=True, exist_ok=True)
    _barrier(world_size)

    selections = read_selection(args.selection)
    sft_rows = read_sft_rows(args.sft_data)
    if len(selections) != len(sft_rows):
        raise ValueError("selection and filtered SFT rows must have equal length")
    train_view, _ = load_production_manifest_views(
        args.dataset_manifest,
        stage=ManifestStage.A2,
    )
    if len(train_view) != len(selections):
        raise ValueError("A2 train manifest and fixed selection have different row counts")
    manifest_by_id = {
        record.query.runtime.query_id: record
        for record in train_view.records
        if isinstance(record, A2QueryRecord)
    }

    backbone = load_llamafactory_backbone(
        args.yaml,
        llamafactory_root=args.llamafactory_root,
    )
    runtime_value = build_runtime(backbone, backbone.ttt_config)
    if not isinstance(runtime_value, ProductionTrainerRuntime):
        raise TypeError("runtime factory did not return ProductionTrainerRuntime")
    runtime = runtime_value
    model_owner = runtime.model
    loss_step = runtime.stage_a_loss_step
    if not isinstance(model_owner, nn.Module) or not isinstance(loss_step, ProductionA2LossStep):
        raise TypeError("A2-static requires the production A2 runtime")
    load_outer_checkpoint(model_owner, args.checkpoint)
    model_owner.to(device=device)
    model_owner.requires_grad_(False)
    model_owner.eval()
    disable_gc = getattr(model_owner, "gradient_checkpointing_disable", None)
    if callable(disable_gc):
        disable_gc()
    state_model = loss_step.runner.model
    if not isinstance(state_model, StateTTTModel):
        raise TypeError("production A2 runner lost its StateTTTModel")

    torch.cuda.reset_peak_memory_stats(device)
    rank_predictions = rank_dir / f"rank_{rank}.jsonl"
    rank_failures = rank_dir / f"rank_{rank}_failed.jsonl"
    if rank_predictions.exists() or rank_failures.exists():
        raise FileExistsError(f"rank output already exists for rank {rank}")

    succeeded = failed = 0
    for selection in selections:
        if selection.selection_index % world_size != rank:
            continue
        record = manifest_by_id.get(selection.manifest_query_id)
        if record is None:
            raise KeyError(f"missing manifest Query {selection.manifest_query_id}")
        label = _selection_label(sft_rows[selection.selection_index], selection)
        sample_started = time.monotonic()
        try:
            collated = runtime.data_collator([record])
            if not isinstance(collated, dict):
                raise TypeError("A2 collator must return a mapping")
            prepared = collated.get("prepared_a2")
            if not isinstance(prepared, PreparedA2Record):
                raise TypeError("A2 collator did not return PreparedA2Record")
            batch = loss_step.materializer.a2(prepared)
            batch = _prepare_support_batch(batch, loss_step)
            generated, audit = generate_a2_static(
                batch,
                model=state_model,
                query_encoder_reuse=loss_step.runner.query_encoder_reuse,
                max_new_tokens=args.max_new_tokens,
            )
            payload = {
                "selection_index": selection.selection_index,
                "q_id": selection.q_id,
                "manifest_query_id": selection.manifest_query_id,
                "predict": generated.answer_text,
                "label": label,
                "elapsed_seconds": time.monotonic() - sample_started,
                "audit": asdict(audit),
            }
            _append_jsonl(rank_predictions, payload)
            succeeded += 1
        except Exception as exc:
            _append_jsonl(
                rank_failures,
                {
                    "selection_index": selection.selection_index,
                    "q_id": selection.q_id,
                    "manifest_query_id": selection.manifest_query_id,
                    "label": label,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": time.monotonic() - sample_started,
                },
            )
            failed += 1
            torch.cuda.empty_cache()

    _write_json(
        rank_dir / f"rank_{rank}_summary.json",
        {
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "succeeded": succeeded,
            "failed": failed,
            "elapsed_seconds": time.monotonic() - started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        },
    )
    _barrier(world_size)

    total_failed = torch.tensor([failed], dtype=torch.int64, device=device)
    if world_size > 1:
        dist.all_reduce(total_failed, op=dist.ReduceOp.SUM)
    if rank == 0:
        prediction_rows = [
            row
            for shard in (rank_dir / f"rank_{index}.jsonl" for index in range(world_size))
            for row in _read_jsonl(shard)
        ]
        failure_rows = [
            row
            for shard in (
                rank_dir / f"rank_{index}_failed.jsonl" for index in range(world_size)
            )
            for row in _read_jsonl(shard)
        ]
        combined = prediction_rows + failure_rows
        if len(combined) != len(selections):
            raise ValueError(
                f"distributed A2-static output is incomplete: "
                f"{len(combined)} != {len(selections)}"
            )
        indices = [int(row["selection_index"]) for row in combined]
        if len(set(indices)) != len(indices) or set(indices) != set(range(len(selections))):
            raise ValueError("distributed A2-static output indices are duplicated or missing")
        (output_dir / "failed.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                for row in sorted(failure_rows, key=lambda item: int(item["selection_index"]))
            ),
            encoding="utf-8",
        )
        (output_dir / "succeeded.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                for row in sorted(
                    prediction_rows, key=lambda item: int(item["selection_index"])
                )
            ),
            encoding="utf-8",
        )
        generated_by_index = {
            int(row["selection_index"]): row for row in prediction_rows
        }
        if not failure_rows or args.allow_failures:
            generated_path = output_dir / "generated_predictions.jsonl"
            with generated_path.open("w", encoding="utf-8") as stream:
                for selection in selections:
                    generated_row = generated_by_index.get(selection.selection_index)
                    prediction = "" if generated_row is None else str(generated_row["predict"])
                    stream.write(
                        json.dumps(
                            {"predict": prediction, "label": selection.label},
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
        _write_json(
            output_dir / "generation_summary.json",
            {
                "status": "completed" if not failure_rows else "completed_with_failures",
                "mode": "a2_static",
                "evaluation_scope": "train_set",
                "expected_rows": len(selections),
                "succeeded": len(prediction_rows),
                "failed": len(failure_rows),
                "world_size": world_size,
                "inner_sgd_attempted": 0,
                "inner_sgd_updated": 0,
                "transient_fast_state": False,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    _barrier(world_size)
    exit_code = 0 if args.allow_failures or int(total_failed.item()) == 0 else 1
    if world_size > 1:
        dist.destroy_process_group()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
