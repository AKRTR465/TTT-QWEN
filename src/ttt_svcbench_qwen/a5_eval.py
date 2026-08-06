"""Distributed A5 slot-memory generation over the fixed SVCBench train-set selection.

The counterpart to :mod:`ttt_svcbench_qwen.a2_static_eval`.  That module deliberately
forbids the memory path (``_assert_no_fast_state``, ``fast_state_row_count == 0``), so it
cannot measure the thing A5 trains.  This module runs the identical episode data path --
manifest -> collator -> materializer -> the same 3,706-row selection -- and adds exactly one
thing: a zero-initialized per-video memory that each Support chunk writes into, so the Answer
Query reads ``M_T``.

Keeping the episode path identical is deliberate: it makes the A5 number directly comparable
with the A2-static number on the same rows.  The write itself reuses the production online
path's mechanics (see ``inference.OnlineTTTUpdater``): both of its inputs hang off
``observation.soft_intermediates``, which ``model.observe_chunk`` already returns here, so no
part of the inference request machinery is needed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

# Reused verbatim from the A2-static evaluator rather than duplicated: the selection sharding,
# jsonl IO, prompt-boundary derivation and support batching must stay byte-identical for the two
# numbers to be comparable, and the external scorer consumes the same output format.
from ttt_svcbench_qwen.a2_static_eval import (
    _append_jsonl,
    _barrier,
    _initialize_distributed,
    _prepare_support_batch,
    _read_jsonl,
    _selection_label,
    _write_json,
    prompt_only_answer_inputs,
)
from ttt_svcbench_qwen.episode_data import (
    A2QueryRecord,
    ManifestStage,
    load_production_manifest_views,
)
from ttt_svcbench_qwen.fast_ttt import FastMemoryState, FastTTTAdapter
from ttt_svcbench_qwen.llamafactory_trainer import (
    ProductionTrainerRuntime,
    _load_warmup_bundle,
)
from ttt_svcbench_qwen.memory_write import apply_memory_writes
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
from ttt_svcbench_qwen.svcbench_train_eval import read_selection, read_sft_rows
from ttt_svcbench_qwen.trainer import (
    StageAEpisodeInputs,
    StageATrainingBatch,
    answer_query_request,
    prepared_query_pair,
)


@dataclass(frozen=True, slots=True)
class A5MemoryGenerationAudit:
    """Per-episode evidence that the memory was actually exercised.

    The invariants here are the *inverse* of ``A2StaticGenerationAudit``'s on purpose.  A run
    whose writes silently never happened would produce exactly the A2-static answers while
    being reported as A5, which is the one failure mode that cannot be spotted from the
    accuracy number alone.  So a zero-write episode is rejected rather than recorded.
    """

    mode: str
    support_count: int
    observed_chunk_count: int
    memory_writes_attempted: int
    memory_writes_applied: int
    memory_writes_skipped: int
    fast_state_row_count: int
    final_write_version: int
    final_memory_norm: float
    reader_status: str
    reader_exact_count: int | None
    reader_selected_record_count: int
    lifecycle_observation_count: int
    lifecycle_prefill_count: int
    lifecycle_decode_count: int

    def __post_init__(self) -> None:
        if self.mode != "a5_memory":
            raise ValueError("A5 memory generation audit has the wrong mode")
        if self.fast_state_row_count != 1:
            raise ValueError("A5 evaluation requires exactly one per-video memory row")
        if self.support_count <= 0:
            raise ValueError("A5 evaluation requires at least one Support chunk")
        if self.memory_writes_attempted != self.support_count:
            raise ValueError("A5 must attempt exactly one write per Support chunk")
        if self.memory_writes_applied <= 0:
            raise ValueError("A5 evaluation cannot report an episode with no applied write")
        if self.memory_writes_applied + self.memory_writes_skipped != self.support_count:
            raise ValueError("A5 write accounting does not cover every Support chunk")
        if self.final_write_version != self.memory_writes_applied:
            raise ValueError("A5 final write version disagrees with the applied write count")
        if not self.final_memory_norm > 0.0:
            raise ValueError("A5 evaluation cannot report an all-zero final memory")


@torch.inference_mode()  # type: ignore[untyped-decorator]
def generate_a5_memory(
    batch: StageATrainingBatch,
    *,
    model: StateTTTModel,
    fast_adapter: FastTTTAdapter,
    query_encoder_reuse: bool,
    max_new_tokens: int,
) -> tuple[StateTTTGenerationOutput, A5MemoryGenerationAudit]:
    """Observe every chunk causally, write each Support chunk into ``M``, then answer."""

    episode = batch.model_inputs
    if not isinstance(episode, StageAEpisodeInputs):
        raise TypeError("A5 memory evaluation requires StageAEpisodeInputs")
    initial = episode.observation_requests[0].runtime_state
    if not isinstance(initial, BatchRuntimeState) or initial.next_chunk_index != 0:
        raise ValueError("A5 episode must begin from reset runtime state")
    if len(initial.rows) != 1:
        raise ValueError("A5 evaluation requires one trajectory row per rank")

    # Zero per video: the memory carries no cross-video state by contract.
    fast = fast_adapter.initialize_fast_state(differentiable=False)
    if fast.write_version != 0 or fast.write_count != 0:
        raise ValueError("A5 episode must start from a pristine zero memory")
    runtime = initial.with_fast_states((fast,))
    bank_states = initial.state_bank_states

    lifecycle = PrefillLifecycle(episode.owner)
    final_observation: ObservationChunkOutput | None = None
    prepared_query: PreparedQueryOutput | None = None
    detached_query: PreparedQueryOutput | None = None
    if query_encoder_reuse:
        final_template = episode.observation_requests[-1]
        prepared_query, detached_query = prepared_query_pair(model, final_template, inference=True)

    attempted = applied = skipped = 0
    for chunk_index, template in enumerate(episode.observation_requests):
        # The last request is the State Query, not a Support chunk: it is observed read-only,
        # matching ``PerVideoRuntimeManager.observe_query_readonly``.
        is_state_query = chunk_index + 1 == len(episode.observation_requests)
        observation_request = replace(
            template,
            inference=True,
            runtime_state=runtime,
            bank_states=bank_states,
            prepared_query=(prepared_query if is_state_query else detached_query),
        )
        with fast_adapter.use_fast_state(fast):
            observed = model.observe_chunk(observation_request, lifecycle)
        runtime = observed.runtime_state
        bank_states = observed.bank_states
        if runtime.next_chunk_index != chunk_index + 1:
            raise ValueError("A5 runtime did not advance exactly once per chunk")
        if not is_state_query:
            attempted += 1
            fast, was_applied = _write_chunk(observed, fast, fast_adapter)
            applied += int(was_applied)
            skipped += int(not was_applied)
            runtime = runtime.with_fast_states((fast,))
        final_observation = observed

    if final_observation is None:
        raise RuntimeError("A5 evaluation produced no causal observation")

    answer = prompt_only_answer_inputs(
        episode.answer,
        batch.supervision.answer.base_labels,
    )
    answer_request = answer_query_request(
        episode.owner,
        replace(final_observation, runtime_state=runtime),
        answer,
    )
    with fast_adapter.use_fast_state(fast):
        generated = model.generate_answer(
            model.prepare_answer(answer_request, lifecycle),
            lifecycle,
            max_new_tokens=max_new_tokens,
        )
    lifecycle_audit = generated.lifecycle
    if (
        lifecycle_audit.observation_count != len(episode.observation_requests)
        or lifecycle_audit.prefill_count != 1
        or lifecycle_audit.decode_count != 0
    ):
        raise ValueError("A5 lifecycle counts are inconsistent")
    reader = generated.reader[0] if generated.reader else None
    audit = A5MemoryGenerationAudit(
        mode="a5_memory",
        support_count=len(episode.observation_requests) - 1,
        observed_chunk_count=len(episode.observation_requests),
        memory_writes_attempted=attempted,
        memory_writes_applied=applied,
        memory_writes_skipped=skipped,
        fast_state_row_count=1,
        final_write_version=int(fast.write_version),
        final_memory_norm=float(torch.linalg.matrix_norm(fast.m.detach()).cpu().item()),
        reader_status=(
            "disabled" if reader is None else str(getattr(reader.status, "value", reader.status))
        ),
        reader_exact_count=None if reader is None else reader.exact_count,
        reader_selected_record_count=(0 if reader is None else len(reader.selected_record_ids)),
        lifecycle_observation_count=lifecycle_audit.observation_count,
        lifecycle_prefill_count=lifecycle_audit.prefill_count,
        lifecycle_decode_count=lifecycle_audit.decode_count,
    )
    return generated, audit


def _write_chunk(
    observed: ObservationChunkOutput,
    fast_state: FastMemoryState,
    fast_adapter: FastTTTAdapter,
) -> tuple[FastMemoryState, bool]:
    """Apply one label-free delta-rule write, mirroring ``inference.OnlineTTTUpdater``."""

    intermediates = observed.soft_intermediates.fast_associative
    if intermediates is None:
        raise ValueError("A5 observation did not capture the associative write tensors")
    spatial = observed.soft_intermediates.spatial
    if spatial is None:
        raise ValueError("A5 observation did not produce spatial slot state")
    if fast_state.differentiable:
        raise ValueError("A5 evaluation memory must be a non-differentiable online leaf")
    batch = fast_adapter.prepare_write(intermediates, spatial)
    result = apply_memory_writes(fast_states=(fast_state,), batch=batch)[0]
    return result.fast_state, bool(result.did_write)


def _require_single_adapter(model: nn.Module) -> FastTTTAdapter:
    # The same idiom the trainer uses in five places to locate the one memory adapter.
    adapters = tuple(module for module in model.modules() if isinstance(module, FastTTTAdapter))
    if len(adapters) != 1:
        raise RuntimeError("A5 evaluation requires exactly one FastTTTAdapter")
    return adapters[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warmup-bundle", type=Path, required=True)
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
    # The bundle overlay verifies provenance against the backbone's own ttt_config, so the
    # bundle path has to be bound before _load_warmup_bundle runs.
    backbone = replace(
        backbone,
        ttt_config=backbone.ttt_config.model_copy(
            update={"warmup_bundle": str(args.warmup_bundle.resolve())}
        ),
    )
    runtime_value = build_runtime(backbone, backbone.ttt_config)
    if not isinstance(runtime_value, ProductionTrainerRuntime):
        raise TypeError("runtime factory did not return ProductionTrainerRuntime")
    runtime = runtime_value
    model_owner = runtime.model
    loss_step = runtime.stage_a_loss_step
    if not isinstance(model_owner, nn.Module) or not isinstance(loss_step, ProductionA2LossStep):
        raise TypeError("A5 evaluation requires the production episode runtime")
    load_outer_checkpoint(model_owner, args.checkpoint)
    bundle_audit = _load_warmup_bundle(
        model=model_owner,
        qwen_model=backbone.model,
        backbone=backbone,
        # Adding an evaluator is itself a commit, so the bundle's recorded commit can never equal
        # the evaluating commit.  Every other provenance field stays strict, and the audit records
        # both commits plus the fact that the exemption was used.
        allow_code_drift=True,
    )
    model_owner.to(device=device)
    model_owner.requires_grad_(False)
    model_owner.eval()
    disable_gc = getattr(model_owner, "gradient_checkpointing_disable", None)
    if callable(disable_gc):
        disable_gc()
    state_model = loss_step.runner.model
    if not isinstance(state_model, StateTTTModel):
        raise TypeError("production runner lost its StateTTTModel")
    fast_adapter = _require_single_adapter(model_owner)

    torch.cuda.reset_peak_memory_stats(device)
    rank_predictions = rank_dir / f"rank_{rank}.jsonl"
    rank_failures = rank_dir / f"rank_{rank}_failed.jsonl"
    if rank_predictions.exists() or rank_failures.exists():
        raise FileExistsError(f"rank output already exists for rank {rank}")
    if rank == 0:
        _write_json(output_dir / "warmup_bundle_audit.json", bundle_audit)

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
                raise TypeError("episode collator must return a mapping")
            prepared = collated.get("prepared_a2")
            if not isinstance(prepared, PreparedA2Record):
                raise TypeError("episode collator did not return PreparedA2Record")
            batch = loss_step.materializer.a2(prepared)
            batch = _prepare_support_batch(batch, loss_step)
            generated, audit = generate_a5_memory(
                batch,
                model=state_model,
                fast_adapter=fast_adapter,
                query_encoder_reuse=loss_step.runner.query_encoder_reuse,
                max_new_tokens=args.max_new_tokens,
            )
            _append_jsonl(
                rank_predictions,
                {
                    "selection_index": selection.selection_index,
                    "q_id": selection.q_id,
                    "manifest_query_id": selection.manifest_query_id,
                    "predict": generated.answer_text,
                    "label": label,
                    "elapsed_seconds": time.monotonic() - sample_started,
                    "audit": asdict(audit),
                },
            )
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

    if rank == 0:
        _merge_rank_outputs(
            output_dir=output_dir,
            rank_dir=rank_dir,
            world_size=world_size,
            selections=selections,
            allow_failures=bool(args.allow_failures),
        )
    _barrier(world_size)
    return 0


def _merge_rank_outputs(
    *,
    output_dir: Path,
    rank_dir: Path,
    world_size: int,
    selections: Any,
    allow_failures: bool,
) -> None:
    """Collect the per-rank shards into the scorer's expected files, fail-closed on gaps."""

    prediction_rows = [
        row
        for shard in (rank_dir / f"rank_{index}.jsonl" for index in range(world_size))
        for row in _read_jsonl(shard)
    ]
    failure_rows = [
        row
        for shard in (rank_dir / f"rank_{index}_failed.jsonl" for index in range(world_size))
        for row in _read_jsonl(shard)
    ]
    combined = prediction_rows + failure_rows
    if len(combined) != len(selections):
        raise ValueError(
            f"distributed A5 output is incomplete: {len(combined)} != {len(selections)}"
        )
    indices = [int(row["selection_index"]) for row in combined]
    if len(set(indices)) != len(indices) or set(indices) != set(range(len(selections))):
        raise ValueError("distributed A5 output indices are duplicated or missing")

    def _dump(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                for row in sorted(rows, key=lambda item: int(item["selection_index"]))
            ),
            encoding="utf-8",
        )

    _dump(output_dir / "failed.jsonl", failure_rows)
    _dump(output_dir / "succeeded.jsonl", prediction_rows)
    if failure_rows and not allow_failures:
        raise RuntimeError(
            f"A5 evaluation had {len(failure_rows)} failed rows; "
            "pass --allow-failures to score a partial run"
        )
    generated_by_index = {int(row["selection_index"]): row for row in prediction_rows}
    with (output_dir / "generated_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for selection in selections:
            row = generated_by_index.get(selection.selection_index)
            payload = {
                "predict": "" if row is None else cast(str, row["predict"]),
                "label": _selection_label_of(row, selection),
            }
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _selection_label_of(row: dict[str, Any] | None, selection: Any) -> str:
    if row is not None:
        return cast(str, row["label"])
    raise ValueError(
        f"selection {selection.selection_index} has no prediction to score"
    )


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
