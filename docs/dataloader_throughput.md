# A2/A5 loader throughput

The production runtime keeps one record/episode per rank and moves only CPU work into DataLoader
workers.  Qwen, State Bank, FSM, Fast-TTT, Meta-TTT, and all backward/optimizer boundaries remain
in the training process.
The production path requires the current Qwen3-VL processor interface (`video_processor`, native
tokenizer, and chat template); older processor/collator compatibility paths are intentionally not
supported.

## Full-prefix A2 path

The formal 256-frame A2 profile keeps `query_cache_mode: disabled`. Query target times still
come from the LLaMA-Factory uniform sampler, but `grouped_seek` opens the container once and uses
at most 16 forward decode groups; non-seekable media falls back to one sequential scan. Workers
release raw RGB after Qwen processing and return a compact payload containing timestamps, tubelet
metadata, patches/grid, labels, and preparation telemetry. A2 consumes the GA=4 sequence lazily,
so each microbatch is fetched immediately before its forward/backward; the optimizer boundary is
unchanged. A5 continues to use the upstream Trainer path.

## Launch settings

The H200 A2/A5 profiles use two persistent workers, prefetch factor two, pinned memory, and
`ttt_qwen.support_prefetch_depth: 2`.  Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before
launching multi-GPU jobs.  To enable the cross-epoch cache, point
`TTT_PREPROCESS_CACHE_ROOT` at a shared or local filesystem with roughly 200 GB available.

## What is cached

`ttt_svcbench_qwen.preprocess_cache.PreprocessCache` stores only decoded/resized RGB frames,
timestamps, Qwen patch tensors, grid metadata, and tubelet audit tensors.  Labels, answers,
State Bank/FSM values, Fast-TTT state, and model outputs are never written.  A safetensors file
and JSON fingerprint sidecar are published with `os.replace`; a mismatched media stat or
processor fingerprint is a miss.

Use the helper before/after a run:

```powershell
python scripts/preprocess_cache.py inspect --root $env:TTT_PREPROCESS_CACHE_ROOT --namespace $env:TTT_CACHE_NAMESPACE --max-gb 200
python scripts/preprocess_cache.py verify --root $env:TTT_PREPROCESS_CACHE_ROOT --namespace $env:TTT_CACHE_NAMESPACE --max-gb 200
python scripts/summarize_dataloader_trace.py $env:RUNTIME_TRACE_DIR
```

Set `ttt_qwen.runtime_trace_mode: cuda` and `ttt_qwen.runtime_trace_dir` in a benchmark profile
to emit buffered per-rank/process JSONL events for query preparation, processor, cache hit/miss,
Support decode, pin-memory/H2D, ViT/prefill CUDA events, and instant-equal loss composition.
Formal training keeps tracing `off`; resolving CUDA events happens only when the run flushes.

## Cost sidecar

`scripts/build_visual_cost_index.py` writes strict schema-3 metadata only; it never stores Query
frames, patches, features, or visual-token tensors. Formal full-prefix A2 requires a runtime-trace
derived index through `VISUAL_COST_INDEX`; schema-2, missing, estimated-only, or fingerprint-mismatched
records fail closed. For an explicit calibration smoke run, set `TTT_VISUAL_COST_PREFLIGHT=1` and
`TTT_SMOKE_MAX_STEPS`; then rebuild the index from the emitted trace before formal training.
Task/support buckets, rank alignment, and epoch-boundary rank-0 EMA broadcast remain unchanged.

Use `scripts/select_dataloader_profile.py` to choose between 2 workers/prefetch 2 and 4/1 from
measured trials. Use `scripts/select_visual_batch_size.py` for Support visual batches 1, 2, 4, 8;
it enforces no OOM, at least 12 GiB free memory, state/loss parity, and at least 5% adjacent P50
improvement. The checked-in batch remains 1 until a four-H200 run selects a larger value. The GPU
utilization, GA-wait, step-time, and memory targets are therefore hardware acceptance gates, not
claims established by local tests.
