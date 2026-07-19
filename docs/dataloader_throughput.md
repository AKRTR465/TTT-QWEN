# A2/A5 loader throughput

The production runtime keeps one record/episode per rank and moves only CPU work into DataLoader
workers.  Qwen, State Bank, FSM, Fast-TTT, Meta-TTT, and all backward/optimizer boundaries remain
in the training process.
The production path requires the current Qwen3-VL processor interface (`video_processor`, native
tokenizer, and chat template); older processor/collator compatibility paths are intentionally not
supported.

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

`scripts/build_visual_cost_index.py` writes an advisory `visual_cost_index.json`.  Set
`ttt_qwen.visual_cost_index` to that file to let the A2 sampler use its exact/estimated visual
cost before falling back to the existing deterministic header proxy.  Task/support buckets,
seed, rank alignment, and episode ownership are unchanged.
