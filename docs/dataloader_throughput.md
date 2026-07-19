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
python scripts/preprocess_cache.py --root $env:TTT_PREPROCESS_CACHE_ROOT --max-gb 200 --prune
python scripts/summarize_dataloader_trace.py $env:RUN_ROOT/samples/rank_0/dataloader.jsonl
```

Set `TTT_DATALOADER_TRACE=1` to emit per-rank JSONL events for query preparation, processor,
cache hit/miss, Support decode, pin-memory/H2D, GPU step, and DataLoader delivery.
The H200 tmux launcher forwards `TTT_PREPROCESS_CACHE_ROOT`, `TTT_DATALOADER_TRACE`, and
`TTT_A2_SUPPORT_PREFETCH` into the training process, so these switches also work when launching
through `scripts/h200/train_a2_a5.sh`.

## Cost sidecar

`scripts/build_visual_cost_index.py` writes an advisory `visual_cost_index.json`.  Set
`ttt_qwen.visual_cost_index` to that file to let the A2 sampler use its exact/estimated visual
cost before falling back to the existing deterministic header proxy.  Task/support buckets,
seed, rank alignment, and episode ownership are unchanged.
