#!/usr/bin/env python3
"""Exercise the A5 warmup handoff ordering on four CUDA/NCCL ranks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from ttt_svcbench_qwen.llamafactory_trainer import (
    _module_bitwise_sha256,
    _prepare_distributed_warmup_handoff,
)


class _DistributedWarmupToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm(256),
        )
        self.fast_slow = nn.Linear(256, 256, bias=False)
        self.register_buffer("state_counter", torch.ones(4))


def _worker(
    rank: int,
    world_size: int,
    init_path: str,
    output_path: str,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        torch.manual_seed(41)
        model = _DistributedWarmupToy().to(device)
        initial_qwen_sha256 = _module_bitwise_sha256(model.qwen)
        with torch.no_grad():
            model.fast_slow.weight.add_(1.0e-3)
            model.state_counter.add_(1.0)
        final_qwen_sha256, prepared = _prepare_distributed_warmup_handoff(
            model=model,
            qwen_model=model.qwen,
            initial_qwen_sha256=initial_qwen_sha256,
            device=device,
        )
        if final_qwen_sha256 != initial_qwen_sha256:
            raise RuntimeError("Qwen digest changed in the finalization smoke")
        torch.distributed.barrier(device_ids=[rank])
        if rank == 0:
            allowlist, tensors = prepared
            if tuple(sorted(tensors)) != allowlist:
                raise RuntimeError("prepared bundle allowlist drifted")
            save_file(tensors, output_path)
        torch.distributed.barrier(device_ids=[rank])
    finally:
        torch.distributed.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if os.environ.get("USER") != "niujunbo":
        raise RuntimeError("H200 finalization smoke must run as niujunbo")
    if args.world_size != 4:
        raise ValueError("the release regression requires exactly four ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.world_size:
        raise RuntimeError("four visible CUDA devices are required")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    init_path = output_dir / "nccl-init"
    weights_path = output_dir / "warmup.safetensors"
    torch.multiprocessing.spawn(
        _worker,
        args=(args.world_size, str(init_path), str(weights_path)),
        nprocs=args.world_size,
        join=True,
    )
    state = load_file(weights_path)
    if not state:
        raise RuntimeError("finalization smoke emitted an empty bundle")
    if any(name.startswith("qwen.") for name in state):
        raise RuntimeError("Qwen tensor entered the finalization smoke bundle")
    required = {"fast_slow.weight", "state_counter"}
    if not required <= set(state):
        raise RuntimeError("finalization smoke bundle is missing persistent tensors")
    summary = {
        "status": "completed",
        "world_size": args.world_size,
        "tensor_count": len(state),
        "qwen_tensor_count": 0,
        "weights": str(weights_path),
    }
    summary_path = output_dir / "run_summary.json"
    temporary = output_dir / ".run_summary.json.incomplete"
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
