#!/usr/bin/env python3
"""Four-rank NCCL smoke matching the ZeRO-2 bucket that stalled in A2."""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=185_689_936)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.timeout_seconds),
        device_id=torch.device("cuda", local_rank),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError(f"smoke requires world_size=4, got {world_size}")

    large = torch.full(
        (args.numel,),
        1.0 / world_size,
        dtype=torch.bfloat16,
        device=local_rank,
    )
    scalar = torch.tensor([1.0 / world_size], dtype=torch.float32, device=local_rank)
    dist.barrier(device_ids=[local_rank])
    for iteration in range(1, args.iterations + 1):
        started = time.monotonic()
        dist.all_reduce(large)
        dist.all_reduce(scalar)
        torch.cuda.synchronize(local_rank)
        elapsed = time.monotonic() - started
        if not torch.allclose(large[0].float(), torch.tensor(1.0, device=local_rank)):
            raise RuntimeError("large all-reduce produced the wrong value")
        if not torch.allclose(scalar, torch.ones_like(scalar)):
            raise RuntimeError("scalar all-reduce produced the wrong value")
        large.fill_(1.0 / world_size)
        scalar.fill_(1.0 / world_size)
        if rank == 0:
            print(
                f"iteration={iteration}/{args.iterations} numel={args.numel} "
                f"elapsed_seconds={elapsed:.3f}",
                flush=True,
            )
    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        print("status=completed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
