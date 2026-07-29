from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

import ttt_svcbench_qwen.llamafactory_trainer as trainer_module


class _DistributedWarmupToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.LayerNorm(256),
        )
        self.fast_slow = nn.Linear(256, 256, bias=False)
        self.register_buffer("state_counter", torch.ones(4))


def _warmup_finalization_worker(
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
        initial_qwen_sha256 = trainer_module._module_bitwise_sha256(model.qwen)
        with torch.no_grad():
            model.fast_slow.weight.add_(1.0e-3)
            model.state_counter.add_(1.0)
        final_qwen_sha256, prepared = (
            trainer_module._prepare_distributed_warmup_handoff(
                model=model,
                qwen_model=model.qwen,
                initial_qwen_sha256=initial_qwen_sha256,
                device=device,
            )
        )
        assert final_qwen_sha256 == initial_qwen_sha256
        torch.distributed.barrier(device_ids=[rank])
        if rank == 0:
            allowlist, tensors = prepared
            assert tuple(sorted(tensors)) == allowlist
            save_file(tensors, output_path)
        torch.distributed.barrier(device_ids=[rank])
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="the distributed finalization regression requires four CUDA devices",
)
def test_four_gpu_warmup_finalization_keeps_collective_order(
    tmp_path: Path,
) -> None:
    world_size = 4
    init_path = tmp_path / "nccl-init"
    output_path = tmp_path / "warmup.safetensors"

    torch.multiprocessing.spawn(
        _warmup_finalization_worker,
        args=(world_size, str(init_path), str(output_path)),
        nprocs=world_size,
        join=True,
    )

    state = load_file(output_path)
    assert state
    assert all(not name.startswith("qwen.") for name in state)
    assert "fast_slow.weight" in state
    assert "state_counter" in state
