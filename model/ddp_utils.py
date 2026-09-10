"""Shared utilities for distributed (multi-GPU, DDP) training.

Launch a training script with torchrun::

    torchrun --nproc_per_node=<N> scripts/<train_x>.py <args>

When not launched through torchrun (no ``RANK`` / ``WORLD_SIZE`` /
``LOCAL_RANK`` env vars), every helper degrades to single-process /
single-device behaviour, so existing single-GPU invocations keep working
unchanged.
"""

from __future__ import annotations

import os

import torch


def is_distributed() -> bool:
    return all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main() -> bool:
    return rank() == 0


def setup(backend: str = "nccl") -> None:
    """Initialise the process group (no-op when not launched via torchrun)."""
    if not is_distributed():
        return
    torch.cuda.set_device(local_rank())
    torch.distributed.init_process_group(backend=backend)
    torch.distributed.barrier()


def cleanup() -> None:
    if is_distributed() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def device() -> torch.device:
    if torch.cuda.is_available():
        if is_distributed():
            return torch.device(f"cuda:{local_rank()}")
        return torch.device("cuda")
    return torch.device("cpu")


def wrap(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap the trainable model in DistributedDataParallel (no-op in single mode)."""
    if not is_distributed():
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank()],
        output_device=local_rank(),
        find_unused_parameters=True,
    )


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module (no-op when not wrapped in DDP)."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def make_sampler(dataset, shuffle: bool):
    """DistributedSampler for DDP, or ``None`` in single-process mode."""
    if not is_distributed():
        return None
    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(dataset, num_replicas=world_size(), rank=rank(), shuffle=shuffle)


def set_epoch(sampler, epoch: int) -> None:
    if sampler is not None:
        sampler.set_epoch(epoch)


def all_reduce_mean(value: float) -> float:
    """Average a scalar across all ranks (no-op when not distributed)."""
    if not is_distributed():
        return value
    tensor = torch.tensor([value], device=torch.cuda.current_device(), dtype=torch.float64)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item() / world_size()
