"""
File: distributed_utils.py
Location: src/utils/

Distributed training utilities for TruthLens.

Supports:
- PyTorch DDP initialization (torchrun compatible)
- Rank/world size management
- Safe logging control (primary process only)
- Synchronization barriers
- Device assignment (per-rank GPU)

Designed for:
- Multi-GPU training
- Research reproducibility
- Production stability
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# torch / torch.distributed are imported lazily inside functions so that
# importing this module does not pull in PyTorch at startup time.
# Any function that needs torch imports it locally with:
#   import torch, torch.distributed as dist
# This keeps CLI startup and test-script startup fast (~5 s saved).

logger = logging.getLogger(__name__)


def _get_dist():
    import torch.distributed as dist
    return dist


# =========================================================
# STATE CHECKS
# =========================================================

def is_distributed() -> bool:
    dist = _get_dist()
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    dist = _get_dist()
    return dist.get_world_size() if is_distributed() else 1


def get_rank() -> int:
    dist = _get_dist()
    return dist.get_rank() if is_distributed() else 0


def is_primary() -> bool:
    return get_rank() == 0


# =========================================================
# INITIALIZATION
# =========================================================

def init_distributed(
    backend: Optional[str] = None,
    timeout_seconds: int = 1800,
) -> None:
    """
    Initialize distributed training.

    Automatically detects torchrun environment.

    Environment variables used:
    - RANK
    - WORLD_SIZE
    - LOCAL_RANK
    """

    if is_distributed():
        logger.warning("Distributed already initialized")
        return

    if "RANK" not in os.environ:
        logger.info("Running in single-process mode")
        return

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    import torch
    import torch.distributed as dist

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    logger.info(
        "Initializing distributed | rank=%d | world_size=%d | backend=%s",
        rank,
        world_size,
        backend,
    )

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.timedelta(seconds=timeout_seconds),
    )

    # Set device per process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    logger.info("Distributed initialized successfully")


# =========================================================
# CLEANUP
# =========================================================

def cleanup_distributed() -> None:
    if is_distributed():
        _get_dist().destroy_process_group()
        logger.info("Distributed process group destroyed")


# =========================================================
# SYNCHRONIZATION
# =========================================================

def barrier() -> None:
    if is_distributed():
        _get_dist().barrier()


# =========================================================
# SAFE EXECUTION (PRIMARY ONLY)
# =========================================================

def run_on_primary(fn, *args, **kwargs):
    """
    Execute function only on primary process.
    """
    if is_primary():
        return fn(*args, **kwargs)
    return None


# =========================================================
# REDUCTION UTILITIES
# =========================================================

def all_reduce_mean(tensor: "torch.Tensor") -> "torch.Tensor":
    """
    Compute mean across all processes.
    """
    if not is_distributed():
        return tensor

    dist = _get_dist()
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def all_reduce_sum(tensor: "torch.Tensor") -> "torch.Tensor":
    if not is_distributed():
        return tensor

    dist = _get_dist()
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# =========================================================
# GATHER UTILITIES
# =========================================================

def gather_tensor(tensor: "torch.Tensor") -> "list[torch.Tensor]":
    """
    Gather tensors from all processes.
    """
    import torch
    if not is_distributed():
        return [tensor]

    dist = _get_dist()
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]

    dist.all_gather(gathered, tensor)

    return gathered


# =========================================================
# BROADCAST
# =========================================================

def broadcast_object(obj, src: int = 0):
    """
    Broadcast Python object from source process.
    """
    if not is_distributed():
        return obj

    dist = _get_dist()
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


# =========================================================
# DEBUG INFO
# =========================================================

def get_distributed_info() -> dict:
    import torch
    return {
        "distributed": is_distributed(),
        "rank": get_rank(),
        "world_size": get_world_size(),
        "device": torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
    }