#src\models\training\distributed_engine.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class DistributedConfig:
    backend: str = "nccl"  # "nccl" | "gloo"
    init_method: str = "env://"
    use_ddp: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True


# =========================================================
# DISTRIBUTED ENGINE
# =========================================================

class DistributedEngine:
    """
    Distributed training engine (DDP-ready).

    Responsibilities:
    - initialize process group
    - wrap model with DDP
    - manage rank/world_size
    - provide distributed utilities
    """

    def __init__(self, config: Optional[DistributedConfig] = None):

        self.config = config or DistributedConfig()

        self.initialized = False
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0

    # =====================================================
    # INIT
    # =====================================================

    def initialize(self):

        if dist.is_available() and not dist.is_initialized():

            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

            logger.info(
                "Initializing distributed | rank=%d | world_size=%d",
                self.rank,
                self.world_size,
            )

            # GPU-3: ``backend="nccl"`` REQUIRES a CUDA build of PyTorch
            # AND visible CUDA devices. The original code hard-coded
            # ``backend=self.config.backend`` (defaulting to "nccl") and
            # then unconditionally called ``torch.cuda.set_device(...)``
            # — both of which raise on any CPU-only host (CI workers,
            # Replit Reserved-VM trial tier, debugging a multi-process
            # run on a laptop, ...) with cryptic errors like
            # "Distributed package doesn't have NCCL built in" or
            # "Torch not compiled with CUDA enabled". Auto-fall back to
            # the ``gloo`` backend (which works on CPU) when CUDA isn't
            # available, and gate ``set_device`` on the same probe.
            backend = self.config.backend
            if backend == "nccl" and not torch.cuda.is_available():
                logger.warning(
                    "GPU-3: NCCL backend requested but CUDA is not "
                    "available — falling back to 'gloo' so single-host "
                    "/ CPU-only runs don't crash at init_process_group."
                )
                backend = "gloo"

            dist.init_process_group(
                backend=backend,
                init_method=self.config.init_method,
                rank=self.rank,
                world_size=self.world_size,
            )

            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)

            self.initialized = True

        else:
            logger.info("Distributed not initialized (single process)")

    # =====================================================
    # MODEL WRAP
    # =====================================================

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:

        if not self.initialized or not self.config.use_ddp:
            return model

        device = torch.device(f"cuda:{self.local_rank}")

        # N-CRIT-3: Previously this called ``model.to(device)`` here — making
        # this the THIRD redundant move (Trainer.__init__ and TrainingStep
        # both also moved the model historically; GPU-1 already removed those
        # two and asserts device match instead). Worse, this move runs AFTER
        # ``create_trainer_fn`` has built the optimizer over the model
        # parameters, so it leaves the optimizer holding parameter refs
        # whose ``.device`` differs from where DDP now expects them — the
        # classic "expected all tensors to be on the same device" failure
        # at the first ``optimizer.step()``. Validate device match and
        # surface a loud error rather than silently re-moving.
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = device

        if model_device != device:
            raise RuntimeError(
                f"N-CRIT-3: DistributedEngine.wrap_model received model on "
                f"{model_device} but DDP requires it on {device}. Move the "
                f"model to its final CUDA device BEFORE building the optimizer "
                f"in create_trainer_fn (the optimizer captures parameter "
                f"references and a post-hoc move silently invalidates them)."
            )

        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.config.find_unused_parameters,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
        )

        logger.info("Model wrapped with DDP")

        return model

    # =====================================================
    # SAMPLER
    # =====================================================

    def create_sampler(self, dataset, shuffle: bool = True):

        if not self.initialized:
            return None

        return torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
        )

    # =====================================================
    # SYNC UTILS
    # =====================================================

    def barrier(self):
        if self.initialized:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:

        if not self.initialized:
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor = tensor / self.world_size

        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0):

        if self.initialized:
            dist.broadcast(tensor, src)

    # =====================================================
    # HELPERS
    # =====================================================

    def is_main_process(self) -> bool:
        return self.rank == 0

    def cleanup(self):

        if self.initialized:
            dist.destroy_process_group()
            logger.info("Distributed process group destroyed")