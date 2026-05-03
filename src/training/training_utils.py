# src/training/training_utils.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# DEVICE
# =========================================================

def get_device(device: Optional[str] = None) -> torch.device:

    if device:
        return torch.device(device)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(
    batch: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:

    if isinstance(batch, torch.Tensor):
        use_nb = (
            non_blocking
            and batch.device.type == "cpu"
            and device.type == "cuda"
            and batch.is_pinned()
        )
        return batch.to(device, non_blocking=use_nb)

    if isinstance(batch, dict):
        return {
            k: move_batch_to_device(v, device, non_blocking)
            for k, v in batch.items()
        }

    if isinstance(batch, (list, tuple)):
        return type(batch)(
            move_batch_to_device(v, device, non_blocking)
            for v in batch
        )

    return batch


# =========================================================
# GRADIENT MONITORING
# =========================================================

def compute_grad_norm(model: nn.Module) -> float:
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is None:
            continue

        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


# =========================================================
# LR UTILITIES
# =========================================================

def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    for group in optimizer.param_groups:
        return float(group["lr"])
    return 0.0


# =========================================================
# THROUGHPUT
# =========================================================

def compute_throughput(
    batch_size: int,
    duration: float,
) -> float:
    if duration <= 0:
        return 0.0
    return batch_size / duration


# =========================================================
# METRICS CONTAINER
# =========================================================

@dataclass
class TrainingMetrics:

    task_losses: Dict[str, float] = field(default_factory=dict)
    losses: Dict[str, float] = field(default_factory=dict)

    step: int = 0
    epoch: int = 0

    grad_norm: float = 0.0
    lr: float = 0.0
    throughput: float = 0.0

    def update(self, name: str, value: float) -> None:
        self.losses[name] = float(value)

    def update_task(self, task: str, loss: float) -> None:
        self.task_losses[task] = float(loss)

    def set_grad_norm(self, value: float) -> None:
        self.grad_norm = float(value)

    def set_lr(self, value: float) -> None:
        self.lr = float(value)

    def set_throughput(self, value: float) -> None:
        self.throughput = float(value)

    def average_loss(self) -> float:
        if not self.task_losses:
            return 0.0
        return sum(self.task_losses.values()) / len(self.task_losses)

    def to_dict(self) -> Dict[str, float]:
        return {
            **self.losses,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "throughput": self.throughput,
        }
