from __future__ import annotations

import math
from typing import Optional, Literal

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


SchedulerType = Literal[
    "linear",
    "cosine",
    "cosine_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
]


# =========================================================
# BASE LAMBDA BUILDERS
# =========================================================

def _linear_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 1.0 - progress)

    return lr_lambda


def _cosine_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)),
        )

    return lr_lambda


def _cosine_restarts_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 3,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        if progress >= 1.0:
            return 0.0

        return max(
            0.0,
            0.5
            * (1.0 + math.cos(math.pi * ((num_cycles * progress) % 1.0))),
        )

    return lr_lambda


def _polynomial_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    power: float = 1.0,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, (1.0 - progress) ** power)

    return lr_lambda


def _constant_schedule():
    def lr_lambda(current_step: int):
        return 1.0

    return lr_lambda


def _constant_with_warmup_schedule(num_warmup_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return lr_lambda


# =========================================================
# FACTORY
# =========================================================

def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: SchedulerType,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    num_cycles: float = 0.5,
    power: float = 1.0,
) -> LambdaLR:

    if scheduler_type == "linear":
        lr_lambda = _linear_schedule(
            num_warmup_steps,
            num_training_steps,
        )

    elif scheduler_type == "cosine":
        lr_lambda = _cosine_schedule(
            num_warmup_steps,
            num_training_steps,
            num_cycles=num_cycles,
        )

    elif scheduler_type == "cosine_restarts":
        lr_lambda = _cosine_restarts_schedule(
            num_warmup_steps,
            num_training_steps,
            num_cycles=int(num_cycles),
        )

    elif scheduler_type == "polynomial":
        lr_lambda = _polynomial_schedule(
            num_warmup_steps,
            num_training_steps,
            power=power,
        )

    elif scheduler_type == "constant":
        lr_lambda = _constant_schedule()

    elif scheduler_type == "constant_with_warmup":
        lr_lambda = _constant_with_warmup_schedule(num_warmup_steps)

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: Optimizer,
    config: dict,
) -> LambdaLR:

    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    scheduler_type = config.get("scheduler_type", "linear")
    num_training_steps = int(config.get("num_training_steps", 1000))
    num_warmup_steps = int(config.get("num_warmup_steps", 0))
    num_cycles = float(config.get("num_cycles", 0.5))
    power = float(config.get("power", 1.0))

    return get_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        num_cycles=num_cycles,
        power=power,
    )