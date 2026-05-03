from __future__ import annotations

import logging
from typing import Iterable, Optional, Dict, Any, Literal

import torch
from torch.optim import Optimizer
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad

logger = logging.getLogger(__name__)


OptimizerType = Literal[
    "adam",
    "adamw",
    "sgd",
    "rmsprop",
    "adagrad",
]


# =========================================================
# PARAM GROUPING (WEIGHT DECAY SPLIT)
# =========================================================

def build_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 0.01,
    no_decay_keywords: tuple[str, ...] = (
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
        "norm.weight",
    ),
) -> list[dict[str, Any]]:
    """
    Splits parameters into decay / no-decay groups.

    Returns:
        [
            {"params": [...], "weight_decay": wd},
            {"params": [...], "weight_decay": 0.0},
        ]

    A6.4 — also excludes anything the model marks as a *calibration*
    parameter (e.g. the post-hoc temperature scalar). Calibration
    parameters are fit on a held-out validation set by a dedicated
    optimiser via ``BaseModel.get_calibration_parameters``; folding
    them into the main training optimiser would (a) drift them every
    train step and (b) double-update them whenever the calibration
    pass also runs. The check is opt-in via duck-typing — non-BaseModel
    callers (plain ``nn.Module``) get the same decay split as before.
    """

    decay_params = []
    no_decay_params = []

    is_calib = getattr(model, "_is_calibration_parameter_name", None)

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        # A6.4: the calibration optimiser owns these — skip them here.
        if callable(is_calib) and is_calib(name):
            continue

        if any(nd in name for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# =========================================================
# OPTIMIZER FACTORY
# =========================================================

def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: OptimizerType = "adamw",
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    use_param_groups: bool = True,
    custom_params: Optional[Iterable[torch.nn.Parameter]] = None,
) -> Optimizer:
    """
    Create optimizer with best practices.

    Supports:
        - weight decay exclusion
        - custom parameter groups
        - multiple optimizers
    """

    if custom_params is not None:
        params = list(custom_params)

    elif use_param_groups:
        params = build_parameter_groups(
            model=model,
            weight_decay=weight_decay,
        )

    else:
        params = model.parameters()

    optimizer_type = optimizer_type.lower()

    logger.info(f"[OPTIMIZER] Using {optimizer_type}")

    # C1.4: Every optimiser branch must honour ``weight_decay``. When
    # ``use_param_groups=True`` the per-group ``weight_decay`` already
    # comes from ``build_parameter_groups`` and the optimiser-level
    # default is harmless. But when ``use_param_groups=False`` (or
    # ``custom_params`` is provided), only the optimiser-level
    # ``weight_decay`` argument applies — and previously AdamW / Adam
    # / RMSprop / Adagrad all silently fell back to ``0`` instead of
    # the value the caller asked for. That is a correctness bug:
    # users believing they trained with WD=0.01 actually got WD=0.
    if optimizer_type == "adamw":
        return AdamW(
            params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    elif optimizer_type == "adam":
        return Adam(
            params,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    elif optimizer_type == "sgd":
        return SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optimizer_type == "rmsprop":
        return RMSprop(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif optimizer_type == "adagrad":
        return Adagrad(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    optimizer_type: OptimizerType = "adamw",
    **kwargs: Any,
) -> Optimizer:

    return create_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        learning_rate=lr,
        weight_decay=weight_decay,
        **kwargs,
    )