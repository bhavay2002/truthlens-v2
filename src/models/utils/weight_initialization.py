from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# INIT CORE
# =========================================================

def initialize_weights(
    model: nn.Module,
    method: str = "xavier",
    bias_value: float = 0.0,
    gain: Optional[float] = None,
) -> None:

    if not isinstance(model, nn.Module):
        raise TypeError("model must be nn.Module")

    logger.info("Weight init | method=%s", method)

    for module in model.modules():

        # -------------------------------------------------
        # LINEAR / CONV
        # -------------------------------------------------

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):

            if method == "xavier":
                nn.init.xavier_uniform_(module.weight, gain=gain or 1.0)

            elif method == "xavier_normal":
                nn.init.xavier_normal_(module.weight, gain=gain or 1.0)

            elif method == "kaiming":
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="relu",
                )

            elif method == "kaiming_normal":
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

            elif method == "normal":
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif method == "uniform":
                nn.init.uniform_(module.weight, -0.1, 0.1)

            else:
                raise ValueError(f"Unsupported method: {method}")

            if module.bias is not None:
                nn.init.constant_(module.bias, bias_value)

        # -------------------------------------------------
        # EMBEDDING
        # -------------------------------------------------

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # -------------------------------------------------
        # LAYERNORM
        # -------------------------------------------------

        elif isinstance(module, nn.LayerNorm):

            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)


# =========================================================
# RESET
# =========================================================

def reset_module_parameters(module: nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


# =========================================================
# APPLY
# =========================================================

def apply_weight_initialization(
    model: nn.Module,
    method: str = "xavier",
    seed: Optional[int] = None,
) -> None:

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    initialize_weights(model, method=method)

    logger.info("Initialization applied | method=%s | seed=%s", method, seed)