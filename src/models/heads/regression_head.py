from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RegressionHeadConfig:
    input_dim: int
    output_dim: int = 1
    hidden_dim: Optional[int] = None
    dropout: float = 0.1
    activation: str = "gelu"
    use_layernorm: bool = False
    return_features: bool = False
    loss_type: str = "mse"  # mse | mae | huber


class RegressionHead(nn.Module):

    SUPPORTED_ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
    }

    SUPPORTED_LOSSES = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": nn.SmoothL1Loss,
    }

    def __init__(self, config: RegressionHeadConfig) -> None:
        super().__init__()

        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if config.output_dim <= 0:
            raise ValueError("output_dim must be positive")

        if not (0.0 <= config.dropout <= 1.0):
            raise ValueError("dropout must be between 0 and 1")

        if config.activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {config.activation}")

        if config.loss_type not in self.SUPPORTED_LOSSES:
            raise ValueError(f"Unsupported loss: {config.loss_type}")

        self.config = config
        self.has_hidden_layer = bool(config.hidden_dim)

        activation_cls = self.SUPPORTED_ACTIVATIONS[config.activation]

        if config.use_layernorm:
            self.norm = nn.LayerNorm(config.input_dim)
        else:
            self.norm = None

        if self.has_hidden_layer:

            if config.hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")

            self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
            self.activation = activation_cls()

            if config.use_layernorm:
                self.norm_hidden = nn.LayerNorm(config.hidden_dim)
            else:
                self.norm_hidden = None

            self.dropout = nn.Dropout(config.dropout)
            self.fc2 = nn.Linear(config.hidden_dim, config.output_dim)

        else:

            self.dropout = nn.Dropout(config.dropout)
            self.fc = nn.Linear(config.input_dim, config.output_dim)

        self.loss_fn = self.SUPPORTED_LOSSES[config.loss_type]()

        self._init_weights()

        logger.info(
            "RegressionHead initialized | input_dim=%d | output_dim=%d",
            config.input_dim,
            config.output_dim,
        )

    # =====================================================
    # INIT
    # =====================================================

    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        if features is None:
            raise ValueError("features cannot be None")

        if features.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {features.shape}")

        if features.size(1) != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, got {features.size(1)}"
            )

        if not features.is_contiguous():
            features = features.contiguous()

        x = features

        if self.norm is not None:
            x = self.norm(x)

        if self.has_hidden_layer:

            x = self.fc1(x)
            x = self.activation(x)

            if self.norm_hidden is not None:
                x = self.norm_hidden(x)

            x = self.dropout(x)
            outputs = self.fc2(x)

        else:

            x = self.dropout(x)
            outputs = self.fc(x)

        # stats
        mean = outputs.mean(dim=-1)
        variance = outputs.var(dim=-1, unbiased=False)

        result: Dict[str, Any] = {
            "outputs": outputs,
            "mean": mean,
            "variance": variance,
        }

        if targets is not None:

            if targets.shape != outputs.shape:
                raise ValueError(
                    f"targets shape must match outputs. "
                    f"Expected {outputs.shape}, got {targets.shape}"
                )

            loss = self.loss_fn(outputs, targets)
            result["loss"] = loss

        if self.config.return_features:
            result["features"] = x

        return result

    # =====================================================
    # UTILS
    # =====================================================

    def get_output_dim(self) -> int:
        return self.config.output_dim