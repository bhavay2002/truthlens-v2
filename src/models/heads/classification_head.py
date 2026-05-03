from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


@dataclass
class ClassificationHeadConfig:
    input_dim: int
    num_classes: int
    hidden_dim: Optional[int] = None
    dropout: float = 0.1
    activation: str = "gelu"
    use_layernorm: bool = False
    return_features: bool = False


class ClassificationHead(nn.Module):

    SUPPORTED_ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }

    def __init__(self, config: ClassificationHeadConfig) -> None:
        super().__init__()

        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if config.num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if not (0.0 <= config.dropout <= 1.0):
            raise ValueError("dropout must be between 0 and 1")

        if config.activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {config.activation}")

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
            self.dropout = nn.Dropout(config.dropout)

            if config.use_layernorm:
                self.norm_hidden = nn.LayerNorm(config.hidden_dim)
            else:
                self.norm_hidden = None

            self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)

        else:

            self.dropout = nn.Dropout(config.dropout)
            self.fc = nn.Linear(config.input_dim, config.num_classes)

        self._init_weights()

        logger.info(
            "ClassificationHead initialized | input_dim=%d | num_classes=%d",
            config.input_dim,
            config.num_classes,
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

    def forward(self, features: torch.Tensor) -> Dict[str, Any]:

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

            logits = self.fc2(x)

        else:

            x = self.dropout(x)
            logits = self.fc(x)

        # Derived statistics (probabilities / confidence / entropy) are
        # only meaningful at inference time. Computing them during
        # training wastes compute and pulls extra activations into the
        # autograd graph for tensors the loss never reads. Skip them
        # when ``self.training`` is True (P1).
        output: Dict[str, Any] = {"logits": logits}

        if not self.training:
            # N1: derive probabilities and entropy from a single
            # ``log_softmax`` pass. The previous formulation
            # (``-sum(p * log(p + 1e-12))``) was numerically dominated
            # by the EPS term whenever the predictive distribution was
            # peaked, because each subdominant ``log(eps)`` contributes
            # a fixed bias. ``log_softmax`` avoids the EPS entirely
            # AND amortises the second softmax pass.
            # A6.2: ``confidence`` comes from ``log_probs.max().exp()``
            # — equivalent to ``probs.max()`` but without the second
            # softmax/max pass over the same tensor.
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            confidence = log_probs.max(dim=-1).values.exp()
            entropy = -(probs * log_probs).sum(dim=-1)
            output["probabilities"] = probs
            output["confidence"] = confidence
            output["entropy"] = entropy

        if self.config.return_features:
            output["features"] = x

        return output

    # =====================================================
    # UTILS
    # =====================================================

    def get_output_dim(self) -> int:
        return self.config.num_classes