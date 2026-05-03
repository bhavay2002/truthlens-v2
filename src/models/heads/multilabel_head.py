from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class MultiLabelHeadConfig:
    input_dim: int
    num_labels: int
    hidden_dim: Optional[int] = None
    dropout: float = 0.1
    activation: str = "gelu"
    threshold: float = 0.5
    use_layernorm: bool = False
    return_features: bool = False


class MultiLabelHead(nn.Module):

    SUPPORTED_ACTIVATIONS = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
    }

    def __init__(self, config: MultiLabelHeadConfig) -> None:
        super().__init__()

        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if config.num_labels <= 0:
            raise ValueError("num_labels must be positive")

        if not (0.0 <= config.dropout <= 1.0):
            raise ValueError("dropout must be between 0 and 1")

        if config.activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {config.activation}")

        if not (0 < config.threshold < 1):
            raise ValueError("threshold must be between 0 and 1")

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
            self.fc2 = nn.Linear(config.hidden_dim, config.num_labels)

        else:

            self.dropout = nn.Dropout(config.dropout)
            self.fc = nn.Linear(config.input_dim, config.num_labels)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self._init_weights()

        logger.info(
            "MultiLabelHead initialized | input_dim=%d | num_labels=%d",
            config.input_dim,
            config.num_labels,
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
        labels: Optional[torch.Tensor] = None,
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

            logits = self.fc2(x)

        else:

            x = self.dropout(x)
            logits = self.fc(x)

        # Sigmoid / threshold / per-label entropy are only useful at
        # inference. Computing them inside the autograd graph during
        # training is dead weight (the loss runs against ``logits``
        # directly via ``BCEWithLogitsLoss``). Skip them in training
        # mode (P1).
        outputs: Dict[str, Any] = {"logits": logits}

        if not self.training:
            # N1: per-label binary entropy via ``logsigmoid`` so we never
            # take ``log`` of a value that has been additively biased
            # by an EPS. ``log p   = logsigmoid(x)`` and
            # ``log (1 - p) = logsigmoid(-x)`` are computed in
            # log-space and are exact at the saturation tails where the
            # old ``log(p + 1e-12)`` formulation was dominated by the
            # eps term. ``probs`` itself is materialised via ``sigmoid``
            # for the threshold/predictions output.
            log_p = F.logsigmoid(logits)
            log_1mp = F.logsigmoid(-logits)
            probs = torch.sigmoid(logits)
            predictions = probs >= self.config.threshold

            confidence = probs.mean(dim=-1)
            entropy = -(probs * log_p + (1.0 - probs) * log_1mp).mean(dim=-1)

            outputs["probabilities"] = probs
            outputs["predictions"] = predictions
            outputs["confidence"] = confidence
            outputs["entropy"] = entropy

        if labels is not None:
            if labels.shape != logits.shape:
                raise ValueError(
                    f"labels shape must match logits shape. "
                    f"Expected {logits.shape}, got {labels.shape}"
                )

            loss = self.loss_fn(logits, labels.float())
            outputs["loss"] = loss

        if self.config.return_features:
            outputs["features"] = x

        return outputs

    # =====================================================
    # UTILS
    # =====================================================

    def get_output_dim(self) -> int:
        return self.config.num_labels