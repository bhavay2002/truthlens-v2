from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class BaseClassifier(BaseModel):

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        dropout: float = 0.1,
        multi_label: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")

        if not (0.0 <= dropout <= 1.0):
            raise ValueError("dropout must be in [0,1]")

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.multi_label = multi_label
        self.label_smoothing = label_smoothing

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)

        if multi_label:
            self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss(
                label_smoothing=label_smoothing
            )

    # =====================================================
    # ENCODE
    # =====================================================

    @abstractmethod
    def encode(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        *inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:

        features = self.encode(*inputs, **kwargs)

        if features.dim() != 2:
            raise ValueError(f"Expected 2D features, got {features.shape}")

        if features.size(1) != self.input_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.input_dim}, got {features.size(1)}"
            )

        features = self.dropout(features)
        logits = self.classifier(features)

        # -------------------------------------------------
        # PROBS / DERIVED STATISTICS
        # -------------------------------------------------
        # During training the loss runs against ``logits`` directly; the
        # derived stats below would only inflate the autograd graph
        # without ever being read. Compute them only at inference (P1).

        output: Dict[str, torch.Tensor] = {"logits": logits}

        if not self.training:
            # N1: stable entropy via ``log_softmax`` / ``logsigmoid`` so
            # we never take ``log(prob + 1e-12)``. The additive EPS
            # dominates the log term whenever the distribution is peaked
            # and biases the entropy toward a fixed lower bound.
            if self.multi_label:
                log_p = F.logsigmoid(logits)
                log_1mp = F.logsigmoid(-logits)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                confidence = probs.max(dim=-1).values
                entropy = -(
                    probs * log_p + (1.0 - probs) * log_1mp
                ).mean(dim=-1)
            else:
                # A6.2: derive ``preds`` and ``confidence`` directly
                # from ``logits`` / ``log_probs``. ``argmax(logits) ==
                # argmax(probs)`` (softmax is monotone) and the maxed
                # log-prob exponentiates to the same value as
                # ``probs.max()`` — but skipping the second softmax /
                # max pass avoids a redundant kernel launch on every
                # eval batch.
                log_probs = F.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                preds = torch.argmax(logits, dim=-1)
                confidence = log_probs.max(dim=-1).values.exp()
                entropy = -(probs * log_probs).sum(dim=-1)

            output["probabilities"] = probs
            output["predictions"] = preds
            output["confidence"] = confidence
            output["entropy"] = entropy

        # -------------------------------------------------
        # LOSS
        # -------------------------------------------------

        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output["loss"] = loss

        if return_features:
            output["features"] = features

        return output

    # =====================================================
    # LOSS
    # =====================================================

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        if self.multi_label:
            labels = labels.float()
            return self.loss_fn(logits, labels)

        # A4.3: only cast to ``long`` when the label tensor is integer-
        # typed. ``CrossEntropyLoss`` happily accepts a float tensor
        # whose last dim equals ``num_classes`` and treats it as a
        # smoothed-soft target distribution (the upstream MixUp
        # pipeline relies on this). Unconditionally calling ``.long()``
        # on a float-soft-label tensor truncates the probabilities to
        # the integer class index 0 / 1 / 2 …, silently destroying the
        # smoothed signal and biasing training toward whichever class
        # the rounding happened to land on.
        if torch.is_floating_point(labels):
            return self.loss_fn(logits, labels)

        return self.loss_fn(logits, labels.long())

    # =====================================================
    # PREDICT
    # =====================================================

    @torch.inference_mode()
    def predict(
        self,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(*inputs, **kwargs)
        finally:
            if was_training:
                self.train()

        return {
            "predictions": outputs["predictions"],
            "probabilities": outputs["probabilities"],
            "confidence": outputs["confidence"],
        }

    # =====================================================
    # LOGITS
    # =====================================================

    @torch.inference_mode()
    def predict_logits(
        self,
        *inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(*inputs, **kwargs)
        finally:
            if was_training:
                self.train()

        return outputs["logits"]

    # =====================================================
    # CONFIG
    # =====================================================

    def get_config(self) -> Dict[str, Any]:

        return {
            "num_classes": self.num_classes,
            "input_dim": self.input_dim,
            "multi_label": self.multi_label,
            "label_smoothing": self.label_smoothing,
        }