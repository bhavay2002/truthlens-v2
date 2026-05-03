from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...base.base_model import BaseModel
from ...config import HeadConfig, TaskConfig, MultiTaskModelConfig
from ...encoder.encoder_config import EncoderConfig
from ...encoder.encoder_factory import EncoderFactory
from ...heads.classification_head import (
    ClassificationHead,
    ClassificationHeadConfig,
)
from ...heads.regression_head import RegressionHead, RegressionHeadConfig

# A1: no imports from ``src.training`` — keep the dependency arrow
# pointing strictly training -> models.

logger = logging.getLogger(__name__)


@dataclass
class PropagandaDetectorConfig:
    """Standalone-test config for :class:`PropagandaDetector`.

    CFG4: see :class:`BiasClassifierConfig` — the canonical source for
    ``model_name`` / ``pooling`` / ``dropout`` / ``device`` /
    ``label_smoothing`` in production is :class:`MultiTaskModelConfig`,
    via :meth:`PropagandaDetector.from_model_config`. This dataclass
    exists so the detector can be instantiated in isolation (unit
    tests, single-task experiments).
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    label_smoothing: float = 0.1
    device: Optional[str] = None
    use_regression_head: bool = False
    regression_output_dim: int = 1
    regression_hidden_dim: Optional[int] = None
    regression_activation: str = "gelu"
    # P3: opt-in gradient checkpointing (off by default).
    enable_gradient_checkpointing: bool = False


class PropagandaDetector(BaseModel):

    NUM_CLASSES = 2

    def __init__(self, config: PropagandaDetectorConfig) -> None:
        super().__init__()

        if not isinstance(config, PropagandaDetectorConfig):
            raise TypeError("config must be PropagandaDetectorConfig")

        self.config = config

        # -------------------------------------------------
        # Encoder
        # -------------------------------------------------

        self.encoder = EncoderFactory.create_transformer_encoder(
            EncoderConfig(
                model_name=config.model_name,
                pooling=config.pooling,
                device=config.device,
            )
        )

        if (
            config.enable_gradient_checkpointing
            and hasattr(self.encoder, "gradient_checkpointing_enable")
        ):
            self.encoder.gradient_checkpointing_enable()

        # -------------------------------------------------
        # Head
        # -------------------------------------------------

        self.classifier_head = ClassificationHead(
            ClassificationHeadConfig(
                input_dim=self.encoder.hidden_size,
                num_classes=self.NUM_CLASSES,
                dropout=config.dropout,
                return_features=False,
            )
        )

        self.regression_head: Optional[RegressionHead] = None

        if config.use_regression_head:
            self.regression_head = RegressionHead(
                RegressionHeadConfig(
                    input_dim=self.encoder.hidden_size,
                    output_dim=config.regression_output_dim,
                    hidden_dim=config.regression_hidden_dim,
                    dropout=config.dropout,
                    activation=config.regression_activation,
                )
            )

        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing,
        )

        # -------------------------------------------------
        # Calibration
        # -------------------------------------------------

        self.temperature = nn.Parameter(torch.ones(1))

        logger.info(
            "PropagandaDetector initialized | model=%s | classes=%d",
            config.model_name,
            self.NUM_CLASSES,
        )

    # -----------------------------------------------------
    # FORWARD
    # -----------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = encoder_outputs["pooled_output"]

        if not pooled_output.is_contiguous():
            pooled_output = pooled_output.contiguous()

        head_output = self.classifier_head(pooled_output)

        raw_logits = head_output["logits"]

        # Temperature scaling is a post-hoc calibration step. Apply it
        # only at inference; compute the loss on raw logits so gradients
        # cannot reduce loss by inflating T.
        if self.training:
            scaled_logits = raw_logits
        else:
            temperature = torch.clamp(self.temperature, 0.5, 5.0)
            scaled_logits = raw_logits / temperature

        outputs: Dict[str, Any] = {
            "logits": scaled_logits,
            "embeddings": pooled_output,
        }

        # P1: derived statistics only run at inference. Mirror the
        # classification head, which now also skips them in train mode.
        if not self.training:
            probs = F.softmax(scaled_logits, dim=-1)
            preds = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values

            outputs["probabilities"] = probs
            outputs["predictions"] = preds
            outputs["confidence"] = confidence

            entropy = head_output.get("entropy")
            if entropy is not None:
                outputs["entropy"] = entropy

        if self.regression_head is not None:
            outputs["regression"] = self.regression_head(pooled_output)

        if labels is not None:

            if labels.dim() != 1:
                raise ValueError("labels must be 1D tensor")

            if not ((labels >= 0).all() and (labels < self.NUM_CLASSES).all()):
                raise ValueError("labels out of range")

            loss = self.loss_fn(raw_logits, labels.long())
            outputs["loss"] = loss

        return outputs

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        finally:
            if was_training:
                self.train()

        return {
            "predictions": outputs["predictions"],
            "probabilities": outputs["probabilities"],
            "confidence": outputs["confidence"],
        }

    # -----------------------------------------------------
    # LABELS
    # -----------------------------------------------------

    def get_output_labels(self) -> Dict[int, str]:
        return {
            0: "non_propaganda",
            1: "propaganda",
        }

    # -----------------------------------------------------
    # FACTORIES
    # -----------------------------------------------------

    @classmethod
    def from_task_config(
        cls,
        task_config: TaskConfig,
        head_config: HeadConfig,
        model_name: str = "roberta-base",
        pooling: str = "cls",
        device: Optional[str] = None,
        label_smoothing: float = 0.1,
    ) -> "PropagandaDetector":

        cfg = PropagandaDetectorConfig(
            model_name=model_name,
            pooling=pooling,
            dropout=head_config.dropout,
            label_smoothing=label_smoothing,
            device=device,
            use_regression_head=(
                task_config.regression.enabled
                if task_config.regression is not None
                else False
            ),
            regression_output_dim=(
                task_config.regression.output_dim
                if task_config.regression is not None
                else 1
            ),
            regression_hidden_dim=(
                task_config.regression.hidden_dim
                if task_config.regression is not None
                else None
            ),
            regression_activation=(
                task_config.regression.activation
                if task_config.regression is not None
                else "gelu"
            ),
        )

        return cls(cfg)

    @classmethod
    def from_model_config(
        cls,
        model_config: MultiTaskModelConfig,
    ) -> "PropagandaDetector":

        task_cfg = model_config.tasks.get("propaganda")

        if task_cfg is None:
            raise KeyError("Task 'propaganda' not found")

        return cls.from_task_config(
            task_config=task_cfg,
            head_config=HeadConfig(
                input_dim=0,
                output_dim=task_cfg.num_labels,
                dropout=model_config.dropout,
            ),
            model_name=model_config.encoder.model_name,
            pooling=model_config.encoder.pooling,
            device=model_config.encoder.device,
            label_smoothing=float(
                model_config.metadata.get("propaganda_label_smoothing", 0.1)
            ),
        )

    # A1: ``create_trainer`` removed. Trainer construction lives in
    # ``src.training`` (see ``src.training.create_trainer_fn``); the
    # models layer must not depend on the training layer.