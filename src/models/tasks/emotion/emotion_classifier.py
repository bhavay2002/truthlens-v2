from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ...base.base_model import BaseModel
from ...config import HeadConfig, TaskConfig, MultiTaskModelConfig
from ...encoder.encoder_config import EncoderConfig
from ...encoder.encoder_factory import EncoderFactory
from ...heads.multilabel_head import MultiLabelHead, MultiLabelHeadConfig
from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    NUM_EMOTION_LABELS,
)

# A1: no imports from ``src.training`` — the models package must not
# depend on the training layer.

logger = logging.getLogger(__name__)


@dataclass
class EmotionClassifierConfig:
    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    device: Optional[str] = None
    threshold: float = 0.5
    # P3: opt-in gradient checkpointing (off by default).
    enable_gradient_checkpointing: bool = False


class EmotionClassifier(BaseModel):

    # EMOTION-11: derived from the canonical schema (was hardcoded 20).
    # The live count is set in src/features/emotion/emotion_schema.py;
    # this attribute mirrors it so model code can use a class-level
    # constant without a circular import path.
    NUM_EMOTIONS = NUM_EMOTION_LABELS

    def __init__(self, config: EmotionClassifierConfig) -> None:
        super().__init__()

        if not isinstance(config, EmotionClassifierConfig):
            raise TypeError("config must be EmotionClassifierConfig")

        self.config = config

        # ------------------------------------------------
        # Encoder
        # ------------------------------------------------

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

        # ------------------------------------------------
        # Head
        # ------------------------------------------------

        self.classifier_head = MultiLabelHead(
            MultiLabelHeadConfig(
                input_dim=self.encoder.hidden_size,
                num_labels=self.NUM_EMOTIONS,
                dropout=config.dropout,
                threshold=config.threshold,
                return_features=False,
            )
        )

        logger.info(
            "EmotionClassifier initialized | model=%s | num_emotions=%d",
            config.model_name,
            self.NUM_EMOTIONS,
        )

    # ------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        # SHAPE-1: ``training.loss_balancer.plan_for_dataframe`` may drop
        # single-class (degenerate) emotion columns from the *dataset* —
        # the resulting label tensor has shape ``(B, K)`` where ``K`` is
        # the number of surviving columns (``K <= NUM_EMOTIONS``). The
        # model still emits full-width ``(B, NUM_EMOTIONS)`` logits, and
        # ``training.task_loss_router._multilabel_loss`` is the layer
        # responsible for ``index_select``-ing those logits down to the
        # surviving columns using ``cfg.valid_label_indices`` so the loss
        # is computed against the reduced label set. The previous strict
        # ``labels.shape[-1] != NUM_EMOTIONS`` check rejected exactly that
        # legitimate path and crashed the pipeline as soon as the emotion
        # task hit the trainer (e.g. "Dropped 9 single-class multilabel
        # column(s) → kept 11/20" → ``ValueError: labels must have shape
        # (batch_size, 20)``). Relax to "labels must be 2-D and not wider
        # than the model" — the loss router enforces the exact slice.
        if labels is not None:
            if labels.dim() != 2:
                raise ValueError(
                    "labels must be a 2-D tensor of shape (batch_size, K)"
                )
            if labels.shape[-1] > self.NUM_EMOTIONS:
                raise ValueError(
                    f"labels width {labels.shape[-1]} exceeds the model's "
                    f"output width {self.NUM_EMOTIONS}"
                )

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = encoder_outputs["pooled_output"]

        if not pooled_output.is_contiguous():
            pooled_output = pooled_output.contiguous()

        # SHAPE-1 (cont.): do NOT forward ``labels`` into the head. The
        # head's ``BCEWithLogitsLoss`` would crash on the same shape
        # mismatch (logits ``(B, NUM_EMOTIONS)`` vs reduced labels
        # ``(B, K)``), and the head-computed ``loss`` is dead weight in
        # the training loop anyway: ``LossEngine.compute`` reads
        # ``outputs["task_logits"]`` (synthesised from ``outputs["logits"]``
        # for single-task callers) and ``batch["labels"]`` directly,
        # ignoring any ``outputs["loss"]`` the model might emit. Computing
        # a second BCE inside the head was wasted work even before the
        # column-dropping path made it crash. The external loss engine
        # remains the single source of truth.
        head_outputs = self.classifier_head(pooled_output)

        outputs: Dict[str, Any] = {
            "logits": head_outputs["logits"],
            "embeddings": pooled_output,
        }

        # P1: derived stats only at inference (the multilabel head also
        # skips them in train mode).
        if not self.training:
            for key in ("probabilities", "predictions", "confidence", "entropy"):
                value = head_outputs.get(key)
                if value is not None:
                    outputs[key] = value

        return outputs

    # ------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # LABELS
    # ------------------------------------------------------------

    def get_output_labels(self) -> Dict[int, str]:
        return {i: EMOTION_LABELS[i] for i in range(self.NUM_EMOTIONS)}

    def get_training_labels(self) -> Dict[int, str]:
        return {i: f"emotion{i}" for i in range(self.NUM_EMOTIONS)}

    # ------------------------------------------------------------
    # FACTORIES
    # ------------------------------------------------------------

    @classmethod
    def from_task_config(
        cls,
        task_config: TaskConfig,
        head_config: HeadConfig,
        model_name: str = "roberta-base",
        pooling: str = "cls",
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> "EmotionClassifier":

        cfg = EmotionClassifierConfig(
            model_name=model_name,
            pooling=pooling,
            dropout=head_config.dropout,
            device=device,
            threshold=threshold,
        )

        return cls(cfg)

    @classmethod
    def from_model_config(
        cls,
        model_config: MultiTaskModelConfig,
    ) -> "EmotionClassifier":

        task_cfg = model_config.tasks.get("emotion")

        if task_cfg is None:
            raise KeyError("Task 'emotion' not found")

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
            threshold=float(
                model_config.metadata.get("emotion_threshold", 0.5)
            ),
        )

    # A1: ``create_trainer`` removed. Trainer construction lives in
    # ``src.training`` (see ``src.training.create_trainer_fn``); the
    # models layer must not depend on the training layer.