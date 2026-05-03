from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn

from ...base.base_model import BaseModel
from ...config import HeadConfig, TaskConfig, MultiTaskModelConfig
from ...encoder.encoder_config import EncoderConfig
from ...encoder.encoder_factory import EncoderFactory
from ...heads.multilabel_head import MultiLabelHead, MultiLabelHeadConfig
from ...heads.regression_head import RegressionHead, RegressionHeadConfig

# A1: no imports from ``src.training`` — the models package must not
# depend on the training layer.

logger = logging.getLogger(__name__)


@dataclass
class NarrativeDetectorConfig:
    """Standalone-test config for :class:`NarrativeDetector`.

    CFG4: see :class:`BiasClassifierConfig` — the canonical source for
    ``model_name`` / ``pooling`` / ``dropout`` / ``device`` is
    :class:`MultiTaskModelConfig`, via
    :meth:`NarrativeDetector.from_model_config`. This dataclass exists
    so the detector can be instantiated in isolation (unit tests,
    single-task experiments).
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    threshold: float = 0.5
    device: Optional[str] = None
    use_regression_head: bool = False
    regression_output_dim: int = 1
    regression_hidden_dim: Optional[int] = None
    regression_activation: str = "gelu"
    # P3: opt-in gradient checkpointing (off by default).
    enable_gradient_checkpointing: bool = False


class NarrativeDetector(BaseModel):
    """Single-task narrative-role detector.

    Predicts the three narrative roles — ``hero`` / ``villain`` /
    ``victim`` — as a 3-way multi-label head. This matches the
    canonical dataset schema:

        narrative columns: text, hero, villain, victim,
                           hero_entities, villain_entities, victim_entities

    The three ``*_entities`` columns are *text spans*, not classification
    targets — they identify which entities fill each role and are
    consumed by the entity-extraction stage downstream, never by this
    head. Narrative *frames* (``RE`` / ``HI`` / ``CO`` / ``MO`` /
    ``EC``) live on a separate ``narrative_frame`` head; see
    :class:`MultiTaskTruthLensModel` for the wired-up multi-task path.

    .. note::
       The previous version of this class concatenated the 3 roles,
       3 entity-marker labels, and 5 frame labels into a single
       11-class head. That conflated three orthogonal axes under one
       loss signal and produced the well-known shape mismatch:
       model emits ``[B, 11]`` but the dataset only carries 3 role
       columns, giving ``labels [B, 3]`` vs ``logits [B, 11]``.
    """

    LABELS: List[str] = [
        "hero",
        "villain",
        "victim",
    ]

    NUM_LABELS = len(LABELS)

    LABEL_MAPPING = {i: label for i, label in enumerate(LABELS)}

    def __init__(self, config: NarrativeDetectorConfig):
        super().__init__()

        if not isinstance(config, NarrativeDetectorConfig):
            raise TypeError("config must be NarrativeDetectorConfig")

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

        self.classifier_head = MultiLabelHead(
            MultiLabelHeadConfig(
                input_dim=self.encoder.hidden_size,
                num_labels=self.NUM_LABELS,
                dropout=config.dropout,
                threshold=config.threshold,
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

        logger.info(
            "NarrativeDetector initialized | model=%s | labels=%d",
            config.model_name,
            self.NUM_LABELS,
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

        # SHAPE-1 (mirrors the EmotionClassifier fix): if the dataset
        # ever drops single-class narrative columns, the label tensor
        # arrives as ``(B, K)`` with ``K < NUM_LABELS``, while the head
        # still emits ``(B, NUM_LABELS)`` logits. The reduction to the
        # surviving columns is handled by
        # ``training.task_loss_router._multilabel_loss`` via
        # ``index_select`` against ``cfg.valid_label_indices`` — passing
        # the raw labels into ``MultiLabelHead`` would fail its strict
        # ``labels.shape == logits.shape`` check before the loss router
        # ever sees them, and the head's ``BCEWithLogitsLoss`` is
        # redundant with ``LossEngine.compute`` (which reads ``logits``
        # and ``batch["labels"]`` directly and ignores any
        # ``outputs["loss"]`` the model emits). Validate width here and
        # stop forwarding labels into the head.
        if labels is not None:
            if labels.dim() != 2:
                raise ValueError(
                    "labels must be a 2-D tensor of shape (batch_size, K)"
                )
            if labels.shape[-1] > self.NUM_LABELS:
                raise ValueError(
                    f"labels width {labels.shape[-1]} exceeds the model's "
                    f"output width {self.NUM_LABELS}"
                )

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = encoder_outputs["pooled_output"]

        if not pooled_output.is_contiguous():
            pooled_output = pooled_output.contiguous()

        head_outputs = self.classifier_head(pooled_output)

        outputs: Dict[str, Any] = {
            "logits": head_outputs["logits"],
            "embeddings": pooled_output,
        }

        # P1: derived stats are inference-only. The multilabel head
        # also skips them in train mode, so use ``.get`` to stay safe.
        if not self.training:
            for key in ("probabilities", "predictions", "confidence", "entropy"):
                value = head_outputs.get(key)
                if value is not None:
                    outputs[key] = value

        if self.regression_head is not None:
            outputs["regression"] = self.regression_head(pooled_output)

        return outputs

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:

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

        probs = outputs["probabilities"]

        if threshold is not None:
            preds = (probs >= float(threshold)).long()
        else:
            preds = outputs["predictions"]

        return {
            "predictions": preds,
            "probabilities": probs,
            "confidence": outputs["confidence"],
            "labels": self.LABEL_MAPPING,
        }

    # -----------------------------------------------------
    # LABELS
    # -----------------------------------------------------

    def get_output_labels(self) -> Dict[int, str]:
        return self.LABEL_MAPPING

    def get_label_list(self) -> List[str]:
        return list(self.LABELS)

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
        threshold: float = 0.5,
    ) -> "NarrativeDetector":

        cfg = NarrativeDetectorConfig(
            model_name=model_name,
            pooling=pooling,
            dropout=head_config.dropout,
            threshold=threshold,
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
    ) -> "NarrativeDetector":

        task_cfg = model_config.tasks.get("narrative")

        if task_cfg is None:
            raise KeyError("Task 'narrative' not found")

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
                model_config.metadata.get("narrative_threshold", 0.5)
            ),
        )

    # A1: ``create_trainer`` removed. Trainer construction lives in
    # ``src.training`` (see ``src.training.create_trainer_fn``); the
    # models layer must not depend on the training layer.