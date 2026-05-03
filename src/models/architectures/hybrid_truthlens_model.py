"""Hybrid Transformer + engineered-feature fusion model.

Moved here from ``src/features/fusion/feature_scaling.py`` (audit task 1).
That file is for the per-feature numeric scaler — the model does not
belong in the features layer.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


# =========================================================
# DEFAULT TASK HEAD SIZES
# =========================================================
#
# These defaults are *aligned with the standalone task classifiers*
# (BiasClassifier.NUM_CLASSES, IdeologyClassifier.NUM_CLASSES, ...) so
# the hybrid model and the per-task models agree on what each head
# means. Callers that need different shapes (different label
# vocabularies, frame schemas, etc.) should pass `task_num_labels=...`
# or build the hybrid via :meth:`HybridTruthLensModel.from_model_config`.

DEFAULT_TASK_NUM_LABELS: Dict[str, int] = {
    "bias": 2,            # matches BiasClassifier.NUM_CLASSES
    "propaganda": 2,      # matches PropagandaDetector.NUM_CLASSES
    "ideology": 3,        # matches IdeologyClassifier.NUM_CLASSES
    "narrative": 3,       # hero / villain / victim role aggregation
    "narrative_frame": 5, # RE / HI / CO / MO / EC
    "emotion": 11,        # EMOTION-11: matches EmotionClassifier.NUM_EMOTIONS
}


class HybridTruthLensModel(nn.Module):
    """Hybrid Transformer + Engineered Feature Model.

    Multi-head architecture:
      * encoder      : a HuggingFace transformer (RoBERTa / XLM-R / ...)
      * feature_proj : projection of the engineered feature vector into
                       ``hidden_dim``
      * fusion       : concat(cls, projected_features) → hidden_dim
      * task heads   : one ``nn.Linear(hidden_dim, num_labels)`` per
                       configured task

    The engineered feature vector must be pre-scaled by
    :class:`src.features.fusion.feature_scaling.FeatureScalingPipeline`
    using a scaler fitted on the training set.

    Per-task head widths are taken from ``task_num_labels`` (or
    :data:`DEFAULT_TASK_NUM_LABELS` when omitted). The defaults match
    the canonical per-task classifier modules so a model trained with
    the hybrid architecture is interchangeable with the standalone
    classifier suite.
    """

    def __init__(
        self,
        model_name: str,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        task_num_labels: Optional[Mapping[str, int]] = None,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        encoder_dim = self.encoder.config.hidden_size

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # -------------------------------------------------
        # Resolve per-task head sizes (parameterised, no more
        # hardcoded mismatches with the standalone classifiers).
        # -------------------------------------------------
        resolved = dict(DEFAULT_TASK_NUM_LABELS)
        if task_num_labels:
            for name, n in task_num_labels.items():
                if int(n) <= 0:
                    raise ValueError(
                        f"task_num_labels[{name!r}] must be positive "
                        f"(got {n})"
                    )
                resolved[name] = int(n)

        self.task_num_labels: Dict[str, int] = resolved

        self.task_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, n) for name, n in resolved.items()}
        )

        # CFG1: explicit Xavier init for the projection / fusion /
        # classification heads. ``nn.Linear`` defaults to
        # ``kaiming_uniform_`` which is tuned for ReLU *hidden* layers
        # — fine for ``feature_proj`` and ``fusion`` but it leaves the
        # task heads with a noticeably wider output distribution than
        # the per-task standalone classifiers (which initialise on top
        # of HF's pretrained head with much smaller variance). The
        # mismatch shows up at evaluation time as the hybrid model
        # being systematically over-confident on cold-start runs. We
        # re-init explicitly so behaviour is deterministic and matches
        # the rest of the model family.
        self._init_weights()

    # =====================================================
    # INIT  (CFG1)
    # =====================================================

    def _init_weights(self) -> None:
        for module in (self.feature_proj, self.fusion):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        for head in self.task_heads.values():
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, input_ids, attention_mask, features) -> Dict[str, Any]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = outputs.last_hidden_state[:, 0]

        feat = self.feature_proj(features)

        fused = torch.cat([cls, feat], dim=1)
        fused = self.fusion(fused)

        return {name: head(fused) for name, head in self.task_heads.items()}

    # =====================================================
    # CONFIG-DRIVEN FACTORY
    # =====================================================

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: Optional[float] = None,
    ) -> "HybridTruthLensModel":
        """Build a hybrid model from a :class:`MultiTaskModelConfig`.

        Head widths are taken from ``model_config.tasks[name].num_labels``
        so the hybrid head sizes can never silently disagree with the
        rest of the pipeline.
        """

        from src.models.config import MultiTaskModelConfig

        if not isinstance(model_config, MultiTaskModelConfig):
            raise TypeError(
                "model_config must be a MultiTaskModelConfig "
                f"(got {type(model_config).__name__})"
            )

        if not model_config.tasks:
            raise ValueError("model_config.tasks must be non-empty")

        task_num_labels = {
            name: int(task_cfg.num_labels)
            for name, task_cfg in model_config.tasks.items()
        }

        return cls(
            model_name=model_config.encoder.model_name,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=(
                dropout if dropout is not None else float(model_config.dropout)
            ),
            task_num_labels=task_num_labels,
        )
