from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ensemble._utils import extract_logits

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class WeightedEnsembleConfig:
    weights: Optional[List[float]] = None
    device: str = "cpu"
    return_probabilities: bool = True


# =========================================================
# MODEL
# =========================================================

class WeightedEnsembleModel(nn.Module):

    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[WeightedEnsembleConfig] = None,
    ) -> None:
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided")

        self.config = config or WeightedEnsembleConfig()
        self.models = nn.ModuleList(models)

        if self.config.weights is not None:
            if len(self.config.weights) != len(models):
                raise ValueError("weights length mismatch")

            weights_tensor = torch.tensor(
                self.config.weights,
                dtype=torch.float32,
            )
        else:
            weights_tensor = torch.ones(len(models), dtype=torch.float32)

        self.register_buffer("_weights", weights_tensor)

        self.device = torch.device(self.config.device)
        self.to(self.device)

        logger.info(
            "WeightedEnsembleModel | models=%d",
            len(models),
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, *args, **kwargs) -> Dict[str, Any]:

        logits_list: List[torch.Tensor] = []

        # P2.2: do NOT move ``model`` onto ``self.device`` inside the
        # forward loop — every member is already a child of this
        # ``nn.Module`` (registered via ``nn.ModuleList``) and the
        # ``self.to(self.device)`` call in ``__init__`` placed it
        # there. Re-issuing ``.to(self.device)`` per batch is at best a
        # no-op and at worst forces a synchronous device probe + tensor
        # walk on every step. Inputs are expected to already be on the
        # device by the caller convention used elsewhere in the model.
        for model in self.models:
            output = model(*args, **kwargs)
            logits = extract_logits(output)
            logits_list.append(logits)

        stacked = torch.stack(logits_list, dim=0)

        weights = self._weights.to(stacked.device).view(
            -1, *([1] * (stacked.dim() - 1))
        )

        logits = (stacked * weights).sum(dim=0) / weights.sum()

        probs = F.softmax(logits, dim=-1)
        # N1-FIX: compute entropy in log-space (matches ensemble_model.py).
        # ``log(probs + 1e-12)`` introduces a bias when probs are near 1.0
        # because the additive epsilon shifts the argument off the simplex.
        # F.log_softmax reuses the existing softmax denominator and is exact.
        preds = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        return {
            "logits": logits,
            "probabilities": probs if self.config.return_probabilities else None,
            "predictions": preds,
            "confidence": confidence,
            "entropy": entropy,
        }

    # =====================================================
    # UTILS
    # =====================================================

    def set_weights(self, weights: List[float]) -> None:
        if len(weights) != len(self.models):
            raise ValueError("weights length mismatch")

        self._weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

    def get_weights(self) -> torch.Tensor:
        return self._weights.detach().cpu()

    def get_num_models(self) -> int:
        return len(self.models)