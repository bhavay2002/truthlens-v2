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
class StackingEnsembleConfig:
    device: str = "cpu"
    return_probabilities: bool = True


# =========================================================
# MODEL
# =========================================================

class StackingEnsembleModel(nn.Module):

    def __init__(
        self,
        base_models: List[nn.Module],
        meta_model: nn.Module,
        config: Optional[StackingEnsembleConfig] = None,
    ) -> None:
        super().__init__()

        if not base_models:
            raise ValueError("At least one base model must be provided")

        if meta_model is None:
            raise ValueError("meta_model is required")

        self.config = config or StackingEnsembleConfig()

        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model

        self.device = torch.device(self.config.device)
        self.to(self.device)

        logger.info(
            "StackingEnsembleModel | base_models=%d",
            len(base_models),
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, *args, **kwargs) -> Dict[str, Any]:

        logits_list: List[torch.Tensor] = []

        # P2.2: do NOT move ``model`` onto ``self.device`` inside the
        # forward loop — every base model is already a child of this
        # ``nn.Module`` (registered via ``nn.ModuleList``) and the
        # ``self.to(self.device)`` call in ``__init__`` placed it
        # there. Re-issuing ``.to(self.device)`` per batch is at best a
        # no-op and at worst forces a synchronous device probe + tensor
        # walk on every step. Inputs are expected to already be on the
        # device by the caller convention used elsewhere in the model.
        for model in self.base_models:
            output = model(*args, **kwargs)
            logits = extract_logits(output)
            logits_list.append(logits)

        concatenated = torch.cat(logits_list, dim=-1)

        meta_output = self.meta_model(concatenated)
        logits = extract_logits(meta_output)

        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

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

    def add_base_model(self, model: nn.Module) -> None:
        self.base_models.append(model)

    def set_meta_model(self, model: nn.Module) -> None:
        self.meta_model = model

    def get_num_models(self) -> int:
        return len(self.base_models)