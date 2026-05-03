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
class EnsembleConfig:
    strategy: str = "average"  # average | weighted | majority_vote
    weights: Optional[List[float]] = None
    device: str = "cpu"
    return_probabilities: bool = True


# =========================================================
# MODEL
# =========================================================

class EnsembleModel(nn.Module):

    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[EnsembleConfig] = None,
    ) -> None:
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided")

        self.config = config or EnsembleConfig()
        self.models = nn.ModuleList(models)

        if self.config.weights is not None:
            if len(self.config.weights) != len(models):
                raise ValueError("weights length mismatch")

            weights_tensor = torch.tensor(
                self.config.weights,
                dtype=torch.float32,
            )
            self.register_buffer("_weights", weights_tensor)
        else:
            self._weights = None

        # G3: ``self.config.device`` is a *string* captured once at
        # construction, but PyTorch can move the module across devices
        # at any time (``.to(...)``, DDP, autocast contexts, etc.). We
        # therefore record the *initial* placement only as a hint and
        # always derive the current device from the live parameters at
        # forward time via ``_runtime_device()``. ``add_model`` and the
        # initial ``self.to(...)`` use the hint so members are pre-moved.
        self._init_device_hint = torch.device(self.config.device)
        self.to(self._init_device_hint)

        logger.info(
            "EnsembleModel | strategy=%s | models=%d",
            self.config.strategy,
            len(models),
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def _runtime_device(self) -> torch.device:
        """Live device of the ensemble (G3).

        Reading from a parameter tracks any subsequent ``.to(...)`` /
        DDP move; the captured-at-init ``self.config.device`` string
        does not. Falls back to the construction-time hint if the
        module somehow has no parameters yet (e.g. empty stub models
        in tests).
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._init_device_hint

    def forward(self, *args, **kwargs) -> Dict[str, Any]:

        logits_list: List[torch.Tensor] = []

        # A5.5: free the per-member ``output`` dict (which can carry
        # probabilities, attention maps, hidden states …) as soon as
        # we've extracted the one tensor we actually need. On CUDA we
        # also nudge the allocator with ``empty_cache`` so the freed
        # blocks become available to the next member's forward — at
        # ensemble sizes >= 5 this materially reduces peak memory on
        # 80 GB cards. ``empty_cache`` is a host-side hint, not a
        # synchronisation, and is gated on actually being on CUDA.
        on_cuda = (
            torch.cuda.is_available()
            and self._runtime_device().type == "cuda"
        )

        for model in self.models:
            output = model(*args, **kwargs)
            logits = extract_logits(output)
            logits_list.append(logits)
            del output
            if on_cuda:
                torch.cuda.empty_cache()

        stacked = torch.stack(logits_list, dim=0)

        if self.config.strategy == "majority_vote":
            # `_majority_vote` returns per-class vote *counts*. These are
            # not real-valued logits, so feeding them into softmax (the
            # general-strategy code path below) would smear the
            # probability mass across all classes — exactly the bug the
            # audit flagged. Convert counts directly to a normalized
            # probability distribution and recover a logit-shaped tensor
            # by taking the log of those probabilities.
            vote_counts = self._majority_vote(stacked)
            denom = vote_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
            probs = vote_counts / denom
            logits = torch.log(probs.clamp(min=1e-12))
            log_probs = logits  # already in log-space (== log(probs))

        elif self.config.strategy == "weighted" and self._weights is not None:
            # G3: align weights with the *live* tensor's device, not the
            # init-time string captured in ``self.config.device``.
            weights = self._weights.to(stacked.device).view(
                -1, *([1] * (stacked.dim() - 1))
            )
            logits = (stacked * weights).sum(dim=0) / weights.sum()
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

        else:
            logits = stacked.mean(dim=0)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

        preds = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values
        # N1: entropy in log-space — never ``log(probs + EPS)``.
        entropy = -(probs * log_probs).sum(dim=-1)

        return {
            "logits": logits,
            "probabilities": probs if self.config.return_probabilities else None,
            "predictions": preds,
            "confidence": confidence,
            "entropy": entropy,
        }

    # =====================================================
    # MAJORITY VOTE
    # =====================================================

    def _majority_vote(self, stacked: torch.Tensor) -> torch.Tensor:
        """Return per-class vote *counts* (not probabilities or logits).

        ``stacked`` is shape ``[num_models, ..., num_classes]``. For each
        model we take its ``argmax`` over the class dimension and add a
        one-hot vote into the accumulator. The caller is responsible for
        turning these counts into a probability distribution.
        """

        predictions = stacked.argmax(dim=-1)

        vote_counts = torch.zeros_like(stacked[0])

        for pred in predictions:
            one_hot = torch.zeros_like(vote_counts)
            one_hot.scatter_(-1, pred.unsqueeze(-1), 1.0)
            vote_counts += one_hot

        return vote_counts

    # =====================================================
    # UTILS
    # =====================================================

    def add_model(self, model: nn.Module) -> None:
        # P2 + G3: keep the device-placement invariant — every member of
        # ``self.models`` lives on the *current* runtime device, which
        # may differ from the init-time hint after a ``.to(...)`` move.
        self.models.append(model.to(self._runtime_device()))

    def get_num_models(self) -> int:
        return len(self.models)