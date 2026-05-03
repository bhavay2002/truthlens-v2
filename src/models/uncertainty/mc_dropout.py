from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# UTILS
# =========================================================

def _enable_dropout(module: nn.Module) -> None:
    """
    Enable dropout layers during inference for MC Dropout.
    """
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def _sigmoid(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


# =========================================================
# CORE
# =========================================================

class MCDropoutPredictor:
    """
    Monte Carlo Dropout predictor for uncertainty estimation.

    Runs multiple stochastic forward passes with dropout enabled.
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str,
        mc_samples: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.task_type = task_type
        self.mc_samples = mc_samples
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

    # =====================================================
    # FORWARD SAMPLING
    # =====================================================

    def _forward_pass(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: Optional[str] = None,
    ) -> torch.Tensor:

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
            )

        return outputs["logits"]

    def sample_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns:
            prob_samples: (T, B, C)

        G2: build the sample stack on the *device* (one tensor per
        forward pass kept as a GPU tensor, then ``torch.stack`` once,
        then a single ``.cpu().numpy()`` at the end). The previous
        implementation called ``.detach().cpu().numpy()`` after every
        forward pass, which serialises ``mc_samples`` extra
        device→host syncs and dominates wall-time on GPU.
        """

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        _enable_dropout(self.model)

        samples: list[torch.Tensor] = []

        for _ in range(self.mc_samples):

            logits = self._forward_pass(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
            )

            if self.task_type == "multiclass":
                probs = _softmax(logits)
            else:
                probs = _sigmoid(logits)

            samples.append(probs.detach())

        # Single device→host transfer for the entire (T, B, C) block.
        return torch.stack(samples, dim=0).cpu().numpy()

    # =====================================================
    # UNCERTAINTY METRICS
    # =====================================================

    @staticmethod
    def predictive_mean(prob_samples: np.ndarray) -> np.ndarray:
        return np.mean(prob_samples, axis=0)

    @staticmethod
    def predictive_variance(prob_samples: np.ndarray) -> np.ndarray:
        return np.var(prob_samples, axis=0).mean(axis=1)

    @staticmethod
    def _xlogx(probs: np.ndarray) -> np.ndarray:
        """``p * log(p)`` with the convention ``0 * log(0) := 0``.

        N1: avoids the ``log(probs + EPS)`` formulation, whose additive
        bias dominates the log term whenever ``probs`` is small. We
        ``np.where`` zero-mass entries to ``0`` (the analytic limit)
        and ``log`` only the strictly positive entries, so the sum is
        exact for any peaked distribution.
        """
        probs = np.asarray(probs)
        out = np.zeros_like(probs, dtype=float)
        positive = probs > 0
        out[positive] = probs[positive] * np.log(probs[positive])
        return out

    @classmethod
    def predictive_entropy(cls, prob_samples: np.ndarray) -> np.ndarray:
        mean_probs = np.mean(prob_samples, axis=0)
        return -np.sum(cls._xlogx(mean_probs), axis=1)

    @classmethod
    def mutual_information(cls, prob_samples: np.ndarray) -> np.ndarray:
        mean_probs = np.mean(prob_samples, axis=0)

        entropy_mean = -np.sum(cls._xlogx(mean_probs), axis=1)
        entropy_expected = -np.mean(
            np.sum(cls._xlogx(prob_samples), axis=2),
            axis=0,
        )

        return entropy_mean - entropy_expected

    # =====================================================
    # MAIN API
    # =====================================================

    def predict_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:

        prob_samples = self.sample_predictions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=task,
        )

        mean_probs = self.predictive_mean(prob_samples)

        results = {
            "mean_probabilities": mean_probs,
            "variance": self.predictive_variance(prob_samples),
            "entropy": self.predictive_entropy(prob_samples),
            "mutual_information": self.mutual_information(prob_samples),
        }

        # confidence
        results["confidence"] = np.max(mean_probs, axis=1)

        return results


# =========================================================
# FUNCTIONAL API
# =========================================================

def mc_dropout_inference(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    task_type: str,
    mc_samples: int = 10,
    task: Optional[str] = None,
) -> Dict[str, Any]:

    predictor = MCDropoutPredictor(
        model=model,
        task_type=task_type,
        mc_samples=mc_samples,
    )

    return predictor.predict_with_uncertainty(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task=task,
    )