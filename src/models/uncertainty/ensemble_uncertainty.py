from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =========================================================
# UTILS
# =========================================================

def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# =========================================================
# C1.7: NUMERICALLY STABLE ENTROPY / MUTUAL-INFORMATION
# =========================================================
#
# The previous implementation computed predictive entropy and mutual
# information as ``-sum(p * log(p + EPS))`` over numpy arrays. That
# pattern has two well-known failure modes:
#
#   1. Bias near saturation. For p ≈ 0 the term ``log(p + EPS)`` is
#      not zero — it asymptotes to ``log(EPS)``, which silently inflates
#      every "low-confidence" entry by ``EPS * log(EPS)`` per class and
#      shifts the entropy floor away from 0. The bias is small per
#      element but accumulates into a *systematic* over-estimate of
#      epistemic uncertainty across the batch.
#
#   2. Loss of precision in the log-domain. Computing ``log(softmax(x))``
#      via ``log(exp(x) / Σ exp(x))`` cancels the dominant term twice;
#      ``F.log_softmax`` is the only formulation that retains float32
#      precision for logits with large dynamic range (≈ ±50). The
#      multilabel branch has the same issue — ``log(sigmoid(x) + EPS)``
#      vs the numerically stable ``F.logsigmoid(x)``.
#
# Fixes:
#   • Hold *per-member logits* internally (not pre-softmaxed numpy
#     probabilities) and route entropy / MI through ``log_softmax``
#     (multiclass) or ``logsigmoid`` + ``logsigmoid(-x)`` (multilabel).
#   • Average across ensemble members in log-space using ``logsumexp``
#     so the "mean probability" is also clamp-free.
#   • The legacy static helpers (``predictive_entropy`` /
#     ``mutual_information``) keep their numpy signature for back-compat
#     but use the ``np.where(p > 0, p*log(p), 0)`` idiom to honour
#     ``0 · log(0) = 0`` exactly, removing the EPS bias.

EPS = 1e-12  # only used inside the legacy numpy helpers for divisions, never inside log()


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def _sigmoid(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


# =========================================================
# ENSEMBLE UNCERTAINTY
# =========================================================

class EnsembleUncertainty:
    """
    Deep Ensemble uncertainty estimator.

    Uses multiple independently trained models to estimate:
    - predictive mean
    - variance
    - entropy
    - mutual information (epistemic uncertainty)
    """

    def __init__(
        self,
        models: List[nn.Module],
        task_type: str,
        device: Optional[torch.device] = None,
    ) -> None:
        if not models:
            raise ValueError("models list cannot be empty")

        self.models = models
        self.task_type = task_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        for model in self.models:
            model.to(self.device)
            model.eval()

    # =====================================================
    # FORWARD
    # =====================================================

    def _forward_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: Optional[str],
    ) -> torch.Tensor:

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
            )

        return outputs["logits"]

    def _collect_member_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: Optional[str],
    ) -> torch.Tensor:
        """Collect logits from every ensemble member, shape (M, B, C)."""

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        member_logits: List[torch.Tensor] = []
        for model in self.models:
            logits = self._forward_model(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
            )
            member_logits.append(logits)

        return torch.stack(member_logits, dim=0)  # (M, B, C)

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns:
            prob_samples: (M, B, C)
        """

        member_logits = self._collect_member_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=task,
        )

        if self.task_type == "multiclass":
            probs = _softmax(member_logits)
        else:
            probs = _sigmoid(member_logits)

        return _to_numpy(probs)

    # =====================================================
    # CORE: TORCH LOG-DOMAIN UNCERTAINTY
    # =====================================================

    def _uncertainty_from_logits(
        self,
        member_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute predictive mean, variance, entropy, MI, confidence
        directly from per-member logits in log-space.

        member_logits: (M, B, C)
        """

        if member_logits.dim() != 3:
            raise ValueError(
                f"member_logits must be (M, B, C); got {tuple(member_logits.shape)}"
            )

        M = member_logits.size(0)

        if self.task_type == "multiclass":
            # log_p_m: (M, B, C). Numerically stable softmax-log.
            log_p = F.log_softmax(member_logits, dim=-1)
            p = log_p.exp()

            # Per-member entropy: H[p_m] = -Σ_c p_m log p_m
            H_per_member = -(p * log_p).sum(dim=-1)              # (M, B)
            H_expected = H_per_member.mean(dim=0)                # (B,) — aleatoric

            # Mean probability over members in log-space:
            # log p̄ = logsumexp(log p_m, dim=0) - log M
            log_p_mean = torch.logsumexp(log_p, dim=0) - math.log(M)   # (B, C)
            p_mean = log_p_mean.exp()

            # Predictive entropy: H[p̄]
            H_predictive = -(p_mean * log_p_mean).sum(dim=-1)    # (B,) — total

            mutual_info = H_predictive - H_expected              # (B,) — epistemic

            variance = p.var(dim=0).mean(dim=-1)                 # (B,)
            confidence = p_mean.max(dim=-1).values               # (B,)

        else:
            # Multilabel: each output dim is an independent Bernoulli.
            # ``logsigmoid`` is the stable sibling of ``log(sigmoid(x))``.
            log_p = F.logsigmoid(member_logits)                  # (M, B, L)
            log_1mp = F.logsigmoid(-member_logits)               # (M, B, L)
            p = log_p.exp()
            one_minus_p = log_1mp.exp()

            # Per-member Bernoulli entropy averaged over labels.
            H_per_member = -(p * log_p + one_minus_p * log_1mp).mean(dim=-1)  # (M, B)
            H_expected = H_per_member.mean(dim=0)                # (B,)

            # Mean Bernoulli parameters in log-space — both sides via
            # logsumexp so we never need a clamp.
            log_p_mean = torch.logsumexp(log_p, dim=0) - math.log(M)        # (B, L)
            log_1mp_mean = torch.logsumexp(log_1mp, dim=0) - math.log(M)    # (B, L)
            p_mean = log_p_mean.exp()
            one_minus_p_mean = log_1mp_mean.exp()

            H_predictive = -(
                p_mean * log_p_mean + one_minus_p_mean * log_1mp_mean
            ).mean(dim=-1)                                       # (B,)

            mutual_info = H_predictive - H_expected              # (B,)

            variance = p.var(dim=0).mean(dim=-1)                 # (B,)
            confidence = p_mean.max(dim=-1).values               # (B,)

        return {
            "mean_probabilities": p_mean,
            "variance": variance,
            "entropy": H_predictive,
            "mutual_information": mutual_info,
            "confidence": confidence,
        }

    # =====================================================
    # LEGACY NUMPY METRICS (BACKWARDS-COMPAT API)
    # =====================================================

    @staticmethod
    def predictive_mean(prob_samples: np.ndarray) -> np.ndarray:
        return np.mean(prob_samples, axis=0)

    @staticmethod
    def predictive_variance(prob_samples: np.ndarray) -> np.ndarray:
        return np.var(prob_samples, axis=0).mean(axis=1)

    @staticmethod
    def predictive_entropy(prob_samples: np.ndarray) -> np.ndarray:
        # ``np.where(p > 0, p*log(p), 0)`` honours ``0 · log(0) = 0``
        # without the ``log(p + EPS)`` bias. This is a faithful
        # multiclass entropy of the *mean* distribution.
        mean_probs = np.mean(prob_samples, axis=0)
        plogp = np.where(mean_probs > 0, mean_probs * np.log(mean_probs), 0.0)
        return -plogp.sum(axis=-1)

    @staticmethod
    def mutual_information(prob_samples: np.ndarray) -> np.ndarray:
        mean_probs = np.mean(prob_samples, axis=0)

        plogp_mean = np.where(
            mean_probs > 0, mean_probs * np.log(mean_probs), 0.0
        )
        H_predictive = -plogp_mean.sum(axis=-1)

        plogp_member = np.where(
            prob_samples > 0, prob_samples * np.log(prob_samples), 0.0
        )
        H_expected = -plogp_member.sum(axis=-1).mean(axis=0)

        return H_predictive - H_expected

    # =====================================================
    # MAIN API
    # =====================================================

    def predict_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:

        member_logits = self._collect_member_logits(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task=task,
        )

        unc = self._uncertainty_from_logits(member_logits)

        results: Dict[str, Any] = {
            "mean_probabilities": _to_numpy(unc["mean_probabilities"]),
            "variance": _to_numpy(unc["variance"]),
            "entropy": _to_numpy(unc["entropy"]),
            "mutual_information": _to_numpy(unc["mutual_information"]),
            "confidence": _to_numpy(unc["confidence"]),
            "num_models": len(self.models),
        }

        return results


# =========================================================
# FUNCTIONAL API
# =========================================================

def ensemble_inference(
    models: List[nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    task_type: str,
    task: Optional[str] = None,
) -> Dict[str, Any]:

    ensemble = EnsembleUncertainty(
        models=models,
        task_type=task_type,
    )

    return ensemble.predict_with_uncertainty(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task=task,
    )
