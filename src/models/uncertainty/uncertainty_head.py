from __future__ import annotations

import math
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# UTILS
# =========================================================

def _softplus(x: torch.Tensor) -> torch.Tensor:
    # stable positive transform
    return F.softplus(x) + EPS


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + EPS), dim=-1)


# =========================================================
# GAUSSIAN REGRESSION HEAD (HETEROSCEDASTIC)
# =========================================================

class GaussianRegressionHead(nn.Module):
    """
    Predicts mean and variance for regression tasks.

    Outputs:
        mean: (B, D)
        variance: (B, D)  (positive)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.dropout = nn.Dropout(dropout)

        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.dropout(self.act(self.fc1(x)))

        mean = self.mean_head(h)
        logvar = self.logvar_head(h)

        var = _softplus(logvar)

        return {
            "mean": mean,
            "variance": var,
            "log_variance": logvar,
        }


# =========================================================
# CLASSIFICATION UNCERTAINTY HEAD
# =========================================================

class ClassificationUncertaintyHead(nn.Module):
    """
    Predicts logits and uncertainty (variance / confidence proxy).

    Outputs:
        logits: (B, C)
        variance: (B, C) (optional uncertainty per class)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        predict_variance: bool = True,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.logits_head = nn.Linear(hidden_dim, num_classes)

        self.predict_variance = predict_variance
        if predict_variance:
            self.var_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.dropout(self.act(self.fc1(x)))

        logits = self.logits_head(h)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
        }

        if self.predict_variance:
            logvar = self.var_head(h)
            var = _softplus(logvar)

            out["variance"] = var
            out["log_variance"] = logvar

        return out


# =========================================================
# DIRICHLET EVIDENTIAL HEAD
# =========================================================

class DirichletEvidentialHead(nn.Module):
    """
    Evidential Deep Learning head for classification.

    Outputs evidence -> Dirichlet parameters:
        alpha = evidence + 1

    Returns:
        logits (for compatibility),
        evidence,
        alpha,
        probs,
        uncertainty
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_dim = hidden_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.evidence_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.dropout(self.act(self.fc1(x)))

        evidence = F.relu(self.evidence_head(h))
        alpha = evidence + 1.0

        S = torch.sum(alpha, dim=-1, keepdim=True)

        probs = alpha / (S + EPS)

        # epistemic uncertainty (Dirichlet)
        uncertainty = alpha.shape[-1] / (S + EPS)

        logits = torch.log(probs + EPS)

        return {
            "logits": logits,
            "evidence": evidence,
            "alpha": alpha,
            "probs": probs,
            "uncertainty": uncertainty.squeeze(-1),
        }


# =========================================================
# WRAPPER HEAD
# =========================================================

class UncertaintyHead(nn.Module):
    """
    Unified uncertainty head wrapper.

    Modes:
        - "gaussian"
        - "classification"
        - "evidential"
    """

    def __init__(
        self,
        mode: str,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.mode = mode

        if mode == "gaussian":
            self.head = GaussianRegressionHead(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        elif mode == "classification":
            self.head = ClassificationUncertaintyHead(
                input_dim=input_dim,
                num_classes=output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        elif mode == "evidential":
            self.head = DirichletEvidentialHead(
                input_dim=input_dim,
                num_classes=output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        out = self.head(x)

        # add generic confidence + entropy if classification-like
        if "logits" in out:
            probs = F.softmax(out["logits"], dim=-1)
            out["probabilities"] = probs
            out["confidence"] = torch.max(probs, dim=-1).values
            out["entropy"] = _entropy_from_probs(probs)

        return out