"""
NeuralAggregator — learned scoring over structured feature vectors.

Spec §4 (Aggregation Engine v2).

Three architectures
-------------------
MLPAggregator (§4.1 — production default)
    LayerNorm → Linear → GELU → Dropout → Linear → GELU → multi-output head

FeatureAttentionAggregator (§4.2 — recommended)
    Learns per-feature importance weights via a softmax attention gate.
    The attention vector doubles as the explanation output (§6.2).

Multi-output head (§5)
----------------------
Every architecture shares the same final head:
    credibility_score  = sigmoid(z)          — scalar in [0, 1]
    risk_logits        = Linear(z, 3)        — low / medium / high (raw)
    explanation_weights = softmax(Linear(z)) — per-feature importance

Usage
-----
    builder = AggregatorFeatureBuilder()
    agg     = NeuralAggregator.build(cfg, input_dim=builder.feature_dim)
    x       = torch.from_numpy(builder.build(model_outputs, signals, profile))
    out     = agg(x.unsqueeze(0))
    score   = out.credibility_score.item()   # scalar
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =========================================================
# OUTPUT CONTRACT
# =========================================================

class NeuralAggregatorOutput(NamedTuple):
    """Multi-output result from any NeuralAggregator variant.

    Attributes
    ----------
    credibility_score : (B,) float in [0, 1]
        Probability of the content being credible.
    risk_logits : (B, 3) float
        Raw logits for LOW / MEDIUM / HIGH risk classes. Caller applies
        softmax for probabilities or argmax for the predicted level.
    explanation_weights : (B, D) float
        Per-feature importance weights summing to 1. For the attention
        aggregator these are the learned attention scores; for the MLP
        variant they are gradient-based softmax importances computed
        from the final projection.
    """

    credibility_score: torch.Tensor
    risk_logits: torch.Tensor
    explanation_weights: torch.Tensor


# =========================================================
# SHARED MULTI-OUTPUT HEAD
# =========================================================

class _MultiOutputHead(nn.Module):
    """Shared final head shared by all aggregator architectures (spec §5)."""

    def __init__(self, in_features: int, feature_dim: int) -> None:
        super().__init__()
        self.cred_head = nn.Linear(in_features, 1)
        self.risk_head = nn.Linear(in_features, 3)
        self.exp_head  = nn.Linear(in_features, feature_dim)

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cred = torch.sigmoid(self.cred_head(z)).squeeze(-1)   # (B,)
        risk = self.risk_head(z)                               # (B, 3)
        exp_w = torch.softmax(self.exp_head(z), dim=-1)        # (B, D)
        return cred, risk, exp_w


# =========================================================
# MLP AGGREGATOR  (spec §4.1 — baseline)
# =========================================================

class MLPAggregator(nn.Module):
    """Baseline MLP aggregator (spec §4.1).

    Architecture
    ------------
    LayerNorm(D) → Linear(D, H) → GELU → Dropout(p) →
    Linear(H, 128) → GELU → MultiOutputHead
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
        )
        self.head = _MultiOutputHead(128, input_dim)

        self._init_weights()
        logger.info(
            "MLPAggregator | D=%d H=%d drop=%.2f | params=%d",
            input_dim, hidden_dim, dropout, self._n_params(),
        )

    def forward(self, x: torch.Tensor) -> NeuralAggregatorOutput:
        """
        Parameters
        ----------
        x : (B, D)

        Returns
        -------
        NeuralAggregatorOutput
        """
        z = self.net(x.float())
        cred, risk, exp_w = self.head(z)
        return NeuralAggregatorOutput(cred, risk, exp_w)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =========================================================
# FEATURE ATTENTION AGGREGATOR  (spec §4.2 — recommended)
# =========================================================

class FeatureAttentionAggregator(nn.Module):
    """Feature attention aggregator (spec §4.2).

    Architecture
    ------------
    x → LayerNorm(D) → attention_gate (softmax(W_attn x)) →
    z = gate ⊙ x_norm → Linear(D, H) → GELU → Dropout →
    Linear(H, 128) → GELU → MultiOutputHead

    The per-feature attention weights serve as the explanation output
    (§6.2 attention-based explanation), exposing which signals drove
    the credibility estimate without a separate backwards pass.

    Learning dynamics
    -----------------
    * W_attn has no bias — it maps feature magnitudes to relative
      importance without a learnable offset that could collapse all
      gates to the same value.
    * The gate is applied to the LayerNorm output, not the raw input,
      so the attention scores are scale-invariant across batches.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim

        self.norm = nn.LayerNorm(input_dim)
        self.attn = nn.Linear(input_dim, input_dim, bias=False)

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
        )
        self.head = _MultiOutputHead(128, input_dim)

        self._init_weights()
        logger.info(
            "FeatureAttentionAggregator | D=%d H=%d drop=%.2f | params=%d",
            input_dim, hidden_dim, dropout, self._n_params(),
        )

    def forward(self, x: torch.Tensor) -> NeuralAggregatorOutput:
        """
        Parameters
        ----------
        x : (B, D)

        Returns
        -------
        NeuralAggregatorOutput with learned attention as explanation_weights.
        """
        x_norm = self.norm(x.float())                        # (B, D)
        attn_logits = self.attn(x_norm)                      # (B, D)
        attn_weights = torch.softmax(attn_logits, dim=-1)    # (B, D)
        z_input = attn_weights * x_norm                      # (B, D)
        z = self.proj(z_input)                               # (B, 128)
        cred, risk, _ = self.head(z)
        return NeuralAggregatorOutput(cred, risk, attn_weights)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw (B, D) attention weight tensor for a feature batch."""
        with torch.no_grad():
            x_norm = self.norm(x.float())
            return torch.softmax(self.attn(x_norm), dim=-1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =========================================================
# FACTORY
# =========================================================

class NeuralAggregator:
    """Factory and checkpoint utilities for aggregator variants.

    This is NOT a subclass of ``nn.Module``. Call :meth:`build` to
    obtain the concrete module instance.

    Supported architectures
    -----------------------
    ``"mlp"``
        Spec §4.1. Fast, robust baseline. Use for latency-critical
        deployments (<1ms per sample on CPU).
    ``"attention"``
        Spec §4.2. Recommended default. Learns per-feature importance
        weights that are exposed as explanation output.
    """

    @classmethod
    def build(
        cls,
        config: "NeuralAggregatorConfig",
        input_dim: int,
    ) -> nn.Module:
        """Construct and return the configured aggregator module.

        Parameters
        ----------
        config : NeuralAggregatorConfig
            Architecture / hyper-parameter config.
        input_dim : int
            Dimensionality of the feature vector produced by
            ``AggregatorFeatureBuilder``. Must match the input_dim
            used at training time.

        Returns
        -------
        nn.Module (MLPAggregator or FeatureAttentionAggregator)
        """
        arch = getattr(config, "architecture", "attention")
        h    = getattr(config, "hidden_dim", 256)
        drop = getattr(config, "dropout", 0.2)

        if arch == "mlp":
            return MLPAggregator(input_dim, h, drop)
        if arch == "attention":
            return FeatureAttentionAggregator(input_dim, h, drop)

        raise ValueError(
            f"NeuralAggregator: unknown architecture {arch!r}. "
            "Supported: 'mlp', 'attention'."
        )

    @staticmethod
    def save(
        module: nn.Module,
        path: str | Path,
        *,
        input_dim: int,
        meta: Optional[Dict] = None,
    ) -> None:
        """Serialise ``module`` weights + metadata to ``path``.

        Format: ``torch.save`` dict with keys:
            ``state_dict``, ``input_dim``, ``architecture``, ``meta``

        Parameters
        ----------
        module : nn.Module
            Trained aggregator instance.
        path : str | Path
        input_dim : int
            Feature dimension (needed to reconstruct the module).
        meta : dict, optional
            Arbitrary metadata (epoch, val_auc, config snapshot, …)
            stored alongside the weights for provenance.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arch = (
            "attention"
            if isinstance(module, FeatureAttentionAggregator)
            else "mlp"
        )

        torch.save(
            {
                "state_dict": module.state_dict(),
                "input_dim": input_dim,
                "architecture": arch,
                "meta": meta or {},
            },
            path,
        )
        logger.info("NeuralAggregator saved → %s", path)

    @staticmethod
    def load(
        path: str | Path,
        config: "NeuralAggregatorConfig",
        *,
        device: str = "cpu",
        strict: bool = True,
    ) -> nn.Module:
        """Load aggregator weights from ``path``.

        Parameters
        ----------
        path : str | Path
            Checkpoint written by :meth:`save`.
        config : NeuralAggregatorConfig
            Config used to select the architecture (overridden by the
            ``architecture`` key stored in the checkpoint).
        device : str
            Target device.
        strict : bool
            Passed to ``module.load_state_dict``.

        Returns
        -------
        nn.Module in ``eval()`` mode on ``device``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"NeuralAggregator checkpoint not found: {path}"
            )

        ckpt = torch.load(path, map_location=device, weights_only=False)
        input_dim = ckpt["input_dim"]
        arch_in_ckpt = ckpt.get("architecture", getattr(config, "architecture", "attention"))

        # Override config arch with the one baked into the checkpoint
        import copy
        cfg = copy.copy(config)
        if hasattr(cfg, "__dict__"):
            cfg.__dict__["architecture"] = arch_in_ckpt
        elif hasattr(cfg, "model_copy"):
            cfg = cfg.model_copy(update={"architecture": arch_in_ckpt})

        module = NeuralAggregator.build(cfg, input_dim)
        module.load_state_dict(ckpt["state_dict"], strict=strict)
        module.to(device)
        module.eval()

        meta = ckpt.get("meta", {})
        logger.info(
            "NeuralAggregator loaded ← %s | arch=%s dim=%d meta=%s",
            path, arch_in_ckpt, input_dim, meta,
        )
        return module
