"""
AggregatorTrainer — training script for the NeuralAggregator.

Spec §8 (Aggregation Engine v2).

Loss function (spec §8.2)
--------------------------
    L = BCE(neural_score, y_true)
      + λ1 * MSE(neural_score, rule_score)   ← auxiliary alignment
      + λ2 * calibration_loss               ← differentiable soft ECE

* The BCE term trains the aggregator on ground-truth credibility labels.
* The MSE term anchors the neural output close to the rule-based score,
  preventing the network from drifting arbitrarily while still letting it
  improve on ambiguous cases.
* The calibration term (soft Expected Calibration Error) minimises the
  gap between predicted probability and observed accuracy per confidence
  bin — producing well-calibrated credibility estimates.

Regularisation (spec §8.3)
---------------------------
* Dropout is built into the aggregator modules.
* Weight decay is applied via the optimizer (AdamW default).
* Early stopping on validation AUC (``patience`` epochs).

Usage
-----
    from src.aggregation.neural_aggregator import NeuralAggregator
    from src.aggregation.aggregator_trainer import AggregatorTrainer, AggregatorDataset

    agg   = NeuralAggregator.build(config, input_dim=47)
    train = AggregatorDataset(X_train, y_train, rule_scores_train)
    val   = AggregatorDataset(X_val, y_val, rule_scores_val)

    trainer = AggregatorTrainer(agg, lambda1=0.1, lambda2=0.05)
    history = trainer.fit(
        DataLoader(train, batch_size=32, shuffle=True),
        DataLoader(val,   batch_size=64),
        epochs=20, patience=4,
    )
    NeuralAggregator.save(agg, "checkpoints/aggregator.pt", input_dim=47,
                          meta={"val_auc": history["best_val_auc"]})
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =========================================================
# DATASET
# =========================================================

class AggregatorDataset(Dataset):
    """Minimal dataset for AggregatorTrainer.

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Feature vectors from AggregatorFeatureBuilder.
    y : np.ndarray, shape (N,)
        Binary credibility labels (1 = credible, 0 = not credible).
    rule_scores : np.ndarray, shape (N,), optional
        Rule-based credibility scores from TruthLensScoreCalculator.
        Used for MSE auxiliary loss; if ``None``, that term is skipped.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rule_scores: Optional[np.ndarray] = None,
    ) -> None:
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.rule = (
            torch.from_numpy(np.asarray(rule_scores, dtype=np.float32))
            if rule_scores is not None
            else None
        )

        assert self.X.shape[0] == self.y.shape[0], \
            f"X and y length mismatch: {self.X.shape[0]} vs {self.y.shape[0]}"
        if self.rule is not None:
            assert self.rule.shape[0] == self.y.shape[0]

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "x": self.X[idx],
            "y": self.y[idx],
        }
        if self.rule is not None:
            item["rule"] = self.rule[idx]
        return item


# =========================================================
# TRAINER
# =========================================================

class AggregatorTrainer:
    """Training + evaluation loop for NeuralAggregator variants.

    Parameters
    ----------
    aggregator : nn.Module
        MLPAggregator or FeatureAttentionAggregator instance.
    optimizer : torch.optim.Optimizer, optional
        Defaults to AdamW with lr=1e-3, weight_decay=1e-4.
    lambda1 : float
        Weight for the MSE(neural, rule) auxiliary term (spec λ1).
    lambda2 : float
        Weight for the calibration loss term (spec λ2).
    n_cal_bins : int
        Number of bins for the soft ECE calibration loss.
    device : str
        Training device. Defaults to CUDA when available, else CPU.
    """

    def __init__(
        self,
        aggregator: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        lambda1: float = 0.10,
        lambda2: float = 0.05,
        n_cal_bins: int = 10,
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.aggregator = aggregator.to(self.device)

        self.optimizer = optimizer or torch.optim.AdamW(
            aggregator.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )

        if not (0.0 <= lambda1 <= 10.0):
            raise ValueError(f"lambda1 out of range: {lambda1}")
        if not (0.0 <= lambda2 <= 10.0):
            raise ValueError(f"lambda2 out of range: {lambda2}")

        self.lambda1   = float(lambda1)
        self.lambda2   = float(lambda2)
        self.n_cal_bins = int(n_cal_bins)

        logger.info(
            "AggregatorTrainer | device=%s λ1=%.3f λ2=%.3f bins=%d",
            self.device, lambda1, lambda2, n_cal_bins,
        )

    # -----------------------------------------------------------------------
    # LOSS
    # -----------------------------------------------------------------------

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        rule_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward + loss computation.

        Parameters
        ----------
        x : (B, D)
        y : (B,) binary labels in {0, 1}
        rule_scores : (B,), optional

        Returns
        -------
        dict with keys: ``total``, ``bce``, ``mse``, ``cal``
        """
        from src.aggregation.neural_aggregator import NeuralAggregatorOutput

        out: NeuralAggregatorOutput = self.aggregator(x)
        pred = out.credibility_score   # (B,)

        # ── BCE (main) ────────────────────────────────────────────────────
        L_bce = F.binary_cross_entropy(pred, y)

        # ── MSE alignment with rule scores (λ1) ───────────────────────────
        if rule_scores is not None and self.lambda1 > 0:
            L_mse = F.mse_loss(pred, rule_scores)
        else:
            L_mse = torch.tensor(0.0, device=self.device)

        # ── Calibration loss (λ2) — differentiable soft ECE ──────────────
        if self.lambda2 > 0:
            L_cal = self._soft_ece(pred, y)
        else:
            L_cal = torch.tensor(0.0, device=self.device)

        total = L_bce + self.lambda1 * L_mse + self.lambda2 * L_cal

        # PERF-AG-TRAINER: return `pred` so callers (evaluate) can reuse
        # the predictions without a second aggregator forward pass.
        return {"total": total, "bce": L_bce, "mse": L_mse, "cal": L_cal, "pred": pred}

    def _soft_ece(
        self, pred: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Differentiable soft Expected Calibration Error.

        Uses soft bin assignments via Gaussian kernels centred on bin
        midpoints so the loss is differentiable through ``pred``.

        Parameters
        ----------
        pred : (B,) predicted probabilities.
        y    : (B,) binary labels.
        """
        B = float(pred.numel())
        if B == 0:
            return torch.tensor(0.0, device=self.device)

        n = self.n_cal_bins
        centers = torch.linspace(
            1.0 / (2 * n), 1.0 - 1.0 / (2 * n), n,
            device=self.device,
        )
        sigma = 1.0 / (2.0 * n)

        # Soft membership: (B, n)
        diff = pred.unsqueeze(1) - centers.unsqueeze(0)       # (B, n)
        membership = torch.exp(-0.5 * (diff / sigma) ** 2)    # (B, n)
        bin_weights = membership.sum(dim=0).clamp_min(1e-9)    # (n,)

        # Weighted mean prediction and accuracy per bin
        bin_conf = (membership * pred.unsqueeze(1)).sum(dim=0) / bin_weights
        bin_acc  = (membership * y.unsqueeze(1)).sum(dim=0)   / bin_weights

        # ECE = Σ_b (|count_b| / N) * |acc_b − conf_b|
        ece = (bin_weights / B * (bin_acc - bin_conf).abs()).sum()
        return ece

    # -----------------------------------------------------------------------
    # TRAIN / EVAL
    # -----------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run one training epoch.

        Returns
        -------
        dict with averaged ``total``, ``bce``, ``mse``, ``cal`` losses.
        """
        self.aggregator.train()
        totals: Dict[str, float] = {"total": 0.0, "bce": 0.0, "mse": 0.0, "cal": 0.0}
        n_batches = 0

        for batch in dataloader:
            x    = batch["x"].to(self.device)
            y    = batch["y"].to(self.device)
            rule = batch.get("rule")
            if rule is not None:
                rule = rule.to(self.device)

            self.optimizer.zero_grad()
            losses = self.compute_loss(x, y, rule)
            losses["total"].backward()

            # Gradient clipping (spec §8.3 — regularisation)
            nn.utils.clip_grad_norm_(self.aggregator.parameters(), max_norm=1.0)
            self.optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

        if n_batches == 0:
            return totals
        return {k: v / n_batches for k, v in totals.items()}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a DataLoader.

        Returns
        -------
        dict with ``total``, ``bce``, ``mse``, ``cal``, ``auc`` keys.

        AUC is the area under the ROC curve, computed without sklearn
        via the trapezoidal rule over sorted predictions.
        """
        self.aggregator.eval()
        totals: Dict[str, float] = {"total": 0.0, "bce": 0.0, "mse": 0.0, "cal": 0.0}
        all_preds: list = []
        all_labels: list = []
        n_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                x    = batch["x"].to(self.device)
                y    = batch["y"].to(self.device)
                rule = batch.get("rule")
                if rule is not None:
                    rule = rule.to(self.device)

                losses = self.compute_loss(x, y, rule)
                for k in totals:
                    totals[k] += losses[k].item()

                # PERF-AG-TRAINER: reuse the predictions already computed
                # inside compute_loss (stored as "pred" in the return dict)
                # instead of running a second aggregator forward pass.
                all_preds.extend(losses["pred"].cpu().tolist())
                all_labels.extend(y.cpu().tolist())
                n_batches += 1

        if n_batches == 0:
            return {**totals, "auc": 0.5}

        metrics = {k: v / n_batches for k, v in totals.items()}
        metrics["auc"] = _compute_auc(all_preds, all_labels)
        return metrics

    # -----------------------------------------------------------------------
    # FIT LOOP
    # -----------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        epochs: int = 10,
        patience: int = 3,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        input_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping on validation AUC.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader, optional
            When provided, early stopping is applied on ``val_auc``.
        epochs : int
            Maximum number of training epochs.
        patience : int
            Early-stopping patience (stop after this many epochs without
            improvement in val AUC). Ignored when ``val_loader`` is None.
        scheduler : LR scheduler, optional
            Called with ``scheduler.step()`` after each epoch.
        checkpoint_path : str, optional
            If provided AND ``input_dim`` is given, saves the best-AUC
            checkpoint using ``NeuralAggregator.save``.
        input_dim : int, optional
            Required when ``checkpoint_path`` is set.

        Returns
        -------
        dict with ``history`` (list of per-epoch dicts) and ``best_val_auc``.
        """
        history: list = []
        best_auc = 0.0
        no_improve = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)

            epoch_log: Dict[str, Any] = {"epoch": epoch, "train": train_metrics}

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                epoch_log["val"] = val_metrics
                val_auc = val_metrics.get("auc", 0.0)

                logger.info(
                    "Epoch %d/%d | train_loss=%.4f bce=%.4f | "
                    "val_loss=%.4f val_auc=%.4f",
                    epoch, epochs,
                    train_metrics["total"],
                    train_metrics["bce"],
                    val_metrics["total"],
                    val_auc,
                )

                if val_auc > best_auc:
                    best_auc = val_auc
                    no_improve = 0
                    if checkpoint_path and input_dim:
                        from src.aggregation.neural_aggregator import NeuralAggregator
                        NeuralAggregator.save(
                            self.aggregator,
                            checkpoint_path,
                            input_dim=input_dim,
                            meta={"epoch": epoch, "val_auc": best_auc},
                        )
                else:
                    no_improve += 1
                    logger.info(
                        "No improvement (%d/%d) — best_auc=%.4f",
                        no_improve, patience, best_auc,
                    )
                    if no_improve >= patience:
                        logger.info(
                            "Early stopping at epoch %d (val_auc=%.4f)",
                            epoch, best_auc,
                        )
                        history.append(epoch_log)
                        break
            else:
                logger.info(
                    "Epoch %d/%d | train_loss=%.4f bce=%.4f mse=%.4f cal=%.4f",
                    epoch, epochs,
                    train_metrics["total"],
                    train_metrics["bce"],
                    train_metrics["mse"],
                    train_metrics["cal"],
                )

            history.append(epoch_log)

            if scheduler is not None:
                scheduler.step()

        return {"history": history, "best_val_auc": best_auc}


# =========================================================
# AUC (no sklearn dependency)
# =========================================================

def _compute_auc(
    preds: Iterable[float],
    labels: Iterable[float],
) -> float:
    """Trapezoid-rule AUC over sorted prediction scores.

    Returns 0.5 (random) when fewer than 2 unique label values are
    present.
    """
    pairs = sorted(zip(preds, labels), key=lambda x: x[0], reverse=True)

    pos = sum(1 for _, y in pairs if y > 0.5)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return 0.5

    tp = fp = 0
    auc = 0.0
    prev_fp = 0

    for _, label in pairs:
        if label > 0.5:
            tp += 1
        else:
            fp += 1
            # Accumulate area (trapezoid)
            auc += tp / pos * (1.0 / neg)

    return float(np.clip(auc, 0.0, 1.0))
