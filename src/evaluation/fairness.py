"""Group-fairness metrics for evaluation reports.

Provides simple, dependency-free implementations of the most common fairness
metrics so the evaluator can surface disparate-impact signals when sensitive
attributes are available in the dataset.

All functions accept array-like inputs and return floats / dictionaries that
serialize cleanly into JSON reports.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# =========================================================
# UTILITIES
# =========================================================

def _validate(y_true, y_pred, groups) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(y_pred).reshape(-1)
    g = np.asarray(groups).reshape(-1)

    if not (y.shape == p.shape == g.shape):
        raise ValueError(
            f"Shape mismatch: y_true {y.shape}, y_pred {p.shape}, groups {g.shape}"
        )
    if y.size == 0:
        raise ValueError("inputs cannot be empty")
    return y, p, g


# =========================================================
# PER-GROUP METRICS
# =========================================================

def per_group_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    groups: Iterable,
    *,
    positive_label: int = 1,
    task_type: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-group classification metrics.

    Section 10: accept an explicit ``task_type`` and drop the ``{0, 1}``
    heuristic. The previous code routed any group whose labels happened to
    fall in ``{0, 1}`` through binary averaging — so a 3-class slice that
    happened to contain only labels 0 and 1 silently flipped its average
    rule. Callers that know the task should always pass ``task_type``.
    """
    y, p, g = _validate(y_true, y_pred, groups)
    unique_groups = np.unique(g)

    out: Dict[str, Dict[str, float]] = {}
    for group in unique_groups:
        mask = g == group
        if not mask.any():
            continue

        y_g = y[mask]
        p_g = p[mask]

        # Section 10: prefer the authoritative task_type; only fall back to
        # the legacy {0, 1} heuristic when no caller hint is provided.
        if task_type == "binary":
            use_binary = True
        elif task_type in ("multiclass", "multilabel"):
            use_binary = False
        else:
            use_binary = set(np.unique(y_g)).issubset({0, 1})

        if use_binary:
            prec = precision_score(y_g, p_g, average="binary", pos_label=positive_label, zero_division=0)
            rec = recall_score(y_g, p_g, average="binary", pos_label=positive_label, zero_division=0)
            f1v = f1_score(y_g, p_g, average="binary", pos_label=positive_label, zero_division=0)
        else:
            prec = precision_score(y_g, p_g, average="macro", zero_division=0)
            rec = recall_score(y_g, p_g, average="macro", zero_division=0)
            f1v = f1_score(y_g, p_g, average="macro", zero_division=0)

        out[str(group)] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y_g, p_g)),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1v),
            "positive_rate": float(np.mean(p_g == positive_label)),
        }

    return out


# =========================================================
# DEMOGRAPHIC PARITY
# =========================================================

def demographic_parity(
    y_pred: Iterable,
    groups: Iterable,
    *,
    positive_label: int = 1,
) -> Dict[str, float]:
    p = np.asarray(y_pred).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    if p.shape != g.shape:
        raise ValueError("y_pred and groups must have the same shape")

    unique_groups = np.unique(g)
    rates = {
        str(group): float(np.mean(p[g == group] == positive_label))
        for group in unique_groups
        if (g == group).any()
    }

    if not rates:
        return {"max_diff": 0.0, "ratio": 1.0, "rates": {}}

    values = list(rates.values())
    max_diff = float(max(values) - min(values))
    ratio = float(min(values) / max(values)) if max(values) > 0 else 1.0

    return {"rates": rates, "max_diff": max_diff, "ratio": ratio}


# =========================================================
# EQUAL OPPORTUNITY (TPR PARITY) + EQUALIZED ODDS
# =========================================================

def _per_group_rates(y, p, g, positive_label) -> Dict[str, Dict[str, float]]:
    rates: Dict[str, Dict[str, float]] = {}
    for group in np.unique(g):
        mask = g == group
        if not mask.any():
            continue

        y_g = y[mask]
        p_g = p[mask]
        pos = y_g == positive_label
        neg = ~pos

        tpr = float(np.mean(p_g[pos] == positive_label)) if pos.any() else float("nan")
        fpr = float(np.mean(p_g[neg] == positive_label)) if neg.any() else float("nan")
        rates[str(group)] = {"tpr": tpr, "fpr": fpr}
    return rates


def equal_opportunity(
    y_true: Iterable,
    y_pred: Iterable,
    groups: Iterable,
    *,
    positive_label: int = 1,
) -> Dict[str, float]:
    y, p, g = _validate(y_true, y_pred, groups)
    rates = _per_group_rates(y, p, g, positive_label)

    valid = [r["tpr"] for r in rates.values() if not np.isnan(r["tpr"])]
    if not valid:
        return {"per_group_tpr": rates, "max_diff": 0.0}

    return {"per_group_tpr": rates, "max_diff": float(max(valid) - min(valid))}


def equalized_odds(
    y_true: Iterable,
    y_pred: Iterable,
    groups: Iterable,
    *,
    positive_label: int = 1,
) -> Dict[str, Any]:
    y, p, g = _validate(y_true, y_pred, groups)
    rates = _per_group_rates(y, p, g, positive_label)

    tprs = [r["tpr"] for r in rates.values() if not np.isnan(r["tpr"])]
    fprs = [r["fpr"] for r in rates.values() if not np.isnan(r["fpr"])]

    return {
        "per_group": rates,
        "tpr_max_diff": float(max(tprs) - min(tprs)) if tprs else 0.0,
        "fpr_max_diff": float(max(fprs) - min(fprs)) if fprs else 0.0,
    }


# =========================================================
# TOP-LEVEL ENTRY
# =========================================================

def fairness_report(
    y_true: Iterable,
    y_pred: Iterable,
    groups: Iterable,
    *,
    positive_label: int = 1,
    group_name: Optional[str] = None,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a full fairness report for a single sensitive attribute."""
    try:
        y, p, g = _validate(y_true, y_pred, groups)
    except ValueError as exc:
        logger.warning("fairness_report aborted: %s", exc)
        return {"error": str(exc)}

    return {
        "attribute": group_name,
        "task_type": task_type,
        # Section 10: forward ``task_type`` so per_group_metrics doesn't fall
        # back to the {0, 1} heuristic that misroutes pruned multiclass slices.
        "per_group_metrics": per_group_metrics(
            y, p, g, positive_label=positive_label, task_type=task_type
        ),
        "demographic_parity": demographic_parity(p, g, positive_label=positive_label),
        "equal_opportunity": equal_opportunity(y, p, g, positive_label=positive_label),
        "equalized_odds": equalized_odds(y, p, g, positive_label=positive_label),
    }


def fairness_report_multi(
    y_true: Iterable,
    y_pred: Iterable,
    sensitive_attributes: Dict[str, Iterable],
    *,
    positive_label: int = 1,
    task_type: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run :func:`fairness_report` for each sensitive attribute provided."""
    return {
        name: fairness_report(
            y_true=y_true,
            y_pred=y_pred,
            groups=values,
            positive_label=positive_label,
            group_name=name,
            task_type=task_type,
        )
        for name, values in sensitive_attributes.items()
    }


# =========================================================
# BOOTSTRAP CI HELPER — Section 10
# =========================================================

def bootstrap_metric_ci(
    y_true: Iterable,
    y_pred: Iterable,
    metric_fn,
    *,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval for a single scalar metric.

    Section 10: fairness gaps reported as point estimates (e.g. "TPR diff =
    0.04") were being treated as deterministic by downstream alerting. This
    helper resamples ``y_true`` / ``y_pred`` indices with replacement and
    returns the lower / upper bounds at the requested ``alpha`` so the
    pipeline can emit "0.04 [-0.01, 0.09]" style intervals instead.
    """
    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    if y.shape[0] != p.shape[0] or y.shape[0] == 0:
        raise ValueError("y_true and y_pred must be non-empty and same length")

    rng = np.random.default_rng(seed)
    n = y.shape[0]
    samples: List[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        try:
            value = float(metric_fn(y[idx], p[idx]))
        except Exception:
            continue
        if np.isfinite(value):
            samples.append(value)

    if not samples:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}

    arr = np.asarray(samples, dtype=float)
    low = float(np.quantile(arr, alpha / 2.0))
    high = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return {
        "point": float(np.mean(arr)),
        "low": low,
        "high": high,
        "n": int(arr.size),
    }


__all__ = [
    "bootstrap_metric_ci",
    "demographic_parity",
    "equal_opportunity",
    "equalized_odds",
    "fairness_report",
    "fairness_report_multi",
    "per_group_metrics",
]
