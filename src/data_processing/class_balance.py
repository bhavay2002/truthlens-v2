"""
Class-balance analysis driven by data contracts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import pandas as pd

from src.data_processing.data_contracts import get_contract

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class ClassBalanceConfig:
    imbalance_threshold: float = 0.2
    compute_weights: bool = True
    normalize_weights: bool = True


# =========================================================
# RESULT
# =========================================================

@dataclass
class ClassBalanceReport:
    task: str
    type: str  # classification | multilabel
    distribution: Dict[str, Any]
    imbalance_detected: bool
    weights: Optional[Dict[Any, float]] = None


# =========================================================
# CLASSIFICATION
# =========================================================

def analyze_classification(
    df: pd.DataFrame,
    label_col: str,
    *,
    config: Optional[ClassBalanceConfig] = None,
) -> ClassBalanceReport:
    config = config or ClassBalanceConfig()

    counts = df[label_col].value_counts().sort_index()
    total = counts.sum()
    dist = (counts / total).to_dict() if total > 0 else {}

    min_ratio = min(dist.values()) if dist else 0.0
    imbalance = min_ratio < config.imbalance_threshold

    weights = (
        _compute_class_weights(counts, normalize=config.normalize_weights)
        if config.compute_weights
        else None
    )

    logger.info(
        "Class balance | %s | imbalance=%s | min_ratio=%.3f",
        label_col, imbalance, min_ratio,
    )

    return ClassBalanceReport(
        task=label_col,
        type="classification",
        distribution=dist,
        imbalance_detected=imbalance,
        weights=weights,
    )


# =========================================================
# MULTILABEL
# =========================================================

def analyze_multilabel(
    df: pd.DataFrame,
    label_cols: List[str],
    *,
    config: Optional[ClassBalanceConfig] = None,
) -> ClassBalanceReport:
    config = config or ClassBalanceConfig()

    dist: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    imbalance = False

    total = max(len(df), 1)
    for col in label_cols:
        pos = float(df[col].sum())
        ratio = pos / total
        dist[col] = ratio
        if ratio < config.imbalance_threshold:
            imbalance = True
        if config.compute_weights:
            weights[col] = _compute_binary_weight(pos, total)

    logger.info(
        "Multilabel balance | cols=%d | imbalance=%s",
        len(label_cols), imbalance,
    )

    return ClassBalanceReport(
        task="multilabel",
        type="multilabel",
        distribution=dist,
        imbalance_detected=imbalance,
        weights=weights if config.compute_weights else None,
    )


# =========================================================
# CONTRACT-DRIVEN WRAPPER
# =========================================================

def analyze_task_balance(
    df: pd.DataFrame,
    task: str,
    *,
    config: Optional[ClassBalanceConfig] = None,
) -> ClassBalanceReport:
    contract = get_contract(task)
    if contract.task_type == "classification":
        return analyze_classification(df, contract.label_columns[0], config=config)
    if contract.task_type == "multilabel":
        return analyze_multilabel(df, list(contract.label_columns), config=config)
    raise ValueError(f"Unsupported task type: {contract.task_type}")


# =========================================================
# WEIGHT UTILS
# =========================================================

def _compute_class_weights(counts, normalize: bool = True):
    total = counts.sum()
    weights = {cls: total / c for cls, c in counts.items()}
    if normalize:
        s = sum(weights.values())
        weights = {k: v / s for k, v in weights.items()}
    return weights


def _compute_binary_weight(pos: float, total: float) -> float:
    neg = total - pos
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)
