from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

from src.config.task_config import TASK_CONFIG, get_task_type

logger = logging.getLogger(__name__)

EPS = 1e-12

# Use a delimiter that won't collide with task names that contain underscores
# (e.g. ``narrative_frame``).
_DELIM = "::"


# =========================================================
# NORMALIZATION
# =========================================================

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / (df.std(ddof=0) + EPS)


def _winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    return df.clip(
        lower=df.quantile(lower),
        upper=df.quantile(upper),
        axis=1,
    )


# =========================================================
# TASK FEATURE EXTRACTION
# =========================================================

@lru_cache(maxsize=8)
def _resolve_task_type(task: str) -> Optional[str]:
    """Cache the per-task type lookup.

    Section 7: ``_extract_task_features`` calls this once per column, which
    for a 6-task / multilabel report adds up to dozens of dict lookups per
    correlation pass. Cap at 8 entries — the project has ~6 tasks, so we
    never evict in practice.
    """
    if task in TASK_CONFIG:
        return TASK_CONFIG[task]["type"]
    try:
        return get_task_type(task)
    except (KeyError, AttributeError):
        return None


def _extract_task_features(predictions: Dict[str, Any]) -> pd.DataFrame:
    features: Dict[str, np.ndarray] = {}

    for task, values in predictions.items():
        if isinstance(values, dict):
            # Allow PredictionCollector-style dicts
            values = values.get("y_proba", values.get("probabilities", values.get("y_pred")))

        arr = np.asarray(values)
        task_type = _resolve_task_type(task)

        if task_type == "binary":
            if arr.ndim == 2 and arr.shape[1] == 2:
                arr = arr[:, 1]
            features[task] = arr.reshape(-1)

        elif task_type == "multiclass":
            if arr.ndim == 1:
                features[task] = arr
            else:
                # HIGH E10: drop the last class column (K-1 dummy encoding).
                # All K columns of a softmax sum to 1, so they are perfectly
                # collinear — keeping all of them inflates mean correlation
                # and corrupts ``correlation_statistics``.
                n_keep = max(arr.shape[1] - 1, 1)
                for i in range(n_keep):
                    features[f"{task}{_DELIM}class_{i}"] = arr[:, i]

        elif task_type == "multilabel":
            if arr.ndim == 1:
                features[task] = arr
            else:
                # Multilabel sigmoids are independent per label, so no
                # collinearity adjustment is needed — keep every column.
                for i in range(arr.shape[1]):
                    features[f"{task}{_DELIM}label_{i}"] = arr[:, i]

        else:
            # Unknown task — flatten as best-effort
            if arr.ndim == 1:
                features[task] = arr
            elif arr.ndim == 2:
                for i in range(arr.shape[1]):
                    features[f"{task}{_DELIM}c{i}"] = arr[:, i]

    return pd.DataFrame(features)


# =========================================================
# MAIN CORRELATION
# =========================================================

def compute_task_correlation(
    predictions: Dict[str, Any] | pd.DataFrame,
    *,
    normalize: bool = True,
    method: Literal["pearson", "spearman"] = "spearman",
    robust: bool = True,
    confidence: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    graph_signal: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    logger.info("[CORRELATION] computing (method=%s)", method)

    if isinstance(predictions, dict):
        df = _extract_task_features(predictions)
    else:
        df = predictions.copy()

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise RuntimeError(
            "Insufficient numeric data to compute task correlation"
        )

    if robust:
        df = _winsorize(df)

    if normalize:
        df = _normalize(df)

    if confidence is not None:
        df["global_confidence"] = np.asarray(confidence).reshape(-1)[: len(df)]

    if uncertainty is not None:
        df["global_uncertainty"] = np.asarray(uncertainty).reshape(-1)[: len(df)]

    if graph_signal is not None:
        df["graph_signal"] = np.asarray(graph_signal).reshape(-1)[: len(df)]

    corr = df.corr(method=method)
    corr = corr.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return corr


# =========================================================
# AGGREGATION (UNWEIGHTED MEAN)
# =========================================================

def _column_to_task(col: str) -> str:
    return col.split(_DELIM, 1)[0]


def aggregate_task_correlation(corr: pd.DataFrame) -> pd.DataFrame:
    task_map: Dict[str, list[str]] = {}
    for col in corr.columns:
        task_map.setdefault(_column_to_task(col), []).append(col)

    tasks = list(task_map.keys())
    agg = pd.DataFrame(0.0, index=tasks, columns=tasks)

    for t1 in tasks:
        for t2 in tasks:
            if t1 == t2:
                agg.loc[t1, t2] = 1.0
                continue

            block = corr.loc[task_map[t1], task_map[t2]].values.flatten()
            block = block[np.isfinite(block)]
            if block.size == 0:
                agg.loc[t1, t2] = 0.0
            else:
                agg.loc[t1, t2] = float(np.mean(block))

    return agg.astype(float)


# =========================================================
# MONITORING SIGNALS
# =========================================================

def correlation_statistics(corr: pd.DataFrame) -> Dict[str, float]:
    values = corr.values.flatten()
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            "mean_correlation": 0.0,
            "std_correlation": 0.0,
            "max_correlation": 0.0,
            "min_correlation": 0.0,
            "high_correlation_ratio": 0.0,
        }

    return {
        "mean_correlation": float(np.mean(values)),
        "std_correlation": float(np.std(values)),
        "max_correlation": float(np.max(values)),
        "min_correlation": float(np.min(values)),
        "high_correlation_ratio": float(np.mean(np.abs(values) > 0.8)),
    }


# =========================================================
# SAVE
# =========================================================

def save_correlation_matrix(corr: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(path)
    logger.info("Saved correlation matrix: %s", path)
    return path


__all__ = [
    "aggregate_task_correlation",
    "compute_task_correlation",
    "correlation_statistics",
    "save_correlation_matrix",
]
