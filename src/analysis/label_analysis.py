from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class LabelAnalysisConfig:
    """
    Controls strictness and thresholds.
    """
    imbalance_threshold: float = 0.90   # warn if any class > 90%
    min_samples_per_class: int = 10     # warn if any class < 10
    require_all_classes: bool = False   # fail if some classes missing
    top_k_preview: int = 10             # for logging top classes


# =========================================================
# RESULTS
# =========================================================

@dataclass(frozen=True)
class ClassDistribution:
    counts: Dict[Any, int]
    ratios: Dict[Any, float]
    total: int


@dataclass(frozen=True)
class LabelAnalysisResult:
    per_task: Dict[str, ClassDistribution]
    imbalance_flags: Dict[str, bool]
    rare_class_flags: Dict[str, List[Any]]
    missing_classes: Dict[str, List[Any]]
    summary: Dict[str, Any]


# =========================================================
# CORE UTILITIES
# =========================================================

def _value_counts(series: pd.Series) -> ClassDistribution:
    vc = series.value_counts(dropna=False)
    total = int(vc.sum())

    counts = {k: int(v) for k, v in vc.to_dict().items()}
    ratios = {k: float(v) / max(total, 1) for k, v in counts.items()}

    return ClassDistribution(counts=counts, ratios=ratios, total=total)


def _is_multilabel_column(series: pd.Series) -> bool:
    """
    Heuristic:
    - list/tuple/set per row OR
    - string with delimiters (e.g., "a|b|c")
    """
    if series.empty:
        return False

    sample = series.iloc[0]
    if isinstance(sample, (list, tuple, set)):
        return True
    if isinstance(sample, str) and ("|" in sample or "," in sample):
        return True
    return False


def _explode_multilabel(series: pd.Series) -> pd.Series:
    """
    Convert multi-label column into a flat series of labels.
    """
    if series.empty:
        return series

    if isinstance(series.iloc[0], (list, tuple, set)):
        return series.explode()

    # string case: "a|b|c" or "a,b"
    return series.astype(str).str.replace(",", "|").str.split("|").explode()


# =========================================================
# ANALYSIS
# =========================================================

def analyze_labels(
    df: pd.DataFrame,
    task_columns: Dict[str, str],
    *,
    config: Optional[LabelAnalysisConfig] = None,
) -> LabelAnalysisResult:
    """
    Analyze label distributions for each task.

    Args:
        df: input dataframe
        task_columns: mapping {task_name: column_name}
        config: analysis config

    Returns:
        LabelAnalysisResult
    """
    config = config or LabelAnalysisConfig()

    per_task: Dict[str, ClassDistribution] = {}
    imbalance_flags: Dict[str, bool] = {}
    rare_flags: Dict[str, List[Any]] = {}
    missing_classes: Dict[str, List[Any]] = {}

    for task, col in task_columns.items():

        if col not in df.columns:
            raise ValueError(f"Missing column for task '{task}': {col}")

        series = df[col].dropna()

        # -------------------------
        # Handle multilabel
        # -------------------------
        if _is_multilabel_column(series):
            series = _explode_multilabel(series).dropna()

        dist = _value_counts(series)
        per_task[task] = dist

        # -------------------------
        # Imbalance detection
        # -------------------------
        max_ratio = max(dist.ratios.values()) if dist.ratios else 0.0
        imbalance = max_ratio >= config.imbalance_threshold
        imbalance_flags[task] = imbalance

        # -------------------------
        # Rare classes
        # -------------------------
        rare = [
            cls for cls, cnt in dist.counts.items()
            if cnt < config.min_samples_per_class
        ]
        rare_flags[task] = rare

        # -------------------------
        # Missing classes (optional)
        # -------------------------
        if config.require_all_classes:
            # only meaningful if expected classes known externally
            missing_classes[task] = []  # placeholder
        else:
            missing_classes[task] = []

        # -------------------------
        # Logging (concise)
        # -------------------------
        _log_task_summary(task, dist, config, imbalance, rare)

    summary = _build_summary(per_task, imbalance_flags, rare_flags)

    return LabelAnalysisResult(
        per_task=per_task,
        imbalance_flags=imbalance_flags,
        rare_class_flags=rare_flags,
        missing_classes=missing_classes,
        summary=summary,
    )


# =========================================================
# LOGGING
# =========================================================

def _log_task_summary(
    task: str,
    dist: ClassDistribution,
    config: LabelAnalysisConfig,
    imbalance: bool,
    rare: List[Any],
) -> None:
    if dist.total == 0:
        logger.warning("Task '%s': no labels found", task)
        return

    # top-k classes by count
    top_items = sorted(
        dist.counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[: config.top_k_preview]

    logger.info(
        "LabelDist | task=%s | total=%d | classes=%d",
        task,
        dist.total,
        len(dist.counts),
    )

    logger.debug("Top classes (%s): %s", task, top_items)

    if imbalance:
        logger.warning(
            "Imbalance detected | task=%s | max_ratio=%.3f",
            task,
            max(dist.ratios.values()) if dist.ratios else 0.0,
        )

    if rare:
        logger.warning(
            "Rare classes | task=%s | count=%d | classes=%s",
            task,
            len(rare),
            rare[:10],
        )


# =========================================================
# SUMMARY
# =========================================================

def _build_summary(
    per_task: Dict[str, ClassDistribution],
    imbalance_flags: Dict[str, bool],
    rare_flags: Dict[str, List[Any]],
) -> Dict[str, Any]:

    total_tasks = len(per_task)
    imbalanced_tasks = [t for t, f in imbalance_flags.items() if f]
    tasks_with_rare = [t for t, v in rare_flags.items() if v]

    return {
        "num_tasks": total_tasks,
        "imbalanced_tasks": imbalanced_tasks,
        "tasks_with_rare_classes": tasks_with_rare,
    }


# =========================================================
# OPTIONAL HARD ENFORCEMENT
# =========================================================

def assert_label_health(
    result: LabelAnalysisResult,
    *,
    fail_on_imbalance: bool = False,
    fail_on_rare: bool = False,
) -> None:
    """
    Optional hard guard before training.
    """

    if fail_on_imbalance:
        bad = [t for t, f in result.imbalance_flags.items() if f]
        if bad:
            raise RuntimeError(f"Imbalanced tasks detected: {bad}")

    if fail_on_rare:
        bad = [t for t, v in result.rare_class_flags.items() if v]
        if bad:
            raise RuntimeError(f"Rare classes detected in tasks: {bad}")