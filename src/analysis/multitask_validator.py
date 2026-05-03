from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class MultiTaskValidationConfig:
    """
    Controls strictness and behavior.
    """
    # if True, drop rows with no valid labels across all tasks
    drop_rows_with_no_labels: bool = True

    # if True, enforce that every task column exists
    require_all_task_columns: bool = True

    # if True, fail on invalid label values
    strict_label_values: bool = False

    # if True, coerce multilabel strings ("a|b") -> list[str]
    normalize_multilabel: bool = True

    # delimiter for multilabel strings
    multilabel_delimiter: str = "|"

    # optional whitelist per task: {task: [allowed_labels]}
    allowed_labels: Optional[Dict[str, List[Any]]] = None

    # minimum number of tasks with labels per row
    min_tasks_with_labels: int = 1


# =========================================================
# RESULT
# =========================================================

@dataclass(frozen=True)
class MultiTaskValidationResult:
    num_rows_before: int
    num_rows_after: int
    rows_dropped: int

    missing_columns: List[str]
    invalid_value_counts: Dict[str, int]
    rows_with_no_labels: int

    task_coverage: Dict[str, int]  # rows with at least one label per task
    notes: Dict[str, Any]


# =========================================================
# HELPERS
# =========================================================

def _is_multilabel(series: pd.Series) -> bool:
    if series.empty:
        return False
    v = series.iloc[0]
    if isinstance(v, (list, tuple, set)):
        return True
    if isinstance(v, str) and ("|" in v or "," in v):
        return True
    return False


def _normalize_multilabel_series(
    s: pd.Series,
    delimiter: str,
) -> pd.Series:
    """
    Ensure multilabel values are List[str].
    """
    if s.empty:
        return s

    if isinstance(s.iloc[0], (list, tuple, set)):
        return s.apply(lambda x: list(x) if x is not None else [])

    # string case: "a|b|c" or "a,b"
    s = s.astype(str).str.replace(",", delimiter)
    return s.apply(lambda x: [t for t in x.split(delimiter) if t] if x and x != "nan" else [])


def _has_label(value: Any) -> bool:
    """
    Check if a cell has a valid (non-empty) label.
    """
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    if isinstance(value, str):
        return value.strip() != ""
    return True


# =========================================================
# CORE
# =========================================================

def validate_multitask_dataframe(
    df: pd.DataFrame,
    task_columns: Dict[str, str],
    *,
    config: Optional[MultiTaskValidationConfig] = None,
) -> Tuple[pd.DataFrame, MultiTaskValidationResult]:
    """
    Validate and optionally clean a multi-task dataset.

    Args:
        df: input dataframe
        task_columns: {task_name: column_name}
        config: behavior controls

    Returns:
        (clean_df, result)
    """
    config = config or MultiTaskValidationConfig()

    df = df.copy()
    n_before = len(df)

    missing_cols: List[str] = []
    invalid_value_counts: Dict[str, int] = {}
    task_coverage: Dict[str, int] = {}

    # -----------------------------------------------------
    # 1. COLUMN CHECK
    # -----------------------------------------------------
    for task, col in task_columns.items():
        if col not in df.columns:
            missing_cols.append(col)

    if missing_cols:
        msg = f"Missing task columns: {missing_cols}"
        if config.require_all_task_columns:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    # -----------------------------------------------------
    # 2. NORMALIZATION (MULTILABEL)
    # -----------------------------------------------------
    for task, col in task_columns.items():
        if col not in df.columns:
            continue

        s = df[col]

        if config.normalize_multilabel and _is_multilabel(s):
            df[col] = _normalize_multilabel_series(
                s, config.multilabel_delimiter
            )

    # -----------------------------------------------------
    # 3. INVALID VALUE CHECK
    # -----------------------------------------------------
    for task, col in task_columns.items():
        if col not in df.columns:
            continue

        allowed = None
        if config.allowed_labels and task in config.allowed_labels:
            allowed = set(config.allowed_labels[task])

        invalid_count = 0

        if allowed is not None:
            def _is_invalid(v: Any) -> bool:
                if not _has_label(v):
                    return False
                if isinstance(v, list):
                    return any(item not in allowed for item in v)
                return v not in allowed

            mask_invalid = df[col].apply(_is_invalid)
            invalid_count = int(mask_invalid.sum())

            if invalid_count > 0:
                msg = f"Invalid labels in task '{task}': {invalid_count}"
                if config.strict_label_values:
                    raise ValueError(msg)
                else:
                    logger.warning(msg)

        invalid_value_counts[task] = invalid_count

    # -----------------------------------------------------
    # 4. TASK COVERAGE
    # -----------------------------------------------------
    label_presence = {}

    for task, col in task_columns.items():
        if col not in df.columns:
            task_coverage[task] = 0
            continue

        has_lbl = df[col].apply(_has_label)
        task_coverage[task] = int(has_lbl.sum())
        label_presence[task] = has_lbl

    # -----------------------------------------------------
    # 5. ROW-LEVEL VALIDATION
    # -----------------------------------------------------
    # count how many tasks have labels per row
    per_row_label_counts = pd.DataFrame(label_presence).sum(axis=1)

    rows_with_no_labels = int((per_row_label_counts == 0).sum())

    # filter condition
    if config.drop_rows_with_no_labels:
        keep_mask = per_row_label_counts >= config.min_tasks_with_labels
        df = df.loc[keep_mask].reset_index(drop=True)

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------
    n_after = len(df)

    result = MultiTaskValidationResult(
        num_rows_before=n_before,
        num_rows_after=n_after,
        rows_dropped=n_before - n_after,
        missing_columns=missing_cols,
        invalid_value_counts=invalid_value_counts,
        rows_with_no_labels=rows_with_no_labels,
        task_coverage=task_coverage,
        notes={
            "min_tasks_with_labels": config.min_tasks_with_labels,
            "normalize_multilabel": config.normalize_multilabel,
        },
    )

    _log_summary(result)

    return df, result


# =========================================================
# LOGGING
# =========================================================

def _log_summary(res: MultiTaskValidationResult) -> None:
    logger.info(
        "MultiTaskValidation | rows_before=%d | rows_after=%d | dropped=%d",
        res.num_rows_before,
        res.num_rows_after,
        res.rows_dropped,
    )

    if res.missing_columns:
        logger.warning("Missing columns: %s", res.missing_columns)

    for task, cnt in res.invalid_value_counts.items():
        if cnt > 0:
            logger.warning("Invalid values | task=%s | count=%d", task, cnt)

    logger.info("Rows with no labels (pre-filter): %d", res.rows_with_no_labels)

    logger.info("Task coverage: %s", res.task_coverage)


# =========================================================
# HARD GUARDS (OPTIONAL)
# =========================================================

def assert_multitask_health(
    result: MultiTaskValidationResult,
    *,
    max_drop_ratio: float = 0.5,
) -> None:
    """
    Optional hard stop if too many rows are dropped.
    """
    if result.num_rows_before == 0:
        raise RuntimeError("Empty dataset")

    drop_ratio = result.rows_dropped / max(result.num_rows_before, 1)

    if drop_ratio > max_drop_ratio:
        raise RuntimeError(
            f"Too many rows dropped during validation: {drop_ratio:.2%}"
        )