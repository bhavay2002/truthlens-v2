#src\utils\input_validation.py

from __future__ import annotations

import logging
from typing import Iterable, Sequence, Any, Dict, Optional

import pandas as pd
import numpy as np
import torch

logger = logging.getLogger(__name__)


# =========================================================
# GLOBAL VALIDATION SWITCH
# =========================================================

VALIDATION_ENABLED = True


def _skip_validation():
    return not VALIDATION_ENABLED


# =========================================================
# DATAFRAME SCHEMA VALIDATION
# =========================================================

def ensure_dataframe(
    df: pd.DataFrame,
    *,
    name: str = "df",
    required_columns: Iterable[str] = (),
    min_rows: int = 1,
    dtypes: Optional[Dict[str, str]] = None,
) -> None:

    if _skip_validation():
        return

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be DataFrame")

    if len(df) < min_rows:
        raise ValueError(f"{name} must have >= {min_rows} rows")

    # -----------------------------
    # Required columns
    # -----------------------------
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

    # -----------------------------
    # DTYPE VALIDATION (NEW)
    # -----------------------------
    if dtypes:
        for col, expected_dtype in dtypes.items():
            if col not in df.columns:
                continue

            actual = str(df[col].dtype)

            if expected_dtype not in actual:
                raise TypeError(
                    f"{name}.{col} expected dtype {expected_dtype}, got {actual}"
                )


# =========================================================
# LABEL VALIDATION (UPGRADED)
# =========================================================

def validate_labels(
    labels: Any,
    task_type: str,
    num_labels: int,
) -> None:

    if _skip_validation():
        return

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    labels = np.asarray(labels)

    # -----------------------------
    # BINARY
    # -----------------------------
    if task_type == "binary":
        unique = np.unique(labels)
        if not set(unique).issubset({0, 1}):
            raise ValueError(f"Binary labels must be 0/1, got {unique}")

    # -----------------------------
    # MULTICLASS
    # -----------------------------
    elif task_type == "multiclass":
        if labels.ndim != 1:
            raise ValueError("Multiclass labels must be 1D")

        if labels.min() < 0 or labels.max() >= num_labels:
            raise ValueError(
                f"Labels must be in [0, {num_labels-1}]"
            )

    # -----------------------------
    # MULTILABEL
    # -----------------------------
    elif task_type == "multilabel":
        if labels.ndim != 2:
            raise ValueError("Multilabel must be [B, num_labels]")

        if labels.shape[1] != num_labels:
            raise ValueError("Incorrect label dimension")

        unique = np.unique(labels)
        if not set(unique).issubset({0, 1}):
            raise ValueError("Multilabel must be binary (0/1)")

    else:
        raise ValueError(f"Unknown task_type: {task_type}")


# =========================================================
# TENSOR VALIDATION (ENHANCED)
# =========================================================

def validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_dim: int | None = None,
    dtype: torch.dtype | None = None,
) -> None:

    if _skip_validation():
        return

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be tensor")

    if expected_dim and tensor.dim() != expected_dim:
        raise ValueError(f"{name} must have dim={expected_dim}")

    if dtype and tensor.dtype != dtype:
        raise TypeError(f"{name} expected dtype {dtype}, got {tensor.dtype}")

    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN")

    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf")


# =========================================================
# BATCH VALIDATION (ENHANCED)
# =========================================================

def validate_batch(batch: Dict[str, Any]) -> None:

    if _skip_validation():
        return

    required = {"input_ids", "attention_mask", "labels", "task"}

    missing = required - set(batch.keys())
    if missing:
        raise ValueError(f"Batch missing keys: {missing}")

    validate_tensor(batch["input_ids"], name="input_ids", expected_dim=2)
    validate_tensor(batch["attention_mask"], name="attention_mask", expected_dim=2)

    if not isinstance(batch["task"], str):
        raise TypeError("task must be string")


# =========================================================
# TEXT VALIDATION
# =========================================================

def ensure_non_empty_text(text: str, *, name: str = "text") -> str:

    if _skip_validation():
        return text

    if not isinstance(text, str):
        raise TypeError(f"{name} must be string")

    if not text.strip():
        raise ValueError(f"{name} cannot be empty")

    return text


def ensure_non_empty_text_list(
    texts: Sequence[str] | Iterable[str],
    *,
    name: str = "texts",
) -> list[str]:

    if _skip_validation():
        return list(texts)

    if texts is None:
        raise ValueError(f"{name} cannot be None")

    if isinstance(texts, (str, bytes)):
        texts = [texts]

    normalized = []

    for t in texts:
        if t is None:
            normalized.append("")
            continue

        if isinstance(t, (bytes, bytearray)):
            normalized.append(t.decode("utf-8", errors="ignore"))
            continue

        normalized.append(str(t))

    if not normalized or all(not t.strip() for t in normalized):
        raise ValueError(f"{name} cannot be empty")

    return normalized


def ensure_non_empty_text_column(
    df: pd.DataFrame,
    column: str,
    *,
    name: str = "df",
) -> pd.DataFrame:
    """Validate that ``column`` exists in ``df`` and contains non-empty text."""
    if _skip_validation():
        return df

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be DataFrame")

    if column not in df.columns:
        raise ValueError(f"{name} missing column: {column!r}")

    series = df[column].astype(str).str.strip()
    if series.eq("").all():
        raise ValueError(f"{name}.{column} cannot be entirely empty")

    return df


def ensure_positive_int(value: Any, *, name: str = "value") -> int:
    """Coerce ``value`` to a positive ``int`` or raise ``ValueError``."""
    if _skip_validation() and isinstance(value, int):
        return value

    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from exc

    if ivalue <= 0:
        raise ValueError(f"{name} must be > 0, got {ivalue}")

    return ivalue