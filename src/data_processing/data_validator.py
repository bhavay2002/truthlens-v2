"""
Schema validator for the 6 TruthLens tasks.

The ``TASK_SCHEMAS`` table is *derived* from ``data_contracts.CONTRACTS``
so the validator can never disagree with the dataset factory or cleaning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import pandas as pd

from src.data_processing.data_contracts import (
    CONTRACTS,
    get_contract,
    is_classification,
    is_multilabel,
)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class DataValidatorConfig:
    strict: bool = True
    check_text: bool = True
    min_text_len: int = 3
    max_text_len: int = 10000
    enforce_label_range: bool = True
    enforce_binary_multilabel: bool = True
    sample_errors: int = 5


# =========================================================
# RESULT
# =========================================================

@dataclass
class ValidationReport:
    rows: int
    columns: int
    missing_columns: List[str] = field(default_factory=list)
    invalid_text_rows: int = 0
    invalid_label_rows: Dict[str, int] = field(default_factory=dict)
    label_value_violations: Dict[str, int] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# SCHEMAS (DERIVED FROM CONTRACTS — single source of truth)
# =========================================================

# Classification range per task (inclusive). Multilabel is always [0, 1].
_CLASSIFICATION_RANGES: Dict[str, tuple] = {
    "bias": (0, 1),
    "ideology": (0, 2),
    "propaganda": (0, 1),
}


def _build_schemas() -> Dict[str, Dict[str, Any]]:
    schemas: Dict[str, Dict[str, Any]] = {}
    for task, contract in CONTRACTS.items():
        required = [contract.text_column] + list(contract.label_columns)
        if is_classification(task):
            schemas[task] = {
                "required": required,
                "type": "classification",
                "label_col": contract.label_columns[0],
                "range": _CLASSIFICATION_RANGES.get(
                    task, (0, (contract.num_classes or 1) - 1)
                ),
            }
        elif is_multilabel(task):
            schemas[task] = {
                "required": required,
                "type": "multilabel",
                "cols": list(contract.label_columns),
            }
    return schemas


TASK_SCHEMAS: Dict[str, Dict[str, Any]] = _build_schemas()


# =========================================================
# CORE
# =========================================================

def validate_dataframe(
    df: pd.DataFrame,
    *,
    task: str,
    config: Optional[DataValidatorConfig] = None,
) -> ValidationReport:
    if task not in TASK_SCHEMAS:
        raise ValueError(f"Unknown task: {task}")

    config = config or DataValidatorConfig()
    schema = TASK_SCHEMAS[task]

    report = ValidationReport(rows=len(df), columns=len(df.columns))

    # 1. column presence
    missing = [c for c in schema["required"] if c not in df.columns]
    if missing:
        report.missing_columns = missing
        _handle_error(f"[{task}] Missing columns: {missing}", config)

    # 2. text validation
    contract = get_contract(task)
    text_col = contract.text_column
    if config.check_text and text_col in df.columns:
        text_str = df[text_col].astype(str)
        invalid_mask = (
            df[text_col].isna()
            | (text_str.str.len() < config.min_text_len)
            | (text_str.str.len() > config.max_text_len)
        )
        report.invalid_text_rows = int(invalid_mask.sum())
        if report.invalid_text_rows > 0:
            logger.warning(
                "[%s] Invalid text rows: %d", task, report.invalid_text_rows
            )

    # 3. label validation
    if schema["type"] == "classification":
        _validate_classification(df, schema, report, config, task)
    else:
        _validate_multilabel(df, schema, report, config, task)

    logger.info(
        "Validation | task=%s | rows=%d | text_issues=%d | label_issues=%d",
        task,
        report.rows,
        report.invalid_text_rows,
        sum(report.invalid_label_rows.values()),
    )
    return report


# =========================================================
# CLASSIFICATION
# =========================================================

def _validate_classification(df, schema, report, config, task):
    label_col = schema["label_col"]
    if label_col not in df.columns:
        return

    invalid_mask = df[label_col].isna()
    report.invalid_label_rows[label_col] = int(invalid_mask.sum())

    if config.enforce_label_range:
        low, high = schema["range"]
        # NaNs evaluate False under .between(); treat them with isna() above
        in_range = df[label_col].between(low, high)
        violations = int((~in_range & ~invalid_mask).sum())
        report.label_value_violations[label_col] = violations
        if violations > 0:
            _handle_error(
                f"[{task}] {label_col} has {violations} values outside [{low}, {high}]",
                config,
            )


# =========================================================
# MULTILABEL
# =========================================================

def _validate_multilabel(df, schema, report, config, task):
    for col in schema["cols"]:
        if col not in df.columns:
            continue

        invalid_mask = df[col].isna()
        report.invalid_label_rows[col] = int(invalid_mask.sum())

        if config.enforce_binary_multilabel:
            bad = ~df[col].isin([0, 1])
            violations = int((bad & ~invalid_mask).sum())
            report.label_value_violations[col] = violations
            if violations > 0:
                _handle_error(
                    f"[{task}] {col} has {violations} non-binary rows",
                    config,
                )


# =========================================================
# HELPERS
# =========================================================

def _handle_error(msg: str, config: DataValidatorConfig):
    if config.strict:
        raise ValueError(msg)
    logger.warning(msg)
