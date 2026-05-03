from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config.task_config import TASK_CONFIG
from src.evaluation.calibration import compute_calibration, fit_temperature
from src.evaluation.evaluate_model import evaluate
from src.evaluation.report_writer import save_report
from src.evaluation.task_correlation import compute_task_correlation
from src.evaluation.uncertainty import uncertainty_statistics

logger = logging.getLogger(__name__)


# =========================================================
# VALIDATION
# =========================================================

def validate_inputs(preds: Dict[str, Any], labels: Dict[str, Any]) -> list[str]:
    """Return the list of tasks present in *both* preds and labels."""
    if not isinstance(preds, dict) or not isinstance(labels, dict):
        raise TypeError("preds and labels must be dicts keyed by task")

    common = sorted(set(preds.keys()) & set(labels.keys()))
    if not common:
        raise ValueError("No tasks are common to preds and labels")

    for task in common:
        try:
            preds_len = len(preds[task])
            labels_len = len(labels[task])
        except TypeError as exc:
            raise ValueError(f"task {task!r}: preds/labels must be sized") from exc

        if preds_len != labels_len:
            raise ValueError(
                f"task {task!r}: preds length {preds_len} != labels length {labels_len}"
            )

    extra_preds = sorted(set(preds.keys()) - set(common))
    extra_labels = sorted(set(labels.keys()) - set(common))
    if extra_preds:
        logger.warning("Skipping preds-only tasks: %s", extra_preds)
    if extra_labels:
        logger.warning("Skipping labels-only tasks: %s", extra_labels)

    return common


# =========================================================
# PROBABILITY NORMALIZATION
# =========================================================

def _normalize_probs_for_task(task: str, probs) -> Optional[np.ndarray]:
    """Convert raw probability inputs into a shape acceptable by the metric layer."""
    if probs is None:
        return None

    arr = np.asarray(probs, dtype=float)
    cfg = TASK_CONFIG.get(task)
    task_type = cfg["type"] if cfg else None

    if task_type == "binary":
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 1]
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2 and arr.shape[1] == 1:
            return arr.reshape(-1)
    return arr


# =========================================================
# CORE TASK EVALUATION
# =========================================================

def evaluate_tasks(
    preds: Dict[str, Any],
    labels: Dict[str, Any],
    pred_probs: Optional[Dict[str, Any]] = None,
    *,
    tasks: Optional[list[str]] = None,
) -> Dict[str, Any]:
    if tasks is None:
        tasks = validate_inputs(preds, labels)

    results: Dict[str, Any] = {}
    for task in tasks:
        logger.info("[EVAL] task=%s", task)
        task_kwargs = {"task": task} if task in TASK_CONFIG else {}
        try:
            raw_pred = np.asarray(preds[task])
            # Predictions loaded from JSON may be probability matrices (N, C).
            # Collapse to class indices so the metric layer always gets 1-D input.
            task_type_cfg = (TASK_CONFIG.get(task) or {}).get("type", "")
            if task_type_cfg in ("multiclass", "binary", "classification") and raw_pred.ndim == 2:
                raw_pred = np.argmax(raw_pred, axis=1)
            results[task] = evaluate(
                y_true=labels[task],
                y_pred=raw_pred,
                y_proba=_normalize_probs_for_task(task, (pred_probs or {}).get(task)),
                **task_kwargs,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Evaluation failed for %s: %s", task, exc)

    return results


# =========================================================
# CALIBRATION (LOGITS ONLY)
# =========================================================

def compute_all_calibration(
    logits: Dict[str, Any],
    labels: Dict[str, Any],
    *,
    tasks: Optional[list[str]] = None,
    val_logits: Optional[Dict[str, Any]] = None,
    val_labels: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute calibration metrics for each task.

    CRIT E2: accept ``val_logits`` / ``val_labels`` and, when present, fit the
    per-task temperature on the validation split before applying it to the
    test logits. The previous implementation fitted T on the same logits ECE
    was measured against, which produced optimistically biased calibration
    numbers.
    """
    if tasks is None:
        tasks = sorted(set(logits.keys()) & set(labels.keys()))

    out: Dict[str, Any] = {}
    for task in tasks:
        if task not in TASK_CONFIG:
            logger.debug("[CALIBRATION] skipping unknown task %s", task)
            continue

        task_type = TASK_CONFIG[task]["type"]
        # CRIT E2: try to fit temperature on validation, then apply on test.
        fitted_T: Optional[float] = None
        if val_logits is not None and val_labels is not None and task in val_logits and task in val_labels:
            try:
                fitted_T = fit_temperature(
                    np.asarray(val_logits[task], dtype=float),
                    np.asarray(val_labels[task]),
                    task_type,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Validation T-fit failed for %s: %s", task, exc)
                fitted_T = None

        try:
            out[task] = compute_calibration(
                logits=np.asarray(logits[task]),
                y_true=np.asarray(labels[task]),
                task_type=task_type,
                temperature=fitted_T,
                # Only fall back to fit-on-test when no validation T was fit
                # (the warning inside compute_calibration will fire then).
                apply_temp_scaling=(fitted_T is None),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Calibration failed for %s: %s", task, exc)
    return out


# =========================================================
# MAIN PIPELINE
# =========================================================

def evaluate_and_save(
    preds: Dict[str, Any],
    labels: Dict[str, Any],
    output_path: str | Path,
    *,
    pred_probs: Optional[Dict[str, Any]] = None,
    logits: Optional[Dict[str, Any]] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    tasks = validate_inputs(preds, labels)

    task_results = evaluate_tasks(preds, labels, pred_probs, tasks=tasks)

    summary: Dict[str, Any] = {}
    for task, result in task_results.items():
        if isinstance(result, dict):
            summary[task] = result.get("metrics", {})

    advanced: Dict[str, Any] = {}
    if df is not None:
        try:
            from src.evaluation.advanced_analysis import actor_graph_metrics
            advanced["actor_graph"] = actor_graph_metrics(df)
        except Exception as exc:
            logger.warning("Advanced analysis failed: %s", exc)

    if logits is None:
        logger.info("[CALIBRATION] skipped (no logits provided)")
        calibration: Dict[str, Any] = {}
    else:
        calibration = compute_all_calibration(logits, labels, tasks=tasks)

    uncertainty: Dict[str, Any] = {}
    if pred_probs:
        for task, probs in pred_probs.items():
            if task not in tasks:
                continue
            try:
                arr = np.asarray(probs, dtype=float)
                if arr.ndim == 1:
                    arr = np.column_stack([1.0 - arr, arr])
                task_kwargs = {"task": task} if task in TASK_CONFIG else {}
                uncertainty[task] = uncertainty_statistics(arr, **task_kwargs)
            except Exception as exc:
                logger.warning("Uncertainty failed for %s: %s", task, exc)

    try:
        corr_input = (
            pred_probs if pred_probs is not None
            else logits if logits is not None
            else preds
        )
        if corr_input:
            filtered = {
                t: corr_input[t]
                for t in corr_input
                if t in tasks and t in TASK_CONFIG
            }
            if filtered:
                task_corr = compute_task_correlation(filtered)
                if hasattr(task_corr, "to_dict"):
                    task_corr = task_corr.to_dict()
            else:
                task_corr = {}
        else:
            task_corr = {}
    except Exception as exc:
        logger.warning("Task correlation failed: %s", exc)
        task_corr = {}

    report: Dict[str, Any] = {
        "tasks": task_results,
        "summary": summary,
        "advanced_analysis": advanced,
        "calibration": calibration,
        "uncertainty": uncertainty,
        "task_correlation": task_corr,
    }

    save_report(report, output_path)

    try:
        from src.evaluation.pdf_report import generate_pdf_report
        generate_pdf_report(report, Path(output_path).with_suffix(".pdf"))
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)

    logger.info("[EVALUATION] complete | %d tasks", len(task_results))
    return report


# =========================================================
# LOADERS
# =========================================================

def load_json(path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r") as f:
        return json.load(f)


# =========================================================
# ENTRYPOINT
# =========================================================

def run_evaluation(
    pred_path,
    label_path,
    output_report,
    *,
    pred_probs=None,
    logits=None,
    dataset_path=None,
) -> Dict[str, Any]:
    preds = load_json(pred_path)
    labels = load_json(label_path)

    df = None
    if dataset_path and Path(dataset_path).exists():
        df = pd.read_csv(dataset_path)

    return evaluate_and_save(
        preds=preds,
        labels=labels,
        output_path=output_report,
        pred_probs=pred_probs,
        logits=logits,
        df=df,
    )
