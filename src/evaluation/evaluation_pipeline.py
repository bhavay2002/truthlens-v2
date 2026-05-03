"""End-to-end evaluation pipeline that orchestrates collection → metrics →
calibration → uncertainty → error analysis → correlation → reporting."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.config.task_config import TASK_CONFIG
from src.evaluation.calibration import (
    apply_temperature,
    compute_calibration,
    fit_calibration,
)
from src.evaluation.error_analysis import error_analysis
from src.evaluation.evaluate_model import evaluate, _postprocess_logits
from src.evaluation.fairness import fairness_report_multi
from src.evaluation.pdf_report import generate_pdf_report
from src.evaluation.prediction_collector import collect_all_tasks
from src.evaluation.report_writer import save_report
from src.evaluation.task_correlation import compute_task_correlation
from src.evaluation.threshold_optimizer import optimize_thresholds
from src.evaluation.uncertainty import uncertainty_statistics

try:
    from src.evaluation.mlflow_tracker import (
        log_evaluation_report,
        log_task_metrics,
    )
except Exception:  # pragma: no cover
    log_task_metrics = None
    log_evaluation_report = None

try:
    from src.inference.prediction_service import PredictionService
except Exception:  # pragma: no cover - optional dep at import time
    PredictionService = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# =========================================================
# PREDICTION SHAPE VALIDATION
# =========================================================

def _validate_pred_shape(task: str, preds: np.ndarray) -> np.ndarray:
    """Ensure preds has a shape consistent with the task type.

    - multilabel tasks: keep (N, C) as-is; collapse float proba matrices via
      threshold=0.5 when values are in [0, 1].
    - binary/multiclass: collapse (N, C) → (N,) via argmax.

    Raises ValueError with a clear message if shape cannot be resolved.
    """
    arr = np.asarray(preds)
    task_type = TASK_CONFIG.get(task, {}).get("type", "")

    if task_type == "multilabel":
        # Keep 2-D; convert float probabilities to binary predictions.
        if arr.ndim == 2:
            if arr.dtype.kind == "f":
                logger.debug(
                    "[VALIDATE] task=%s multilabel: thresholding proba %s → binary",
                    task, arr.shape,
                )
                arr = (arr >= 0.5).astype(np.int32)
            return arr
        # 1-D for multilabel: wrap into (N, 1) so downstream sees 2-D.
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        raise ValueError(
            f"[VALIDATE] {task}: multilabel y_pred must be 1-D or 2-D, got shape {arr.shape}."
        )

    # binary / multiclass path
    if arr.ndim == 2:
        logger.debug("[VALIDATE] task=%s converting preds %s → argmax", task, arr.shape)
        arr = np.argmax(arr, axis=1)
    if arr.ndim != 1:
        raise ValueError(
            f"[VALIDATE] {task}: y_pred must be 1-D after argmax but got shape {arr.shape}. "
            "Pass class-index predictions (N,), not probabilities/logits (N, C)."
        )
    return arr


# =========================================================
# PREDICTION SERVICE PATH (BATCHED)
# =========================================================

def _collect_via_prediction_service(
    prediction_service,
    texts: List[str],
    tasks: List[str],
    *,
    batch_size: int = 32,
) -> Dict[str, Dict[str, Any]]:
    """Drive a PredictionService over ``texts`` and collect per-task outputs.

    HIGH E6: pre-allocate typed numpy arrays from ``TASK_CONFIG[task]
    ["num_labels"]`` instead of accumulating Python lists and calling
    ``np.asarray`` on a potentially ragged structure (which silently produced
    object-dtype arrays or raised). Each task gets ``probabilities``,
    ``predictions`` and ``logits`` slots sized by task type.

    HIGH E2: when ``predict_batch`` isn't available, still iterate the texts
    in chunks of ``batch_size`` so behavior matches the batched path (memory
    use, progress logging, and downstream slot indexing are all consistent).
    """
    n = len(texts)

    # --- HIGH E6: pre-allocate typed slots per task ---------------------
    out: Dict[str, Dict[str, Any]] = {}
    task_meta: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        cfg = TASK_CONFIG.get(task) or {}
        ttype = cfg.get("type", "binary")
        nl = int(cfg.get("num_labels", 1) or 1)

        if ttype == "binary":
            proba_shape: tuple[int, ...] = (n,)
            pred_shape: tuple[int, ...] = (n,)
            logits_shape: tuple[int, ...] = (n, nl) if nl > 1 else (n,)
        elif ttype == "multiclass":
            proba_shape = (n, nl)
            pred_shape = (n,)
            logits_shape = (n, nl)
        elif ttype == "multilabel":
            proba_shape = (n, nl)
            pred_shape = (n, nl)
            logits_shape = (n, nl)
        else:
            proba_shape = (n,)
            pred_shape = (n,)
            logits_shape = (n,)

        out[task] = {
            "probabilities": np.zeros(proba_shape, dtype=np.float32),
            "predictions": np.zeros(pred_shape, dtype=np.int32),
            "logits": np.zeros(logits_shape, dtype=np.float32),
        }
        task_meta[task] = {
            "type": ttype,
            "logits_seen": False,
            "pred_dtype_locked": False,
        }

    def _store(idx: int, result: Dict[str, Any]) -> None:
        for task in tasks:
            task_out = (result.get("tasks") or {}).get(task) or {}
            probs = task_out.get("probabilities")
            preds = task_out.get("predictions")
            logits = task_out.get("logits")

            if probs is not None:
                arr = np.asarray(probs, dtype=np.float32)
                target = out[task]["probabilities"]
                # Reshape on first write if pre-allocated shape doesn't match
                # (covers binary heads that emit (2,) instead of scalar).
                if arr.shape != target[idx].shape:
                    if arr.ndim == 1 and arr.size == 2 and target[idx].ndim == 0:
                        arr = arr[1:2].reshape(target[idx].shape)
                    elif target[idx].ndim == 0 and arr.size == 1:
                        arr = arr.reshape(())
                target[idx] = arr

            if preds is not None:
                p = np.asarray(preds)
                target = out[task]["predictions"]
                if p.shape != target[idx].shape:
                    if target[idx].ndim == 0 and p.size == 1:
                        p = p.reshape(())
                target[idx] = p

            if logits is not None:
                la = np.asarray(logits, dtype=np.float32)
                target = out[task]["logits"]
                if la.shape != target[idx].shape:
                    if target[idx].ndim == 0 and la.size == 1:
                        la = la.reshape(())
                target[idx] = la
                task_meta[task]["logits_seen"] = True

    predict_batch = getattr(prediction_service, "predict_batch", None)

    if callable(predict_batch):
        for i in range(0, n, batch_size):
            batch_results = predict_batch(texts[i: i + batch_size])
            for j, result in enumerate(batch_results):
                _store(i + j, result)
    else:
        # HIGH E2: chunk the per-text fallback so memory use and behavior
        # match the batched path. We still call ``predict`` once per text
        # (no implicit batching is possible without a batch API), but
        # iterate in deterministic ``batch_size`` slabs.
        for i in range(0, n, batch_size):
            chunk = texts[i: i + batch_size]
            for j, text in enumerate(chunk):
                _store(i + j, prediction_service.predict(text))

    # Drop logits slots that were never populated.
    for task in tasks:
        if not task_meta[task]["logits_seen"]:
            out[task].pop("logits", None)

    return out


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_evaluation_pipeline(
    *,
    model=None,
    tokenizer=None,
    texts: List[str],
    labels: Dict[str, Any],
    tasks: Optional[List[str]] = None,
    output_path: Optional[str] = None,
    prediction_service=None,
    enable_calibration: bool = True,
    enable_threshold_opt: bool = True,
    enable_uncertainty: bool = True,
    enable_error_analysis: bool = True,
    enable_correlation: bool = True,
    val_logits: Optional[Dict[str, np.ndarray]] = None,
    val_labels: Optional[Dict[str, np.ndarray]] = None,
    sensitive_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline.

    ``val_logits`` / ``val_labels`` enable a fit-on-val / apply-on-test
    calibration split; when omitted the calibrator falls back to fitting on the
    same data it scores (with a warning).
    """
    tasks = tasks or list(TASK_CONFIG.keys())

    # Filter to only tasks that have labels supplied — avoids KeyError when a
    # caller passes labels for a single task (e.g. "bias") but tasks defaults
    # to the full TASK_CONFIG key list.
    tasks = [t for t in tasks if t in labels]
    if not tasks:
        logger.warning("[PIPELINE] No tasks with matching labels — returning empty report")
        return {"tasks": {}, "summary": {}}

    logger.info("[PIPELINE] Collecting predictions for %d tasks", len(tasks))
    if prediction_service is not None:
        predictions = _collect_via_prediction_service(prediction_service, texts, tasks)
    else:
        predictions = collect_all_tasks(
            model=model,
            texts=texts,
            tokenizer=tokenizer,
            tasks=tasks,
        )

    report: Dict[str, Any] = {"tasks": {}}
    all_probs: Dict[str, np.ndarray] = {}
    all_logits: Dict[str, np.ndarray] = {}
    all_confidence: Dict[str, np.ndarray] = {}
    all_uncertainty: Dict[str, float] = {}

    fitted_temperatures: Dict[str, float] = {}
    if val_logits and val_labels:
        for task in tasks:
            if task not in val_logits or task not in val_labels:
                continue
            try:
                t = fit_calibration(
                    val_logits=val_logits[task],
                    val_y_true=val_labels[task],
                    task_type=TASK_CONFIG[task]["type"],
                )
                if t is not None:
                    fitted_temperatures[task] = t
            except Exception as exc:
                logger.warning("Temperature fit failed for %s: %s", task, exc)

    for task in tasks:
        logger.info("[PIPELINE] task=%s", task)

        # Support BOTH dict formats:
        #   Nested: {task: {"predictions": ..., "probabilities": ..., "logits": ...}}
        #   Flat:   {"bias_predictions": ..., "bias_probabilities": ..., "bias_logits": ...}
        task_preds = predictions.get(task, {})
        flat_pred_key = f"{task}_predictions"
        flat_prob_key = f"{task}_probabilities"
        flat_logits_key = f"{task}_logits"

        if flat_pred_key in predictions:
            # Flat-key format — use task-prefixed keys directly from top-level dict.
            logits_raw = predictions.get(flat_logits_key)
            probs_raw = predictions.get(flat_prob_key)
            preds_raw = predictions.get(flat_pred_key)
        else:
            # Nested format — look inside the per-task sub-dict.
            logits_raw = task_preds.get("logits")
            probs_raw = task_preds.get("probabilities", task_preds.get("y_proba"))
            preds_raw = task_preds.get("predictions", task_preds.get("y_pred"))

        logits = np.asarray(logits_raw) if logits_raw is not None else None
        probs = np.asarray(probs_raw) if probs_raw is not None else np.empty(0)
        preds = np.asarray(preds_raw) if preds_raw is not None else np.empty(0)

        # Always use class-index predictions (1-D). If probabilities or logits
        # ended up in the preds slot, collapse them via argmax now.
        task_type_str = TASK_CONFIG.get(task, {}).get("type", "")
        if task_type_str in ("multiclass", "binary", "classification") and preds.ndim == 2:
            preds = np.argmax(preds, axis=1)

        # Final shape guard — collapses any remaining 2-D array and gives a
        # clear error if the shape still can't be resolved to the right shape.
        preds = _validate_pred_shape(task, preds)
        logger.debug("[PIPELINE] task=%s preds_shape=%s probs_shape=%s", task, preds.shape, probs.shape)
        y_true = np.asarray(labels[task])

        # For multilabel tasks: if model output is 1D class-index but y_true
        # is 2D multi-hot, broadcast preds to matching shape via one-hot so
        # shape validation in evaluate() doesn't raise.
        task_type_for_eval = TASK_CONFIG.get(task, {}).get("type", "")
        if task_type_for_eval == "multilabel" and y_true.ndim == 2 and preds.ndim == 1:
            n_labels = y_true.shape[1]
            onehot = np.zeros((len(preds), n_labels), dtype=np.int32)
            for i, cls in enumerate(preds):
                if 0 <= int(cls) < n_labels:
                    onehot[i, int(cls)] = 1
            preds = onehot
            logger.debug(
                "[PIPELINE] task=%s multilabel: broadcast 1-D preds → one-hot %s",
                task, preds.shape,
            )

        try:
            eval_result = evaluate(
                y_true=y_true,
                y_pred=preds,
                y_proba=probs,
                task=task,
            )
        except Exception as exc:
            logger.warning(
                "[PIPELINE] task=%s evaluate() failed (%s) — skipping metrics for this task",
                task, exc,
            )
            report["tasks"][task] = {"metrics": {}, "error": str(exc)}
            continue

        report["tasks"][task] = eval_result

        all_probs[task] = probs
        if logits is not None:
            all_logits[task] = logits

        if enable_threshold_opt:
            try:
                report.setdefault("optimal_thresholds", {})[task] = optimize_thresholds(
                    y_true=y_true, probs=probs, task=task
                )
            except Exception as exc:
                logger.warning("Threshold optimization failed for %s: %s", task, exc)

        if enable_calibration and logits is not None:
            try:
                cal = compute_calibration(
                    logits=logits,
                    y_true=y_true,
                    task_type=TASK_CONFIG[task]["type"],
                    temperature=fitted_temperatures.get(task),
                )
                report.setdefault("calibration", {})[task] = cal
                if "confidence" in cal:
                    all_confidence[task] = np.asarray(cal["confidence"])
            except Exception as exc:
                logger.warning("Calibration failed for %s: %s", task, exc)

            # HIGH E8: when a validation-fit temperature exists, recompute
            # probs/preds from the *scaled* logits and emit a parallel
            # ``metrics_calibrated`` block alongside the raw metrics. This
            # exposes the actual lift from temperature scaling — without it
            # the calibration block updates ECE/Brier but downstream metrics
            # still report the uncalibrated operating point.
            T_fit = fitted_temperatures.get(task)
            if T_fit is not None:
                try:
                    eval_result["metrics_raw"] = eval_result.get("metrics", {})
                    scaled_logits = apply_temperature(logits, T_fit)
                    cal_preds, cal_probs = _postprocess_logits(
                        scaled_logits, TASK_CONFIG[task]["type"]
                    )
                    cal_eval = evaluate(
                        y_true=y_true,
                        y_pred=cal_preds,
                        y_proba=cal_probs,
                        task=task,
                    )
                    eval_result["metrics_calibrated"] = cal_eval.get("metrics", {})
                except Exception as exc:
                    logger.warning(
                        "Calibrated metric recompute failed for %s: %s", task, exc
                    )

        if enable_uncertainty:
            try:
                unc = uncertainty_statistics(
                    np.asarray(probs), task=task, logits=logits
                )
                report.setdefault("uncertainty", {})[task] = unc
                mean_entropy = unc.get("mean_entropy")
                if mean_entropy is not None:
                    all_uncertainty[task] = float(mean_entropy)
            except Exception as exc:
                logger.warning("Uncertainty failed for %s: %s", task, exc)

        if enable_error_analysis:
            try:
                report.setdefault("error_analysis", {})[task] = error_analysis(
                    y_true=y_true,
                    y_pred=preds,
                    probs=probs,
                    texts=texts,
                    task=task,
                )
            except Exception as exc:
                logger.warning("Error analysis failed for %s: %s", task, exc)

        if log_task_metrics is not None:
            try:
                log_task_metrics(task, eval_result.get("metrics", {}))
            except Exception:
                pass

    if enable_correlation:
        try:
            corr = compute_task_correlation(all_probs)
            report["task_correlation"] = corr.to_dict()
        except Exception as exc:
            logger.warning("Correlation failed: %s", exc)

    # Section 10: per-task fairness slicing. ``sensitive_attributes`` is
    # ``{task: {attr_name: values}}``. Only emit a fairness block for tasks
    # the caller actually supplied attributes for so we don't pollute reports
    # with empty slices.
    if sensitive_attributes:
        fairness_block: Dict[str, Any] = {}
        for task, attr_map in sensitive_attributes.items():
            if task not in report["tasks"] or not attr_map:
                continue
            task_preds = predictions.get(task, {})
            preds_arr = np.asarray(
                task_preds.get("predictions", task_preds.get("y_pred"))
            )
            try:
                y_true_arr = np.asarray(labels[task])
            except KeyError:
                continue
            try:
                fairness_block[task] = fairness_report_multi(
                    y_true=y_true_arr,
                    y_pred=preds_arr,
                    sensitive_attributes=attr_map,
                    task_type=TASK_CONFIG[task]["type"],
                )
            except Exception as exc:
                logger.warning("Fairness failed for %s: %s", task, exc)
        if fairness_block:
            report["fairness"] = fairness_block

    # Section 5/4: NaN/Inf guard. A single non-finite value here used to
    # propagate into the JSON report and crash matplotlib downstream.
    # Track per-task f1 to surface ``worst_task_f1`` in the summary.
    summary: Dict[str, float] = {}
    f1_per_task: Dict[str, float] = {}
    for task, data in report["tasks"].items():
        for k, v in (data.get("metrics") or {}).items():
            if not isinstance(v, (int, float)):
                continue
            fv = float(v)
            if not np.isfinite(fv):
                logger.warning(
                    "Non-finite metric %s_%s=%r; coercing to 0.0", task, k, v
                )
                fv = 0.0
            summary[f"{task}_{k}"] = fv
            if k in ("f1", "f1_macro") and task not in f1_per_task:
                f1_per_task[task] = fv

    if f1_per_task:
        worst_task = min(f1_per_task, key=f1_per_task.get)
        summary["worst_task_f1"] = float(f1_per_task[worst_task])
        summary["worst_task_f1_name"] = worst_task  # type: ignore[assignment]
        best = float(max(f1_per_task.values()))
        worst = float(f1_per_task[worst_task])
        summary["f1_imbalance_index"] = (
            float(np.clip((best - worst) / best, 0.0, 1.0)) if best > 0 else 0.0
        )

    report["summary"] = summary

    if output_path:
        save_report(report, output_path)
        try:
            generate_pdf_report(report, str(output_path).replace(".json", ".pdf"))
        except Exception as exc:
            logger.warning("PDF generation failed: %s", exc)

    if log_evaluation_report is not None:
        try:
            log_evaluation_report(report)
        except Exception:
            pass

    logger.info("[PIPELINE] Evaluation complete")
    return report
