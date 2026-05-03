from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from scipy.special import expit, softmax as scipy_softmax
from transformers import AutoTokenizer

from src.config.task_config import get_task_type
from src.evaluation.calibration import (
    apply_temperature,
    compute_calibration,
    fit_temperature,
)
from src.evaluation.error_analysis import ErrorAnalyzer
from src.evaluation.metrics_engine import compute_metrics_from_preds
from src.evaluation.prediction_collector import PredictionCollector
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.utils.device_utils import autocast_context, move_batch

logger = logging.getLogger(__name__)


# =========================================================
# TOKENIZATION
# =========================================================

def _tokenize(tokenizer, texts: List[str], max_length: int = 512):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# =========================================================
# EVALUATOR
# =========================================================

class Evaluator:
    def __init__(self):
        self.collector = PredictionCollector()
        self.error_analyzer = ErrorAnalyzer()
        self.threshold_optimizer = ThresholdOptimizer()

    # =====================================================
    # MODEL INFERENCE
    # =====================================================

    @staticmethod
    def _batched_predict(
        model,
        texts: List[str],
        task: str,
        tokenizer: AutoTokenizer,
        batch_size: int,
    ) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i: i + batch_size]
                encoded = _tokenize(tokenizer, batch_texts)
                encoded = move_batch(encoded, device)

                with autocast_context():
                    out = model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        task=task,
                    )

                outputs.append(out["logits"].detach().cpu().numpy())

        return np.vstack(outputs)

    @staticmethod
    def _postprocess(logits: np.ndarray, task_type: str, *, threshold: float = 0.5):
        """Convert raw logits → ``(preds, probs)``.

        HIGH E3: use ``scipy.special`` activations directly on the numpy
        array, skipping the redundant numpy → torch → numpy round-trip.
        CRIT E7: accept ``threshold`` so binary / multilabel callers can
        apply a fitted operating point instead of the hard-coded ``0.5``.
        """
        arr = np.asarray(logits, dtype=float)

        if task_type == "multiclass":
            probs = scipy_softmax(arr, axis=-1)
            preds = np.argmax(probs, axis=1).astype(int)
        elif task_type == "binary":
            if arr.ndim == 2 and arr.shape[-1] == 2:
                probs = scipy_softmax(arr, axis=-1)[:, 1]
            else:
                probs = expit(arr).reshape(-1)
            preds = (probs >= threshold).astype(int)
        elif task_type == "multilabel":
            probs = expit(arr)
            preds = (probs >= threshold).astype(int)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return preds, probs

    # =====================================================
    # MAIN ENTRYPOINT
    # =====================================================

    def evaluate(
        self,
        *,
        y_true: Iterable,
        task: str,
        y_pred: Optional[Iterable] = None,
        y_proba: Optional[Iterable] = None,
        model=None,
        texts: Optional[List[str]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        batch_size: int = 32,
        return_logits: bool = False,
        # CRIT E1 + CRIT E2: validation-side data so calibration and
        # threshold optimization are fitted on a held-out split and only
        # applied to the test set passed in via y_true / y_pred / y_proba.
        val_logits: Optional[Iterable] = None,
        val_labels: Optional[Iterable] = None,
        val_y_true: Optional[Iterable] = None,
        val_y_proba: Optional[Iterable] = None,
    ) -> Dict[str, Any]:
        task_type = get_task_type(task)
        logits: Optional[np.ndarray] = None

        # CRIT E1: fit temperature on validation logits *before* postprocessing
        # the test logits, so the test-side preds/probs reflect the calibrated
        # operating point (rather than re-using the same logits to fit ECE).
        fitted_T: Optional[float] = None
        val_logits_arr = (
            np.asarray(val_logits, dtype=float) if val_logits is not None else None
        )
        val_labels_arr = (
            np.asarray(val_labels) if val_labels is not None else None
        )
        if val_logits_arr is not None and val_labels_arr is not None:
            try:
                fitted_T = fit_temperature(
                    val_logits_arr, val_labels_arr, task_type
                )
            except Exception as exc:
                logger.warning("Validation T-fit failed: %s", exc)
                fitted_T = None

        # CRIT E1: fit per-task thresholds on validation probabilities, never
        # on the same predictions metrics will be measured against.
        val_y_true_arr = (
            np.asarray(val_y_true) if val_y_true is not None else val_labels_arr
        )
        val_y_proba_arr = (
            np.asarray(val_y_proba, dtype=float) if val_y_proba is not None else None
        )
        # Derive val probs from val logits when not supplied.
        if (
            val_y_proba_arr is None
            and val_logits_arr is not None
            and task_type in ("binary", "multilabel", "multiclass")
        ):
            try:
                _, val_y_proba_arr = self._postprocess(val_logits_arr, task_type)
            except Exception as exc:
                logger.debug("Could not derive val probs from val logits: %s", exc)

        fitted_thresholds: Optional[Dict[str, Any]] = None
        if (
            task_type in ("binary", "multilabel")
            and val_y_true_arr is not None
            and val_y_proba_arr is not None
        ):
            try:
                fitted_thresholds = self.threshold_optimizer.optimize_from_arrays(
                    y_true=val_y_true_arr,
                    probs=val_y_proba_arr,
                    task_type=task_type,
                )
            except AttributeError:
                # Optimizer doesn't expose optimize_from_arrays — fall back to
                # the legacy ``optimize(collected)`` path *on validation data*.
                try:
                    val_collected = self.collector.collect(
                        y_true=val_y_true_arr,
                        y_pred=(val_y_proba_arr >= 0.5).astype(int)
                        if val_y_proba_arr.ndim <= 2
                        else None,
                        y_proba=val_y_proba_arr,
                        logits=val_logits_arr,
                        task=task,
                        task_type=task_type,
                    )
                    fitted_thresholds = self.threshold_optimizer.optimize(val_collected)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Threshold fit on val failed: %s", exc)
            except Exception as exc:  # pragma: no cover
                logger.warning("Threshold fit on val failed: %s", exc)

        # Resolve per-task threshold for the test postprocess step.
        op_threshold = 0.5
        if isinstance(fitted_thresholds, dict):
            t_val = fitted_thresholds.get("threshold")
            if isinstance(t_val, (int, float)):
                op_threshold = float(t_val)

        if model is not None:
            if texts is None or tokenizer is None:
                raise ValueError("model mode requires texts + tokenizer")
            logits = self._batched_predict(model, texts, task, tokenizer, batch_size)
            # CRIT E1 / CRIT E2: scale test logits by validation-fitted T
            # before deriving probs/preds so downstream metrics see the
            # calibrated, operating-point-aware predictions.
            scaled_logits = (
                apply_temperature(logits, fitted_T) if fitted_T is not None else logits
            )
            y_pred, y_proba = self._postprocess(
                scaled_logits, task_type, threshold=op_threshold
            )

        y_true_arr = np.asarray(y_true)
        if y_true_arr.size == 0:
            raise ValueError("y_true cannot be empty")

        if y_pred is None:
            raise ValueError("y_pred must be provided if model is None")

        y_pred_arr = np.asarray(y_pred)
        # Convert probability/logit matrices → class indices for classification tasks
        if task_type in ("binary", "multiclass", "classification") and y_pred_arr.ndim == 2:
            y_pred_arr = np.argmax(y_pred_arr, axis=1)
        y_true_arr = np.asarray(y_true_arr).reshape(-1) if task_type in ("binary", "multiclass", "classification") else y_true_arr
        y_proba_arr = np.asarray(y_proba, dtype=float) if y_proba is not None else None

        collected = self.collector.collect(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_proba=y_proba_arr,
            logits=logits,
            task=task,
            task_type=task_type,
        )

        # CRIT E5: forward the authoritative task_type so the metrics layer
        # never silently misroutes a 3-class slice that happens to contain
        # only labels {0, 1} into the binary code path.
        metrics = compute_metrics_from_preds(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            task_type=task_type,
            y_proba=y_proba_arr,
        )

        calibration: Dict[str, Any] = {}
        if logits is not None:
            try:
                # CRIT E2: pass the validation-fitted temperature explicitly.
                # ``apply_temp_scaling`` defaults to False, so without a fitted
                # T the calibrator reports raw ECE/Brier instead of silently
                # fitting on test data.
                calibration = compute_calibration(
                    logits=logits,
                    y_true=y_true_arr,
                    task_type=task_type,
                    temperature=fitted_T,
                )
            except Exception as exc:
                logger.warning("Calibration failed: %s", exc)

        try:
            error_analysis = self.error_analyzer.analyze(collected)
        except Exception as exc:
            logger.warning("Error analysis failed: %s", exc)
            error_analysis = {}

        # CRIT E1: never fit thresholds on ``collected`` (that is the test
        # split). Return the ones fitted on validation if provided; otherwise
        # report no operating point so callers can detect the missing val data.
        thresholds = fitted_thresholds

        if y_true_arr.ndim == 1:
            labels, counts = np.unique(y_true_arr, return_counts=True)
            class_counts = {str(int(label)): int(count) for label, count in zip(labels, counts)}
            dataset_stats = {
                "num_samples": int(len(y_true_arr)),
                "num_classes": int(len(labels)),
                "class_counts": class_counts,
                "class_distribution": class_counts,
            }
        else:
            dataset_stats = {
                "num_samples": int(y_true_arr.shape[0]),
                "num_labels": int(y_true_arr.shape[1]),
                "density": float(np.mean(y_true_arr)),
            }

        result = {
            "task": task,
            "task_type": task_type,
            "metrics": metrics,
            "calibration": calibration,
            "error_analysis": error_analysis,
            "optimal_thresholds": thresholds,
            "dataset_stats": dataset_stats,
        }
        if return_logits and logits is not None:
            result["logits"] = logits
        return result

    # =====================================================
    # MULTITASK SUMMARY
    # =====================================================

    @staticmethod
    def multitask(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """HIGH E7: roll up the *numeric union* of metrics across tasks.

        The original implementation only summarized a hard-coded triple of
        keys (``accuracy``, ``f1_macro``, ``f1_weighted``), so task-specific
        metrics like ``roc_auc``, ``mcc``, ``balanced_accuracy``, ``log_loss``
        were silently dropped from the multitask report. We now sample-weight
        every numeric metric that appears on any task.
        """
        if not results:
            return {}

        weights: Dict[str, float] = {}
        metric_keys: set[str] = set()
        for task, result in results.items():
            stats = result.get("dataset_stats") or {}
            weights[task] = float(stats.get("num_samples", 1) or 1)
            for k, v in (result.get("metrics") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    metric_keys.add(k)

        summary: Dict[str, float] = {}
        for metric in sorted(metric_keys):
            vals: List[float] = []
            wts: List[float] = []
            for task, result in results.items():
                value = (result.get("metrics") or {}).get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not np.isfinite(float(value)):
                        continue
                    vals.append(float(value))
                    wts.append(weights[task])
            if vals:
                summary[f"weighted_{metric}"] = float(np.average(vals, weights=wts))
        return summary

    # =====================================================
    # FEATURE IMPORTANCE — STATIC TEST-FACING API
    # =====================================================

    @staticmethod
    def feature_importance_ablation(
        *,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        scoring: str = "accuracy",
    ) -> Dict[str, float]:
        """Compute per-feature ablation importance.

        For each feature, replace its column with the column mean, score the
        model, and record the drop in score relative to the baseline. The
        result is a ``{feature_name: importance}`` mapping where larger values
        indicate features the model relies on more.
        """
        from sklearn.metrics import accuracy_score, f1_score

        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != len(feature_names):
            raise ValueError("feature_names length must match X.shape[1]")

        if scoring == "accuracy":
            score_fn = lambda yt, yp: accuracy_score(yt, yp)
        elif scoring == "f1":
            score_fn = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
        else:
            raise ValueError(f"Unsupported scoring: {scoring}")

        baseline = score_fn(y, model.predict(X))
        importances: Dict[str, float] = {}

        for idx, name in enumerate(feature_names):
            X_ablated = X.copy()
            X_ablated[:, idx] = float(np.mean(X[:, idx]))
            try:
                preds = model.predict(X_ablated)
                ablated_score = score_fn(y, preds)
            except Exception as exc:
                logger.warning("Ablation failed for %s: %s", name, exc)
                ablated_score = baseline

            importances[name] = float(baseline - ablated_score)

        return importances

    @staticmethod
    def feature_importance_shap(
        *,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        max_samples: int = 100,
    ) -> Dict[str, float]:
        """Compute SHAP-style importance for tabular ``model``.

        ``max_samples`` controls how many rows the explainer sees and must be
        positive. Falls back to a permutation-based importance signal when the
        ``shap`` library is unavailable so the public contract stays stable.
        """
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[1] != len(feature_names):
            raise ValueError("feature_names length must match X.shape[1]")

        sample = X[: min(max_samples, X.shape[0])]

        try:
            import shap  # type: ignore

            explainer = shap.Explainer(model.predict, sample)
            shap_values = explainer(sample).values
            mean_abs = np.mean(np.abs(shap_values), axis=0)
        except Exception as exc:
            logger.debug("Falling back to permutation importance: %s", exc)
            baseline_pred = np.asarray(model.predict(sample))
            mean_abs = np.zeros(sample.shape[1], dtype=float)
            rng = np.random.default_rng(0)
            for idx in range(sample.shape[1]):
                perturbed = sample.copy()
                perturbed[:, idx] = rng.permutation(perturbed[:, idx])
                try:
                    preds = np.asarray(model.predict(perturbed))
                except Exception:
                    continue
                mean_abs[idx] = float(np.mean(preds != baseline_pred))

        return {
            name: float(mean_abs[idx]) for idx, name in enumerate(feature_names)
        }


__all__ = ["Evaluator"]
