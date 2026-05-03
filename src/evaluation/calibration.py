from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import expit, softmax as scipy_softmax
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.evaluation.reliability_diagram import ReliabilityDiagram
# CFG6: ``TemperatureScaler`` now lives in ``src.models.calibration``.
# We re-export it from this module so the public ``src.evaluation.
# calibration.TemperatureScaler`` symbol stays stable, while the
# import arrow runs ``evaluation -> models`` like every other
# production-stack import (the previous arrangement had the models
# layer importing from evaluation, which was a layering violation).
from src.models.calibration.temperature_scaling import TemperatureScaler  # noqa: F401

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# VALIDATION
# =========================================================

def _validate_inputs(y_true, probs) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)

    if y.shape[0] != p.shape[0]:
        raise ValueError("Mismatch in samples between y_true and probs")

    return y, p


# =========================================================
# ACTIVATIONS
# =========================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax. Section 8: defer to :mod:`scipy.special`
    so this module shares the same implementation as ``evaluate_model`` /
    ``prediction_collector`` instead of carrying a hand-rolled version that
    can drift on edge cases (e.g. ``-inf`` columns)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return scipy_softmax(x, axis=1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Section 8: use :func:`scipy.special.expit` instead of ``1/(1+exp(-x))``.
    The hand-rolled form overflows to ``inf`` for large negative inputs and
    silently produces zeros for small positive inputs, which corrupts both
    calibration metrics and ECE binning."""
    return expit(np.asarray(x, dtype=float))


# =========================================================
# CONFIDENCE
# =========================================================

def extract_confidence(probs: np.ndarray, *, task_type: str = "multiclass") -> np.ndarray:
    """Top-class confidence with multilabel-aware semantics."""
    arr = np.asarray(probs, dtype=float)

    if task_type == "multilabel":
        return np.mean(np.maximum(arr, 1.0 - arr), axis=1)

    if arr.ndim == 1:
        # binary as P(class=1) — confidence is max(p, 1-p)
        return np.maximum(arr, 1.0 - arr)

    if arr.ndim == 2 and arr.shape[1] == 2 and task_type == "binary":
        return np.max(arr, axis=1)

    return np.max(arr, axis=1)


# =========================================================
# TEMPERATURE SCALER
# =========================================================
# CFG6: ``TemperatureScaler`` is now defined in
# ``src.models.calibration.temperature_scaling`` and re-exported above.
# The class is intentionally NOT redefined here.


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    *,
    max_iter: int = 50,
) -> float:
    """Fit a single scalar temperature on validation logits."""
    logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.tensor(np.asarray(labels))

    model = TemperatureScaler()
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=max_iter)

    # Section 9: clamp ``T`` inside the LBFGS closure. Without the in-place
    # clamp LBFGS occasionally walks the parameter to a value <= 0 between
    # steps, which makes ``model(logits)`` produce NaNs and either crashes
    # the loss or silently locks the optimizer at a degenerate point.
    def _clamp_T() -> None:
        with torch.no_grad():
            model.temperature.data.clamp_(min=1e-3)

    if task_type == "multiclass":
        loss_fn = nn.CrossEntropyLoss()
        labels_long = labels_t.long()

        def closure():
            _clamp_T()
            optimizer.zero_grad()
            loss = loss_fn(model(logits_t), labels_long)
            loss.backward()
            return loss

    elif task_type == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
        labels_float = labels_t.float()

        def closure():
            _clamp_T()
            optimizer.zero_grad()
            scaled = model(logits_t).reshape(-1)
            loss = loss_fn(scaled, labels_float.reshape(-1))
            loss.backward()
            return loss

    elif task_type == "multilabel":
        loss_fn = nn.BCEWithLogitsLoss()
        labels_float = labels_t.float()

        def closure():
            _clamp_T()
            optimizer.zero_grad()
            scaled = model(logits_t)
            loss = loss_fn(scaled, labels_float)
            loss.backward()
            return loss

    else:
        raise ValueError(f"Unsupported task_type for temperature scaling: {task_type}")

    optimizer.step(closure)
    _clamp_T()

    T = float(model.temperature.detach().cpu().item())
    if not np.isfinite(T) or T <= 0:
        logger.warning("Temperature optimization produced invalid value %s; falling back to 1.0", T)
        T = 1.0

    logger.info("[CALIBRATION] Learned temperature: %.4f", T)
    return T


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return np.asarray(logits, dtype=float) / max(T, EPS)


# =========================================================
# BRIER SCORE
# =========================================================

def brier_score(y_true, probs, task_type: str) -> float:
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)

    if task_type == "multiclass":
        if p.ndim != 2:
            raise ValueError("multiclass brier requires 2D probs")
        one_hot = np.eye(p.shape[1])[y]
        return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))

    if task_type == "binary":
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        return float(np.mean((p.reshape(-1) - y.reshape(-1)) ** 2))

    if task_type == "multilabel":
        return float(np.mean((p - y) ** 2))

    raise ValueError(f"Unsupported task_type: {task_type}")


# =========================================================
# ECE (Expected Calibration Error)
# =========================================================

def expected_calibration_error(
    y_true,
    probs,
    n_bins: int = 10,
    *,
    task_type: str = "binary",
) -> float:
    """ECE for a 1D confidence vector or 2D (multiclass) prob matrix."""
    y, p = _validate_inputs(y_true, probs)

    if p.ndim == 2 and task_type == "multiclass":
        confidence = np.max(p, axis=1)
        preds = np.argmax(p, axis=1)
        correct = (preds == y).astype(float)
    elif p.ndim == 2 and task_type == "binary" and p.shape[1] == 2:
        confidence = p[:, 1]
        correct = (y == 1).astype(float)
    else:
        confidence = p.reshape(-1)
        correct = y.reshape(-1).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, bins) - 1, 0, n_bins - 1)

    counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    # Section 8: refuse to invent an ECE when no samples landed in any bin.
    # The previous ``max(counts.sum(), 1.0)`` clamp returned 0.0 silently and
    # made empty calibration sets look perfectly calibrated.
    if counts.sum() == 0:
        raise ValueError("expected_calibration_error: empty input (no samples binned)")

    sum_acc = np.bincount(bin_ids, weights=correct, minlength=n_bins)
    sum_conf = np.bincount(bin_ids, weights=confidence, minlength=n_bins)

    safe = np.where(counts > 0, counts, 1.0)
    bin_acc = sum_acc / safe
    bin_conf = sum_conf / safe

    ece = float(np.sum(counts / counts.sum() * np.abs(bin_acc - bin_conf)))
    return ece


def classwise_ece(y_true, probs, n_bins=10) -> Dict[str, float]:
    y, p = _validate_inputs(y_true, probs)
    if p.ndim != 2:
        raise ValueError("classwise_ece requires 2D probs")

    return {
        f"class_{c}": expected_calibration_error(
            (y == c).astype(int), p[:, c], n_bins, task_type="binary"
        )
        for c in range(p.shape[1])
    }


def multilabel_ece(y_true, probs, n_bins=10) -> Dict[str, Any]:
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)

    per_label = [
        expected_calibration_error(
            y[:, i].astype(int), p[:, i], n_bins, task_type="binary"
        )
        for i in range(p.shape[1])
    ]

    return {
        "macro_ece": float(np.mean(per_label)),
        "per_label_ece": per_label,
    }


# =========================================================
# RELIABILITY HELPER
# =========================================================

def compute_reliability(y_true, probs, n_bins: int = 10, *, task_type: str = "multiclass"):
    try:
        diagram = ReliabilityDiagram(n_bins=n_bins)
        return diagram.compute(probs=probs, y_true=y_true, task_type=task_type)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Reliability diagram failed: %s", e)
        return {}


# =========================================================
# FULL PIPELINE — fit + apply
# =========================================================

def compute_calibration(
    logits: Optional[np.ndarray],
    y_true: Iterable,
    task_type: str,
    *,
    apply_temp_scaling: bool = False,
    temperature: Optional[float] = None,
    n_bins: int = 10,
    return_confidence_array: bool = False,
) -> Dict[str, Any]:
    """Fit (or apply pre-fit) temperature scaling and compute calibration metrics.

    Pass ``temperature`` to use a previously fitted value (split fit/apply across
    validation/test data). When ``apply_temp_scaling`` is opt-in ``True`` *and*
    ``temperature`` is not provided, the temperature is fitted on ``logits`` —
    which is the same data ECE/Brier are measured on, so the resulting numbers
    are statistically biased. This is now logged at warning level.

    CRIT E2: ``apply_temp_scaling`` defaults to ``False`` to make the fit-on-test
    path explicit. Callers that want the previous behavior must opt in.
    """
    if logits is None:
        raise ValueError("logits required")

    logits_arr = np.asarray(logits, dtype=float)
    y_true_arr = np.asarray(y_true)

    T: Optional[float] = None
    if temperature is not None:
        T = float(temperature)
        scaled = apply_temperature(logits_arr, T)
    elif apply_temp_scaling:
        # CRIT E2: surface the leakage. Promote from debug to warning so
        # operators see it whenever fit-on-test calibration sneaks back in.
        logger.warning(
            "Temperature being fitted on the same data calibration metrics "
            "will be measured on; ECE/Brier will be optimistically biased. "
            "Pass ``temperature`` (fitted on validation logits) to avoid this."
        )
        try:
            T = fit_temperature(logits_arr, y_true_arr, task_type)
            scaled = apply_temperature(logits_arr, T)
        except Exception as exc:
            logger.warning("Temperature fitting failed: %s", exc)
            T = None
            scaled = logits_arr
    else:
        scaled = logits_arr

    if task_type == "multiclass":
        probs = softmax(scaled)
    elif task_type in ("binary", "multilabel"):
        probs = sigmoid(scaled)
    else:
        raise ValueError(f"Invalid task_type: {task_type}")

    confidence = extract_confidence(probs, task_type=task_type)

    results: Dict[str, Any] = {
        "task_type": task_type,
        "mean_confidence": float(np.mean(confidence)),
        "std_confidence": float(np.std(confidence)),
    }

    if return_confidence_array:
        results["confidence"] = confidence.tolist()

    if task_type == "multilabel":
        results.update(multilabel_ece(y_true_arr, probs, n_bins))
    elif task_type == "binary":
        # DRY-E-CALIBRATION-IF fix: the previous if/else had identical bodies
        # in both branches (sigmoid of binary logits always produces 1-D
        # probs so the ndim==2 branch was dead code).  Collapse to one call;
        # expected_calibration_error already handles both shapes internally.
        results["ece"] = expected_calibration_error(
            y_true_arr, probs, n_bins, task_type="binary"
        )
    else:  # multiclass
        results["ece"] = expected_calibration_error(
            y_true_arr, probs, n_bins, task_type="multiclass"
        )
        results["classwise_ece"] = classwise_ece(y_true_arr, probs, n_bins)

    results["reliability_diagram"] = compute_reliability(
        y_true_arr, probs, n_bins, task_type=task_type
    )

    results["brier"] = brier_score(y_true_arr, probs, task_type)

    if T is not None:
        results["temperature"] = T

    return results


def fit_calibration(
    val_logits: np.ndarray,
    val_y_true: Iterable,
    task_type: str,
) -> Optional[float]:
    """Convenience wrapper exposing temperature fitting for the fit-then-apply flow."""
    if task_type not in ("binary", "multiclass", "multilabel"):
        raise ValueError(f"Invalid task_type: {task_type}")

    return fit_temperature(np.asarray(val_logits, dtype=float), np.asarray(val_y_true), task_type)


# =========================================================
# VECTOR TEMPERATURE (per-label T for multilabel) — Section 9
# =========================================================

class VectorTemperatureScaler:
    """One temperature per multilabel column.

    Section 9: a single scalar T over an N-label sigmoid head squeezes very
    different per-label calibration regimes through one knob, which both
    misses easy wins and re-distorts already-calibrated columns. Fitting one
    T per label restores the per-column degrees of freedom while staying
    in the same fit-on-val / apply-on-test contract as ``fit_temperature``.
    """

    def __init__(self, max_iter: int = 50):
        self.max_iter = int(max_iter)
        self.temperatures_: Optional[np.ndarray] = None  # shape (L,)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "VectorTemperatureScaler":
        logits_arr = np.asarray(logits, dtype=float)
        labels_arr = np.asarray(labels)
        if logits_arr.ndim != 2 or labels_arr.ndim != 2:
            raise ValueError("VectorTemperatureScaler expects 2D logits/labels (N, L)")
        if logits_arr.shape != labels_arr.shape:
            raise ValueError(
                f"shape mismatch: logits {logits_arr.shape} vs labels {labels_arr.shape}"
            )

        n_labels = logits_arr.shape[1]
        out = np.ones(n_labels, dtype=float)
        for i in range(n_labels):
            col_logits = logits_arr[:, i]
            col_labels = labels_arr[:, i]
            # Per-label single-T fit reuses the binary path so the clamp /
            # LBFGS contract is identical to the global temperature fit.
            try:
                out[i] = float(fit_temperature(col_logits, col_labels, "binary"))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Per-label T-fit failed for label %d: %s", i, exc)
                out[i] = 1.0
        self.temperatures_ = out
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.temperatures_ is None:
            raise RuntimeError("VectorTemperatureScaler must be fit before transform")
        arr = np.asarray(logits, dtype=float)
        if arr.shape[1] != self.temperatures_.shape[0]:
            raise ValueError(
                f"logits last dim {arr.shape[1]} != fitted T length {self.temperatures_.shape[0]}"
            )
        denom = np.clip(self.temperatures_, EPS, None)
        return arr / denom[None, :]


# =========================================================
# PLATT + ISOTONIC BASELINES — Section 9
# =========================================================

class PlattCalibrator:
    """Logistic-regression (Platt) calibration on top of a binary score.

    Section 9: temperature scaling is a strict subset of Platt (it can only
    rescale, not shift). On binary heads with a prior shift this baseline
    reliably outperforms scalar T, so we expose it as an alternative the
    pipeline can pick when a fitted T leaves significant ECE on the table.
    """

    def __init__(self) -> None:
        self.model_: Optional[LogisticRegression] = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        y = np.asarray(labels).reshape(-1).astype(int)
        if len(np.unique(y)) < 2:
            logger.warning("PlattCalibrator: single-class fit; identity calibrator used")
            self.model_ = None
            return self
        self.model_ = LogisticRegression(solver="lbfgs", max_iter=200)
        self.model_.fit(s, y)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float).reshape(-1, 1)
        if self.model_ is None:
            return expit(s.reshape(-1))
        return self.model_.predict_proba(s)[:, 1]


class IsotonicCalibrator:
    """Non-parametric isotonic-regression calibration (binary).

    Section 9: as a baseline alongside Platt, isotonic captures any monotone
    distortion (S-curves, plateaus) without assuming a sigmoid shape. It
    should be preferred to Platt when the validation set is large enough
    (>~1k samples) to avoid overfitting the step function.
    """

    def __init__(self) -> None:
        self.model_: Optional[IsotonicRegression] = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        s = np.asarray(scores, dtype=float).reshape(-1)
        y = np.asarray(labels).reshape(-1).astype(int)
        if len(np.unique(y)) < 2:
            logger.warning("IsotonicCalibrator: single-class fit; identity used")
            self.model_ = None
            return self
        self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.model_.fit(s, y)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, dtype=float).reshape(-1)
        if self.model_ is None:
            return s
        return self.model_.transform(s)


__all__ = [
    "IsotonicCalibrator",
    "PlattCalibrator",
    "TemperatureScaler",
    "VectorTemperatureScaler",
    "apply_temperature",
    "brier_score",
    "classwise_ece",
    "compute_calibration",
    "compute_reliability",
    "expected_calibration_error",
    "extract_confidence",
    "fit_calibration",
    "fit_temperature",
    "multilabel_ece",
    "sigmoid",
    "softmax",
]
