from __future__ import annotations

import logging
from typing import Dict, Any, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


logger = logging.getLogger(__name__)

EPS = 1e-12
ArrayLike = Union[np.ndarray, "torch.Tensor"]


# =========================================================
# UTILS
# =========================================================

def _safe_numpy(x):
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + EPS)


def _sigmoid(x):
    x = np.clip(x, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-x))


def _to_output(arr, like):
    if TORCH_AVAILABLE and isinstance(like, torch.Tensor):
        return torch.from_numpy(arr.astype(np.float32)).to(like.device)
    return arr.astype(np.float32)


# =========================================================
# BASE
# =========================================================

class BaseCalibrator:

    def __init__(self):
        self.fitted = False

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        raise NotImplementedError

    def transform(self, logits: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def fit_transform(self, logits, labels):
        self.fit(logits, labels)
        return self.transform(logits)


# =========================================================
# PASSTHROUGH — used when method="none" or unfitted
# =========================================================

class PassThroughCalibrator(BaseCalibrator):

    def __init__(self):
        super().__init__()
        self.fitted = True

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        self.fitted = True

    def transform(self, logits: ArrayLike) -> ArrayLike:
        arr = _safe_numpy(logits)
        return _to_output(np.clip(arr, 0.0, 1.0), logits)


# =========================================================
# TEMPERATURE SCALING (ROBUST)
# =========================================================

class TemperatureScaler(BaseCalibrator):

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temperature = float(init_temp)

    def fit(self, logits: np.ndarray, labels: np.ndarray):

        logits = _safe_numpy(logits)
        labels = labels.astype(int)

        if logits.ndim != 2:
            raise ValueError("Temperature scaling requires 2D logits")

        T = self.temperature

        for _ in range(100):

            probs = _softmax(logits / max(T, 1e-3))
            grad = self._grad(logits, labels, T)
            T -= 0.05 * grad
            T = max(T, 1e-3)

        self.temperature = float(T)
        self.fitted = True

        logger.info("[Calibration] Temperature fitted: %.4f", self.temperature)

    def _grad(self, logits, labels, T):
        eps = 1e-4
        loss1 = self._loss(logits, labels, T)
        loss2 = self._loss(logits, labels, T + eps)
        return (loss2 - loss1) / eps

    def _loss(self, logits, labels, T):
        probs = _softmax(logits / max(T, 1e-3))
        return -np.mean(np.log(probs[np.arange(len(labels)), labels] + EPS))

    def transform(self, logits: ArrayLike) -> ArrayLike:
        if not self.fitted:
            return PassThroughCalibrator().transform(logits)
        arr = _safe_numpy(logits)
        if arr.ndim < 2:
            return _to_output(np.clip(arr, 0.0, 1.0), logits)
        probs = _softmax(arr / max(self.temperature, 1e-3))
        return _to_output(probs, logits)


# =========================================================
# SIGMOID (MULTILABEL SAFE)
# =========================================================

class SigmoidCalibrator(BaseCalibrator):

    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None

    def fit(self, logits: np.ndarray, labels: np.ndarray):

        logits = _safe_numpy(logits)
        labels = _safe_numpy(labels)

        if logits.ndim == 2:
            self.a = np.zeros(logits.shape[1])
            self.b = np.zeros(logits.shape[1])
            for c in range(logits.shape[1]):
                self.a[c], self.b[c] = self._fit_binary(logits[:, c], labels[:, c])
        else:
            self.a, self.b = self._fit_binary(logits, labels)

        self.fitted = True

    def _fit_binary(self, logits, labels):
        a, b = 1.0, 0.0
        for _ in range(100):
            probs = _sigmoid(a * logits + b)
            error = probs - labels
            a -= 0.1 * np.mean(error * logits)
            b -= 0.1 * np.mean(error)
        return float(a), float(b)

    def transform(self, logits: ArrayLike) -> ArrayLike:
        if not self.fitted:
            return PassThroughCalibrator().transform(logits)
        arr = _safe_numpy(logits)
        if arr.ndim == 2:
            probs = _sigmoid(arr * self.a + self.b)
        else:
            probs = _sigmoid(self.a * arr + self.b)
        return _to_output(probs, logits)


# =========================================================
# ISOTONIC (MULTICLASS EXTENSION)
# =========================================================

class IsotonicCalibrator(BaseCalibrator):

    def __init__(self):
        super().__init__()
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")
        self.models = []

    def fit(self, logits: np.ndarray, labels: np.ndarray):

        logits = _safe_numpy(logits)

        if logits.ndim == 2:
            self.models = []
            for c in range(logits.shape[1]):
                model = IsotonicRegression(out_of_bounds="clip")
                binary = (labels == c).astype(int)
                model.fit(logits[:, c], binary)
                self.models.append(model)
        else:
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(logits, labels)
            self.models = [model]

        self.fitted = True

    def transform(self, logits: ArrayLike) -> ArrayLike:
        if not self.fitted:
            return PassThroughCalibrator().transform(logits)
        arr = _safe_numpy(logits)
        if arr.ndim == 2:
            calibrated = np.zeros_like(arr)
            for c, model in enumerate(self.models):
                calibrated[:, c] = model.transform(arr[:, c])
            calibrated = calibrated / (np.sum(calibrated, axis=1, keepdims=True) + EPS)
        else:
            calibrated = self.models[0].transform(arr)
        return _to_output(calibrated, logits)


# =========================================================
# FACTORY
# =========================================================

def get_calibrator(method: str) -> BaseCalibrator:

    method = method.lower()

    if method == "none":
        return PassThroughCalibrator()

    if method == "temperature":
        return TemperatureScaler()

    if method == "sigmoid":
        return SigmoidCalibrator()

    if method == "isotonic":
        return IsotonicCalibrator()

    raise ValueError(f"Unknown calibration method: {method}")
