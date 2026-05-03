"""
File: evaluate_model.py
Location: src/evaluation/

Top-level numpy-friendly :func:`evaluate` entry point used by tests, notebooks
and the saved-model report path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from scipy.special import expit, softmax as scipy_softmax
from transformers import AutoTokenizer

from src.config.task_config import TASK_CONFIG, get_task_type
from src.evaluation.metrics_engine import (
    compute_classification_metrics,
    compute_multilabel_metrics,
)
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
# MODEL PREDICT
# =========================================================

def _predict_model(
    model,
    texts: List[str],
    task: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
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


# =========================================================
# POSTPROCESS
# =========================================================

def _postprocess_logits(logits: np.ndarray, task_type: str, *, threshold: float = 0.5):
    """Convert logits → ``(preds, probs)``.

    HIGH E3: drop the hand-rolled softmax/sigmoid in favor of
    ``scipy.special.softmax`` / ``expit`` (no torch round-trip in the hot path).
    CRIT E7: accept ``threshold`` so binary/multilabel callers can apply a
    fitted threshold instead of the hard-coded ``0.5``.
    """
    arr = np.asarray(logits, dtype=float)

    if task_type == "multiclass":
        probs = scipy_softmax(arr, axis=-1)
        preds = np.argmax(probs, axis=1)
        return preds.astype(int), probs

    if task_type == "binary":
        if arr.ndim == 2 and arr.shape[1] == 2:
            probs = scipy_softmax(arr, axis=-1)[:, 1]
        else:
            probs = expit(arr).reshape(-1)
        preds = (probs >= threshold).astype(int)
        return preds, probs

    if task_type == "multilabel":
        probs = expit(arr)
        preds = (probs >= threshold).astype(int)
        return preds, probs

    raise ValueError(f"Unknown task_type: {task_type}")


# =========================================================
# TASK INFERENCE
# =========================================================

def _infer_task_type(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Best-effort inference used only when ``task`` is not provided.

    CRIT E5: this heuristic is intrinsically unreliable — a 3-class slice that
    happens to contain only labels {0, 1} flips to ``binary``. Callers that
    know the task should always pass ``task`` (or explicitly resolve via
    ``get_task_type``) so this fallback isn't exercised.
    """
    if y_true.ndim == 2:
        return "multilabel"
    classes = set(np.unique(y_true).tolist()) | set(np.unique(y_pred).tolist())
    if classes.issubset({0, 1}) and len(classes) <= 2:
        return "binary"
    return "multiclass"


# =========================================================
# CORE EVALUATION
# =========================================================

def evaluate(
    y_true: Iterable,
    y_pred: Optional[Iterable] = None,
    y_proba: Optional[Iterable] = None,
    *,
    model=None,
    texts: Optional[List[str]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate predictions against labels.

    The ``task`` argument is optional — when omitted the task type is inferred
    from the shape and value range of ``y_true`` / ``y_pred``. ``y_proba`` is
    used only for probability-based metrics (e.g. ROC-AUC) and is fed straight
    through; it is never re-activated when ``y_pred`` is already a hard label.
    """
    # =====================================================
    # MODEL MODE
    # =====================================================
    task_type: Optional[str] = None
    num_labels: Optional[int] = None

    if task is not None:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {task}")
        task_type = get_task_type(task)
        num_labels = TASK_CONFIG[task]["num_labels"]

    if model is not None:
        if texts is None or tokenizer is None or task is None:
            raise ValueError("model mode requires texts + tokenizer + task")

        logits = _predict_model(model=model, texts=texts, task=task, tokenizer=tokenizer)
        y_pred, y_proba = _postprocess_logits(logits, task_type)

    # =====================================================
    # NUMPY MODE
    # =====================================================
    y_true_arr = np.asarray(y_true)
    if y_true_arr.size == 0:
        raise ValueError("y_true cannot be empty")

    if y_pred is None:
        raise ValueError("y_pred must be provided if model is None")

    y_pred_arr = np.asarray(y_pred)
    y_proba_arr = np.asarray(y_proba, dtype=float) if y_proba is not None else None

    if y_proba_arr is not None and y_proba_arr.shape[0] != y_true_arr.shape[0]:
        raise ValueError(
            f"y_proba length {y_proba_arr.shape[0]} does not match y_true length {y_true_arr.shape[0]}"
        )

    if task_type is None:
        task_type = _infer_task_type(y_true_arr, y_pred_arr)

    is_multilabel = task_type == "multilabel"

    # If config declares multilabel but y_true is 1D (caller passed class
    # indices instead of multi-hot vectors), fall back to multiclass evaluation
    # so the metrics layer never sees a shape it can't handle.
    if is_multilabel and y_true_arr.ndim == 1:
        is_multilabel = False
        task_type = "multiclass"

    # Convert probability/logit matrices → class indices for classification.
    # Applied unconditionally here and again right before the shape check as a
    # belt-and-suspenders guard in case any code path re-assigns y_pred_arr.
    if not is_multilabel and y_pred_arr.ndim == 2:
        y_pred_arr = np.argmax(y_pred_arr, axis=1)

    y_true_arr = y_true_arr.reshape(-1) if not is_multilabel else y_true_arr

    # =====================================================
    # SHAPE VALIDATION
    # =====================================================
    if not is_multilabel:
        # Final defensive conversion — catches any re-assignment after the
        # earlier argmax block (e.g. from calibration or postprocessing).
        if y_pred_arr.ndim == 2:
            y_pred_arr = np.argmax(y_pred_arr, axis=1)
        if y_pred_arr.ndim != 1:
            raise ValueError(
                f"y_pred must be 1D for binary/multiclass tasks (got shape {y_pred_arr.shape})"
            )
        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
            )
    else:
        if y_pred_arr.shape != y_true_arr.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true_arr.shape} vs y_pred {y_pred_arr.shape}"
            )

    # =====================================================
    # METRICS
    # =====================================================
    if is_multilabel:
        metrics = compute_multilabel_metrics(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_proba=y_proba_arr,
        )
    else:
        # CRIT E5: forward the authoritative ``task_type`` so the metrics layer
        # doesn't fall back to its own ``{0, 1}``-based binary heuristic.
        metrics = compute_classification_metrics(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_proba=y_proba_arr,
            task_type=task_type,
        )

    # =====================================================
    # DATASET STATS
    # =====================================================
    if not is_multilabel:
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
            "label_density": float(np.mean(y_true_arr)),
            "density": float(np.mean(y_true_arr)),
        }

    # =====================================================
    # FINAL RESULT — flatten metric keys into the top-level
    # dict for backward compatibility with consumers that
    # expect ``results["accuracy"]`` / ``results["f1"]`` etc.
    # =====================================================
    result: Dict[str, Any] = {
        "task": task,
        "task_type": task_type,
        "metrics": metrics,
        "dataset_stats": dataset_stats,
    }
    result.update(metrics)
    if num_labels is not None:
        result["num_labels"] = num_labels
    return result
