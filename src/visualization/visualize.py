from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


# =========================================================
# GLOBAL STYLE (NEW )
# =========================================================

def set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (6, 4),
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })


# =========================================================
# UTILS
# =========================================================

def _ensure_numpy(x: Iterable) -> np.ndarray:
    return np.asarray(x)


def _save_figure(fig: plt.Figure, save_path: Optional[str | Path]):
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved: %s", path)


# =========================================================
# CONFUSION MATRIX
# =========================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[List[str]] = None,
    normalize: bool = False,
    cmap: str = "Blues",
    save_path: Optional[str | Path] = None,
):

    set_plot_style()

    cm = _ensure_numpy(cm)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "g",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    _save_figure(fig, save_path)

    return fig, ax


# =========================================================
# ROC CURVE (UPGRADED )
# =========================================================

def plot_roc_curve(
    y_true: Iterable,
    y_scores: Iterable,
    save_path: Optional[str | Path] = None,
):

    set_plot_style()

    y_true = _ensure_numpy(y_true)
    y_scores = _ensure_numpy(y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "--")

    ax.set_title("ROC Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()

    _save_figure(fig, save_path)

    return fig, ax, roc_auc   #  NEW RETURN


# =========================================================
# PR CURVE
# =========================================================

def plot_precision_recall_curve(
    y_true,
    y_scores,
    save_path=None,
):

    set_plot_style()

    y_true = _ensure_numpy(y_true)
    y_scores = _ensure_numpy(y_scores)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots()

    ax.plot(recall, precision)
    ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    _save_figure(fig, save_path)

    return fig, ax


# =========================================================
# CALIBRATION
# =========================================================

def plot_calibration_curve(
    y_true,
    y_prob,
    n_bins=10,
    save_path=None,
):

    set_plot_style()

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots()

    ax.plot(prob_pred, prob_true, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "--", label="Perfect")

    ax.set_title("Calibration Curve")
    ax.legend()

    _save_figure(fig, save_path)

    return fig, ax


# =========================================================
# TRAINING CURVES
# =========================================================

def plot_training_curves(history: Dict[str, List[float]], save_path=None):

    set_plot_style()

    fig, ax = plt.subplots()

    for key, values in history.items():
        ax.plot(values, label=key)

    ax.set_title("Training Curves")
    ax.legend()

    _save_figure(fig, save_path)

    return fig, ax


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def plot_feature_importance(
    features: List[str],
    scores,
    top_k=20,
    save_path=None,
):

    set_plot_style()

    scores = _ensure_numpy(scores)
    idx = np.argsort(scores)[::-1][:top_k]

    fig, ax = plt.subplots()

    sns.barplot(
        x=scores[idx],
        y=[features[i] for i in idx],
        ax=ax,
    )

    ax.set_title("Feature Importance")

    _save_figure(fig, save_path)

    return fig, ax


# =========================================================
# EMBEDDING PROJECTION (SAFE)
# =========================================================

def plot_embedding_projection(
    embeddings: np.ndarray,
    labels=None,
    method="pca",
    max_samples: int = 5000,   #  NEW SAFETY
    save_path=None,
):

    set_plot_style()

    embeddings = _ensure_numpy(embeddings)

    #  MEMORY SAFETY
    if embeddings.shape[0] > max_samples:
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[idx]
        if labels is not None:
            labels = np.asarray(labels)[idx]

    if method == "pca":
        reducer = PCA(n_components=2)

    elif method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(embeddings) - 1),
            random_state=42,
        )

    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    proj = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots()

    if labels is None:
        ax.scatter(proj[:, 0], proj[:, 1], alpha=0.7)
    else:
        labels = np.asarray(labels)
        for l in np.unique(labels):
            idx = labels == l
            ax.scatter(proj[idx, 0], proj[idx, 1], label=str(l), alpha=0.7)
        ax.legend()

    ax.set_title(f"Embedding ({method.upper()})")

    _save_figure(fig, save_path)

    return fig, ax