import numpy as np
import pytest

from src.visualization.visualize import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_embedding_projection,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
)


def test_plot_confusion_matrix_requires_square():
    with pytest.raises(ValueError, match="must be square"):
        plot_confusion_matrix(np.array([[1, 2, 3], [4, 5, 6]]))


def test_plot_roc_curve_shape_mismatch():
    y_true = np.array([0, 1, 1])
    y_scores = np.array([0.2, 0.7])
    with pytest.raises(ValueError, match="same length"):
        plot_roc_curve(y_true, y_scores)


def test_plot_precision_recall_curve_shape_mismatch():
    y_true = np.array([0, 1, 1])
    y_scores = np.array([0.2, 0.7])
    with pytest.raises(ValueError, match="same length"):
        plot_precision_recall_curve(y_true, y_scores)


def test_plot_calibration_curve_invalid_bins():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.6, 0.2])
    with pytest.raises(ValueError, match="n_bins"):
        plot_calibration_curve(y_true, y_prob, n_bins=1)


def test_plot_feature_importance_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        plot_feature_importance(["a", "b"], [0.1, 0.2], top_k=0)


def test_plot_embedding_projection_label_mismatch():
    emb = np.random.randn(10, 8)
    labels = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="labels length must match"):
        plot_embedding_projection(emb, labels=labels, method="pca")
