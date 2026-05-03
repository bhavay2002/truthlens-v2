from __future__ import annotations

import json
import logging
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)


# =========================================================
# UTILS
# =========================================================

def _ensure_streamlit():
    if st is None:
        raise RuntimeError("Install streamlit")


def load_report(path):
    with open(path) as f:
        return json.load(f)


# =========================================================
# TASK SELECTOR
# =========================================================

def select_task(tasks):
    return st.sidebar.selectbox("Task", list(tasks.keys()))


# =========================================================
# METRICS
# =========================================================

def render_metrics(task_data: Dict):

    metrics = task_data.get("metrics", {})

    st.subheader("Metrics")

    df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    st.dataframe(df, use_container_width=True)

    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    if numeric:
        fig, ax = plt.subplots()
        ax.bar(numeric.keys(), numeric.values())
        plt.xticks(rotation=45)
        st.pyplot(fig)


# =========================================================
# DATASET STATS
# =========================================================

def render_dataset_stats(task_data):

    stats = task_data.get("dataset_stats", {})

    if not stats:
        return

    st.subheader("Dataset Statistics")
    st.json(stats)


# =========================================================
# CONFUSION MATRIX
# =========================================================

def render_confusion(task_data):

    metrics = task_data.get("metrics", {})
    # HIGH E13: metrics_engine writes the matrix under the flat key
    # ``confusion_matrix``, not under ``confusion.matrix``. The dashboard
    # previously read the wrong key so the matrix never rendered.
    confusion_matrix = metrics.get("confusion_matrix") or (
        metrics.get("confusion", {}) if isinstance(metrics.get("confusion"), dict) else {}
    ).get("matrix")

    if confusion_matrix is None:
        return

    st.subheader("Confusion Matrix")

    matrix = np.array(confusion_matrix)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="Blues")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center")

    fig.colorbar(im)
    st.pyplot(fig)


# =========================================================
#  RELIABILITY DIAGRAM
# =========================================================

def render_reliability(cal):

    rd = cal.get("reliability_diagram")

    if not rd:
        return

    st.subheader("Reliability Diagram")

    conf = rd.get("confidence")
    acc = rd.get("accuracy")

    if conf and acc:
        fig, ax = plt.subplots()

        ax.plot(conf, acc, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], "--", label="Perfect")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.legend()

        st.pyplot(fig)


# =========================================================
#  CONFIDENCE DISTRIBUTION
# =========================================================

def render_confidence(cal):

    conf = cal.get("confidence")

    if not conf:
        return

    st.subheader("Confidence Distribution")

    fig, ax = plt.subplots()
    ax.hist(conf, bins=20)
    st.pyplot(fig)


# =========================================================
# CALIBRATION
# =========================================================

def render_calibration(report, task):

    cal = report.get("calibration", {}).get(task)

    if not cal:
        return

    st.subheader("Calibration")

    for k, v in cal.items():
        if isinstance(v, (int, float)):
            st.metric(k, f"{v:.4f}")

    render_confidence(cal)        #  NEW
    render_reliability(cal)       #  NEW


# =========================================================
#  ERROR ANALYSIS
# =========================================================

def render_error_analysis(report, task):

    err = report.get("error_analysis", {}).get(task)

    if not err:
        return

    st.subheader("Error Analysis")

    st.json(err)

    if "error_rate_per_class" in err:
        fig, ax = plt.subplots()
        ax.bar(err["error_rate_per_class"].keys(),
               err["error_rate_per_class"].values())
        st.pyplot(fig)


# =========================================================
# THRESHOLD
# =========================================================

def render_thresholds(report, task):

    th = report.get("optimal_thresholds", {}).get(task)

    if not th:
        return

    st.subheader("Optimal Threshold")

    if isinstance(th, dict):
        if isinstance(th.get("threshold"), (int, float)):
            st.metric("Threshold", f"{th['threshold']:.4f}")
        if isinstance(th.get("score"), (int, float)):
            st.metric("Score", f"{th['score']:.4f}")
        if isinstance(th.get("thresholds"), list):
            df = pd.DataFrame(
                {"label": list(range(len(th["thresholds"]))), "threshold": th["thresholds"]}
            )
            st.dataframe(df, use_container_width=True)
    elif isinstance(th, (int, float)):
        st.metric("Threshold", f"{th:.4f}")
    else:
        st.json(th)


# =========================================================
# UNCERTAINTY
# =========================================================

def render_uncertainty(report, task):

    unc = report.get("uncertainty", {}).get(task)

    if not unc:
        return

    st.subheader("Uncertainty")

    df = pd.DataFrame(unc.items(), columns=["Metric", "Value"])
    st.dataframe(df)


# =========================================================
# CORRELATION
# =========================================================

def render_correlation(report):

    corr = report.get("task_correlation")

    if not corr:
        return

    st.subheader("Task Correlation")

    try:
        if isinstance(corr, dict) and corr and all(isinstance(v, dict) for v in corr.values()):
            matrix = pd.DataFrame(corr).astype(float)
        else:
            matrix = pd.DataFrame(corr)
    except Exception as exc:
        logger.warning("Could not render correlation matrix: %s", exc)
        st.json(corr)
        return

    if matrix.empty:
        return

    fig, ax = plt.subplots()
    cax = ax.matshow(matrix.values, cmap="coolwarm")
    fig.colorbar(cax)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(list(matrix.columns), rotation=45)
    ax.set_yticklabels(list(matrix.index))

    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# ADVANCED
# =========================================================

def render_advanced(report):

    adv = report.get("advanced_analysis")

    if not adv:
        return

    st.subheader("Advanced Analysis")
    st.json(adv)


# =========================================================
# MAIN
# =========================================================

def launch_dashboard(report_path):

    _ensure_streamlit()

    st.set_page_config(layout="wide")

    report = load_report(report_path)

    st.title("TruthLens AI Dashboard")

    tasks = report["tasks"]
    task = select_task(tasks)

    task_data = tasks[task]

    col1, col2 = st.columns(2)

    with col1:
        render_metrics(task_data)
        render_dataset_stats(task_data)
        render_calibration(report, task)
        render_thresholds(report, task)

    with col2:
        render_uncertainty(report, task)
        render_confusion(task_data)
        render_error_analysis(report, task)

    st.divider()

    render_correlation(report)

    st.divider()

    render_advanced(report)