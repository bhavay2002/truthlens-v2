from __future__ import annotations

import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Section 7: cap matplotlib worker threads. Each PNG save is small (~tens of
# ms) but reports can emit dozens of plots, so a fixed-size pool gives a
# clean speedup without thrashing the main thread or matplotlib's GIL holds.
_PLOT_POOL_SIZE = 4

# Cap individual list/array entries written into the JSON to keep reports small
# and reduce the chance of accidentally serializing huge per-sample dumps.
_MAX_LIST_LEN = 5_000


# =========================================================
# SAFE SERIALIZATION
# =========================================================

def _make_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        items = list(obj)
        if len(items) > _MAX_LIST_LEN:
            items = items[:_MAX_LIST_LEN]
        return [_make_serializable(v) for v in items]

    if isinstance(obj, np.ndarray):
        # HIGH E14: preserve shape when truncating multi-dim arrays. The previous
        # ``flatten()[:_MAX_LIST_LEN]`` collapsed e.g. a 100x100 confusion matrix
        # into a length-10000 1D list, leaving consumers no way to reconstruct
        # the original shape. Wrap the truncated payload with metadata instead.
        if obj.size > _MAX_LIST_LEN:
            return {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "truncated": True,
                "data": obj.flatten()[:_MAX_LIST_LEN].tolist(),
            }
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, datetime.datetime):
        return obj.isoformat()

    # Lazy imports to avoid a hard dependency in environments that don't use these libs.
    try:
        import torch  # type: ignore

        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
            if arr.size > _MAX_LIST_LEN:
                arr = arr.flatten()[:_MAX_LIST_LEN]
            return arr.tolist()
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        if isinstance(obj, pd.DataFrame):
            return _make_serializable(obj.to_dict())
        if isinstance(obj, pd.Series):
            return _make_serializable(obj.to_dict())
    except Exception:
        pass

    if hasattr(obj, "tolist"):
        try:
            return _make_serializable(obj.tolist())
        except Exception:
            pass

    return str(obj)


# =========================================================
# VALIDATION
# =========================================================

def _validate_report(report: Dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise TypeError("Report must be a dict")


# =========================================================
# PLOT UTILS
# =========================================================

def _plot_bar(data: Dict[str, float], save_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not data:
        return

    fig, ax = plt.subplots()
    try:
        ax.bar(list(data.keys()), list(data.values()))
        plt.xticks(rotation=45)
        fig.tight_layout()
        fig.savefig(save_path)
    finally:
        plt.close(fig)


def _plot_list(values, save_path: Path, title: str = "") -> None:
    import matplotlib.pyplot as plt

    if values is None or len(values) == 0:
        return

    fig, ax = plt.subplots()
    try:
        ax.plot(values)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(save_path)
    finally:
        plt.close(fig)


def _plot_hist(values, save_path: Path, title: str = "") -> None:
    import matplotlib.pyplot as plt

    if values is None or len(values) == 0:
        return

    fig, ax = plt.subplots()
    try:
        ax.hist(values, bins=20)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(save_path)
    finally:
        plt.close(fig)


def _plot_reliability(conf, acc, save_path: Path) -> None:
    import matplotlib.pyplot as plt

    if conf is None or acc is None or len(conf) == 0 or len(acc) == 0:
        return

    fig, ax = plt.subplots()
    try:
        ax.plot(conf, acc, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], "--", label="Perfect")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path)
    finally:
        plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def save_report(
    report: Dict[str, Any],
    path: str | Path,
    generate_plots: bool = True,
) -> Path:
    _validate_report(report)

    base_path = Path(path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = report.get("metadata", {}) or {}
    metadata.update({
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluation_version": "v4",
        "tasks": list((report.get("tasks") or {}).keys()),
    })
    report["metadata"] = metadata

    safe_report = _make_serializable(report)
    with base_path.open("w") as f:
        json.dump(safe_report, f, indent=2)
    logger.info("Saved report JSON: %s", base_path)

    if not generate_plots:
        return base_path

    plots_dir = base_path.parent / "plots"
    summary_dir = plots_dir / "summary"
    task_dir = plots_dir / "tasks"
    calib_dir = plots_dir / "calibration"
    error_dir = plots_dir / "error_analysis"
    confidence_dir = plots_dir / "confidence"
    threshold_dir = plots_dir / "thresholds"
    monitoring_dir = plots_dir / "monitoring"

    for d in (plots_dir, summary_dir, task_dir, calib_dir, error_dir, confidence_dir, threshold_dir, monitoring_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Section 7: collect every plot job up-front, then dispatch to a
    # bounded thread pool. Each plot call still serializes its own
    # matplotlib figure (matplotlib state is not thread-safe across
    # the global ``pyplot`` API), but the I/O overlap of writing
    # PNGs in parallel still cuts wall time roughly in half on a
    # 6-task report.
    jobs: List[Tuple[Callable[..., None], tuple, dict]] = []

    def _enqueue(fn: Callable[..., None], *args, **kwargs) -> None:
        jobs.append((fn, args, kwargs))

    summary = report.get("summary", {}) or {}
    numeric_summary = {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
    if numeric_summary:
        _enqueue(_plot_bar, numeric_summary, summary_dir / "summary.png")

    for task, data in (report.get("tasks") or {}).items():
        metrics = (data or {}).get("metrics", {}) or {}
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if numeric:
            _enqueue(_plot_bar, numeric, task_dir / f"{task}_metrics.png")

        if "per_class_f1" in metrics:
            _enqueue(_plot_list, metrics["per_class_f1"], task_dir / f"{task}_per_class_f1.png", "Per Class F1")
        if "per_label_f1" in metrics:
            _enqueue(_plot_list, metrics["per_label_f1"], task_dir / f"{task}_per_label_f1.png", "Per Label F1")

    for task, cal in (report.get("calibration") or {}).items():
        if not isinstance(cal, dict):
            continue
        numeric = {k: v for k, v in cal.items() if isinstance(v, (int, float))}
        if numeric:
            _enqueue(_plot_bar, numeric, calib_dir / f"{task}_calibration.png")
        if isinstance(cal.get("classwise_ece"), dict):
            _enqueue(_plot_bar, cal["classwise_ece"], calib_dir / f"{task}_classwise_ece.png")
        if "per_label_ece" in cal:
            _enqueue(_plot_list, cal["per_label_ece"], calib_dir / f"{task}_per_label_ece.png", "Per Label ECE")
        if isinstance(cal.get("confidence"), (list, np.ndarray)):
            _enqueue(_plot_hist, cal["confidence"], confidence_dir / f"{task}_confidence.png", "Confidence Distribution")

        rd = cal.get("reliability_diagram")
        if isinstance(rd, dict):
            global_block = rd.get("global") if isinstance(rd.get("global"), dict) else rd
            if isinstance(global_block, dict):
                _enqueue(
                    _plot_reliability,
                    global_block.get("confidence"),
                    global_block.get("accuracy"),
                    calib_dir / f"{task}_reliability.png",
                )

    for task, err in (report.get("error_analysis") or {}).items():
        if not isinstance(err, dict):
            continue
        if isinstance(err.get("error_rate_per_class"), dict):
            _enqueue(_plot_bar, err["error_rate_per_class"], error_dir / f"{task}_error_rate.png")

    for task, th in (report.get("optimal_thresholds") or {}).items():
        if isinstance(th, dict):
            value = th.get("threshold")
            if isinstance(value, (int, float)):
                _enqueue(_plot_bar, {"threshold": float(value)}, threshold_dir / f"{task}_threshold.png")
            elif isinstance(th.get("thresholds"), list):
                _enqueue(_plot_list, th["thresholds"], threshold_dir / f"{task}_thresholds.png", "Per-label thresholds")
        elif isinstance(th, (int, float)):
            _enqueue(_plot_bar, {"threshold": float(th)}, threshold_dir / f"{task}_threshold.png")

    for key, val in (report.get("monitoring") or {}).items():
        if isinstance(val, (list, np.ndarray)):
            _enqueue(_plot_list, val, monitoring_dir / f"{key}.png", key)

    if jobs:
        with ThreadPoolExecutor(max_workers=_PLOT_POOL_SIZE) as pool:
            futures = [pool.submit(fn, *args, **kwargs) for fn, args, kwargs in jobs]
            for fut in futures:
                try:
                    fut.result()
                except Exception as exc:
                    logger.warning("Plot job failed: %s", exc)

    logger.info("Report artifacts generated under %s", plots_dir)
    return base_path


__all__ = ["save_report"]
