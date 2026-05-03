from __future__ import annotations

import logging
import gc
import time
from typing import Callable, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold, KFold

from src.config.task_config import get_task_type
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


# =========================================================
# METRIC RESOLUTION
# =========================================================

def resolve_metric(
    task: str,
    metrics: Dict[str, Any],
    strategy: str = "auto",
    default: float = float("inf"),
) -> float:

    if strategy == "auto":
        task_type = get_task_type(task)

        if task_type == "multilabel":
            keys = ["micro_f1", "eval_micro_f1"]
        elif task_type == "multiclass":
            keys = ["accuracy", "eval_accuracy"]
        else:
            keys = ["f1", "eval_f1"]

        keys += ["val_loss", "eval_loss"]

    else:
        keys = [strategy]

    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except Exception:
                continue

    logger.warning("[%s] metric not found, using default=%s", task, default)
    return default


# =========================================================
# SPLITS (STRATIFIED + SAFE)
# =========================================================

def build_splits(
    df: pd.DataFrame,
    label_column: str,
    n_splits: int,
    seed: int,
):
    """
    Build CV split indices.

    EDGE-1: ``StratifiedKFold`` requires a 1-D ``y``. Multi-label tasks
    (e.g. emotion has 20 ``emotion_<label>`` columns) have no single
    ``label_column`` to stratify on, and supplying a non-existent column
    used to crash the CV pipeline.  We now:
      * fall back to plain ``KFold`` when ``label_column`` is missing
        (with a warning), so multi-label / regression CV still runs;
      * fall back to plain ``KFold`` when stratification fails because
        the smallest class has fewer than ``n_splits`` members
        (the typical low-resource-task failure mode).
    """

    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2 (got {n_splits})")
    if len(df) < n_splits:
        raise ValueError(
            f"Cannot make {n_splits} folds from {len(df)} rows"
        )

    if label_column not in df.columns:
        logger.warning(
            "[build_splits] label_column=%r not found — falling back "
            "to KFold (no stratification). This is expected for "
            "multi-label / regression tasks.",
            label_column,
        )
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(df))

    y = df[label_column].values

    try:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(df, y))
    except ValueError as exc:
        logger.warning(
            "[build_splits] StratifiedKFold failed (%s) — falling back "
            "to KFold. Smallest class likely has < n_splits members.",
            exc,
        )
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(kf.split(df))


# =========================================================
# MAIN CV (TRAINER-BASED)
# =========================================================

def cross_validate_task(
    *,
    task: str,
    df: pd.DataFrame,
    create_trainer_fn: Callable,
    params: Dict[str, Any],
    label_column: str = "label",
    n_splits: int = 5,
    seed: int = 42,
    metric_strategy: str = "auto",
    return_fold_details: bool = True,
) -> Dict[str, Any]:

    set_seed(seed)

    splits = build_splits(df, label_column, n_splits, seed)

    fold_results: List[Dict[str, Any]] = []
    scores: List[float] = []

    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):

        logger.info("CV | task=%s | fold=%d/%d", task, fold_id, n_splits)

        # N-MED-3: Previously every fold trained with the same global seed
        # set ONCE before the loop. Because torch / numpy / python rng
        # state advances during fold 1, fold 2's training started from
        # fold-1's leftover entropy — which (a) is reproducible only at
        # the granularity of the entire CV run, not per fold, and (b)
        # means the rng correlation between folds inflated the apparent
        # variance reduction. Reseed PER FOLD with a derived seed so each
        # fold is independently reproducible AND independent of the
        # others' rng consumption.
        set_seed(seed + fold_id)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        start = time.time()

        try:
            # -------------------------
            # CREATE TRAINER
            # -------------------------
            trainer = create_trainer_fn(
                task=task,
                train_df=train_df,
                val_df=val_df,
                params=params,
            )

            # -------------------------
            # TRAIN
            # -------------------------
            trainer.train()

            # -------------------------
            # EVALUATE
            # -------------------------
            with torch.no_grad():
                metrics = trainer.evaluate()

            score = resolve_metric(task, metrics, metric_strategy)

            duration = time.time() - start

            scores.append(score)

            fold_result = {
                "fold": fold_id,
                "score": score,
                "metrics": metrics,
                "time": duration,
            }

            if return_fold_details:
                fold_results.append(fold_result)

            logger.info(
                "[%s] Fold %d | score=%.4f | time=%.2fs",
                task,
                fold_id,
                score,
                duration,
            )

        except Exception:
            logger.exception("Fold %d failed", fold_id)

        finally:
            # EDGE-6: ``del trainer`` previously NameError'd whenever
            # ``create_trainer_fn`` itself raised before the binding was
            # made (the ``except Exception: pass`` swallowed the NameError
            # but masked the smell).  Probe ``locals()`` first so the
            # cleanup path is exception-free in both code paths.
            if "trainer" in locals():
                del trainer

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # -----------------------------------------------------
    # POST-CHECK
    # -----------------------------------------------------

    if not scores:
        raise RuntimeError("All CV folds failed")

    scores_np = np.array(scores, dtype=float)

    return {
        "task": task,
        "folds": fold_results if return_fold_details else None,
        "mean": float(scores_np.mean()),
        "std": float(scores_np.std()),
        "min": float(scores_np.min()),
        "max": float(scores_np.max()),
        "num_successful_folds": len(scores),
        "num_failed_folds": n_splits - len(scores),
    }


# =========================================================
# MULTI-TASK CV
# =========================================================

def cross_validate_all_tasks(
    *,
    datasets: Dict[str, pd.DataFrame],
    create_trainer_fn: Callable,
    params: Dict[str, Any],
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:

    results: Dict[str, Any] = {}

    for task, df in datasets.items():

        logger.info("==== CV: %s ====", task)

        results[task] = cross_validate_task(
            task=task,
            df=df,
            create_trainer_fn=create_trainer_fn,
            params=params,
            n_splits=n_splits,
            seed=seed,
        )

    return results


# =========================================================
# DASHBOARD
# =========================================================

def build_dashboard(results: Dict[str, Any]) -> Dict[str, Any]:

    global_scores = [
        r["mean"]
        for r in results.values()
        if r.get("mean") is not None
    ]

    return {
        "tasks": {
            task: {
                "mean": res.get("mean"),
                "std": res.get("std"),
                "folds": res.get("num_successful_folds"),
            }
            for task, res in results.items()
        },
        "global": {
            "mean": float(np.mean(global_scores)) if global_scores else None,
            "std": float(np.std(global_scores)) if global_scores else None,
        },
    }