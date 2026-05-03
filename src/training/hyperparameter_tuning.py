from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

import optuna
import pandas as pd

from src.training.cross_validation import cross_validate_task
from src.training.experiment_tracker import (
    ExperimentTracker,
    ExperimentTrackerConfig,
)
from src.utils.seed_utils import set_seed
from src.config.task_config import get_task_type

logger = logging.getLogger(__name__)


# =========================================================
# TRACKING
#
# N-LOW-5: Previously this module shipped its OWN ``init_tracking`` /
# ``finalize_tracking`` / ``log_trial`` functions that talked to MLflow
# and W&B directly. That duplicated everything ``ExperimentTracker``
# already does (rank guards, distributed safety, error swallowing,
# config / metric flattening) and meant Optuna runs were the only
# pipeline that BYPASSED the tracker's distributed-safe ``_safe`` /
# ``_is_main`` paths — leading to duplicate logs in DDP runs and
# dropped metrics in MLflow when the run wasn't yet started.
#
# Route Optuna through ``ExperimentTracker(group=...)`` so the trial
# runs share an experiment group and inherit all of the tracker's
# safety and reproducibility guarantees.
# =========================================================


def _build_tracker(task: str, backend: str = "none") -> ExperimentTracker:
    config = ExperimentTrackerConfig(
        backend=backend,
        project_name="TruthLens",
        run_name=f"{task}_{int(time.time())}",
        group=f"tune_{task}",
        tags={"task": task, "phase": "tuning"},
    )
    return ExperimentTracker(config)


# =========================================================
# OBJECTIVE
# =========================================================

def build_objective(
    *,
    task: str,
    df: pd.DataFrame,
    create_trainer_fn: Callable,
    multi_objective: bool,
    tracker: ExperimentTracker,
    base_params: Optional[Dict[str, Any]] = None,
):

    # N-HIGH-1: Previously the objective constructed ``params`` from ONLY
    # the four trial-suggested keys (lr, batch_size, epochs, weight_decay)
    # and passed that to ``cross_validate_task`` → ``create_trainer_fn``.
    # That silently DROPPED every other config key the caller had set
    # (gradient_accumulation_steps, max_grad_norm, monitor_metric,
    # log_every_steps, scheduler config, ...).  ``base_params`` is the
    # carrier for those caller-defined defaults; the trial suggestions
    # OVERRIDE them per-trial.
    base_params = dict(base_params or {})

    def objective(trial: optuna.Trial):

        # -------------------------
        # SEED
        # -------------------------
        seed = 42 + trial.number
        set_seed(seed)

        # -------------------------
        # SEARCH SPACE
        # -------------------------
        trial_params = {
            "lr": trial.suggest_float("lr", 1e-6, 5e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "epochs": trial.suggest_int("epochs", 2, 6),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        }

        # N-HIGH-1: trial values WIN over base_params (so a tunable key
        # always reflects the latest suggestion), but every non-tuned
        # caller default flows through unchanged.
        params = {**base_params, **trial_params}

        start = time.time()

        try:
            # -------------------------
            # CROSS VALIDATION
            # -------------------------
            cv_result = cross_validate_task(
                task=task,
                df=df,
                create_trainer_fn=create_trainer_fn,
                params=params,
            )

            score = cv_result["mean"]
            std = cv_result["std"]

        except Exception:
            logger.exception("Trial %d failed", trial.number)
            raise optuna.TrialPruned()

        duration = time.time() - start

        # -------------------------
        # REPORT FOR PRUNING
        # -------------------------
        trial.report(score, step=0)

        if trial.should_prune():
            raise optuna.TrialPruned()

        # -------------------------
        # LOGGING (via ExperimentTracker — N-LOW-5)
        # -------------------------
        tracker.log_params({f"trial/{k}": v for k, v in trial_params.items()})
        tracker.log_metrics(
            {
                "trial/score": score,
                "trial/std": std,
                "trial/time": duration,
            },
            step=trial.number,
        )

        return (score, -std) if multi_objective else score

    return objective


# =========================================================
# STUDY
# =========================================================

def _resolve_direction(task: str) -> str:
    # BUG-8: cross_validate_task returns the model's primary metric
    # (accuracy / micro-F1 for classification, MSE for regression).
    # Classification metrics must be MAXIMISED — Optuna's previous
    # default of "minimize" silently selected the worst trials.
    try:
        ttype = str(get_task_type(task)).replace("_", "").lower()
    except Exception:
        ttype = ""

    if ttype in {"multiclass", "multilabel", "binary"}:
        return "maximize"
    return "minimize"  # regression / loss-style metrics


def create_study(
    *,
    multi_objective: bool,
    storage: Optional[str],
    task: str,
):

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
    )

    pruner = optuna.pruners.MedianPruner()

    score_direction = _resolve_direction(task)

    if multi_objective:
        # objective returns (score, -std) — both should be MAXIMISED
        # when score is a classification metric, otherwise both MINIMISED.
        return optuna.create_study(
            directions=[score_direction, score_direction],
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )

    return optuna.create_study(
        direction=score_direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )


# =========================================================
# MAIN
# =========================================================

def tune_task(
    *,
    task: str,
    df: pd.DataFrame,
    create_trainer_fn: Callable,
    n_trials: int,
    multi_objective: bool = False,
    n_jobs: int = 1,
    storage: Optional[str] = None,
    base_params: Optional[Dict[str, Any]] = None,
    tracker_backend: str = "none",
):

    tracker = _build_tracker(task, backend=tracker_backend)

    study = create_study(
        multi_objective=multi_objective,
        storage=storage,
        task=task,
    )

    objective = build_objective(
        task=task,
        df=df,
        create_trainer_fn=create_trainer_fn,
        multi_objective=multi_objective,
        tracker=tracker,
        base_params=base_params,
    )

    logger.info(
        "Starting tuning | task=%s | trials=%d",
        task,
        n_trials,
    )

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
        )
    finally:
        tracker.finish()

    # -------------------------
    # RESULTS
    # -------------------------

    if not multi_objective:
        return {
            "task": task,
            "best_params": study.best_params,
            "best_score": float(study.best_value),
        }

    return {
        "task": task,
        "pareto_front": [
            {"params": t.params, "values": t.values}
            for t in study.best_trials
        ],
    }


# =========================================================
# MULTI-TASK
# =========================================================

def tune_all_tasks(
    *,
    datasets: Dict[str, pd.DataFrame],
    create_trainer_fn: Callable,
    n_trials: int,
    multi_objective: bool = False,
    n_jobs: int = 1,
    storage: Optional[str] = None,
    base_params: Optional[Dict[str, Any]] = None,
    tracker_backend: str = "none",
):

    results: Dict[str, Any] = {}

    for task, df in datasets.items():

        logger.info("Tuning task: %s", task)

        results[task] = tune_task(
            task=task,
            df=df,
            create_trainer_fn=create_trainer_fn,
            n_trials=n_trials,
            multi_objective=multi_objective,
            n_jobs=n_jobs,
            storage=storage,
            base_params=base_params,
            tracker_backend=tracker_backend,
        )

    return results
