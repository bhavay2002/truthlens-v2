#src\models\training\task_scheduler.py
from __future__ import annotations
 
import logging
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class TaskSchedulerConfig:
    strategy: str = "round_robin"
    task_weights: Optional[Dict[str, float]] = None
    temperature: float = 1.0
    seed: int = 42
    ema_alpha: float = 0.1
    min_prob: float = 1e-3


# =========================================================
# TASK SCHEDULER
# =========================================================

class TaskScheduler:

    def __init__(
        self,
        tasks: List[str],
        config: Optional[TaskSchedulerConfig] = None,
    ):

        if not tasks:
            raise ValueError("tasks cannot be empty")

        self.tasks = list(tasks)
        self.config = config or TaskSchedulerConfig()

        # CFG-6: with a single task every strategy collapses to "always
        # return that task". Warn loudly when the caller has wired a
        # non-trivial strategy (round_robin is the harmless default), so
        # they don't silently believe e.g. ``adaptive`` is doing something.
        # Cache the trivial answer and short-circuit ``next_task`` so we
        # skip the rng / softmax work on every step.
        self._single_task: Optional[str] = (
            self.tasks[0] if len(self.tasks) == 1 else None
        )
        if self._single_task is not None and self.config.strategy != "round_robin":
            logger.warning(
                "TaskScheduler received a single task (%r) with strategy=%r; "
                "this strategy is a no-op for one task and the scheduler will "
                "always return %r.",
                self._single_task, self.config.strategy, self._single_task,
            )

        self.rng = random.Random(self.config.seed)

        self._rr_index = 0

        self.task_weights = self._init_weights()

        # EMA tracking (FIXED)
        self._ema_losses: Dict[str, float] = {
            t: 1.0 for t in self.tasks
        }

        logger.info(
            "TaskScheduler initialized | strategy=%s | tasks=%s",
            self.config.strategy,
            self.tasks,
        )

    # =====================================================
    # MAIN
    # =====================================================

    def next_task(self) -> str:

        # CFG-6: single-task fast path (skip rng / softmax / strategy dispatch).
        if self._single_task is not None:
            return self._single_task

        strategy = self.config.strategy

        if strategy == "round_robin":
            return self._round_robin()

        elif strategy == "random":
            return self._random()

        elif strategy == "weighted":
            return self._weighted()

        elif strategy == "adaptive":
            return self._adaptive()

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # =====================================================
    # STRATEGIES
    # =====================================================

    def _round_robin(self) -> str:
        task = self.tasks[self._rr_index]
        self._rr_index = (self._rr_index + 1) % len(self.tasks)
        return task

    def _random(self) -> str:
        return self.rng.choice(self.tasks)

    def _weighted(self) -> str:
        weights = [self.task_weights[t] for t in self.tasks]
        return self.rng.choices(self.tasks, weights=weights, k=1)[0]

    def _adaptive(self) -> str:
        # N-LOW-7: CONVENTION — this is a "focus on the hard tasks" sampler.
        # The softmax is taken DIRECTLY over the raw EMA losses (no negation),
        # which means HIGHER loss → HIGHER selection probability.  The
        # rationale is that a task whose loss is still high is one where
        # the joint encoder still has room to learn; up-weighting it spends
        # more gradient steps on the head that matters most.  The opposite
        # convention ("sample easier tasks more often") would negate
        # ``scores`` here.  Keep this comment in sync with ``_decide_action``
        # / GradNorm if the policy ever flips.
        scores = [self._ema_losses[t] for t in self.tasks]

        # softmax with temperature
        scaled = [s / self.config.temperature for s in scores]
        exp = [math.exp(x) for x in scaled]
        total = sum(exp)

        if total == 0:
            return self._random()

        probs = [e / total for e in exp]

        # min probability floor
        probs = [max(p, self.config.min_prob) for p in probs]

        return self.rng.choices(self.tasks, weights=probs, k=1)[0]

    # =====================================================
    # UPDATE
    # =====================================================

    def update_losses(self, task_losses: Dict[str, float]):

        for task, loss in task_losses.items():

            if task not in self._ema_losses:
                continue

            if not (loss > 0 and loss < float("inf")):
                continue

            prev = self._ema_losses[task]

            self._ema_losses[task] = (
                self.config.ema_alpha * loss
                + (1 - self.config.ema_alpha) * prev
            )

    # =====================================================
    # INIT
    # =====================================================

    def _init_weights(self) -> Dict[str, float]:

        if self.config.task_weights:
            weights = dict(self.config.task_weights)
        else:
            weights = {t: 1.0 for t in self.tasks}

        # normalize
        s = sum(weights.values())
        if s > 0:
            weights = {k: v / s for k, v in weights.items()}

        return weights

    # =====================================================
    # UTIL
    # =====================================================

    def get_weights(self) -> Dict[str, float]:
        return dict(self.task_weights)

    def get_adaptive_scores(self) -> Dict[str, float]:
        return dict(self._ema_losses)