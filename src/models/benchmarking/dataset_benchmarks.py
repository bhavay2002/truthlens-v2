from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class BenchmarkResult:
    task_name: str
    latency_mean: float
    latency_std: float
    throughput: float
    memory_usage: float
    num_samples: int
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "latency_mean": self.latency_mean,
            "latency_std": self.latency_std,
            "throughput": self.throughput,
            "memory_usage": self.memory_usage,
            "num_samples": self.num_samples,
            "extra_metrics": self.extra_metrics,
        }


# =========================================================
# BENCHMARK RUNNER
# =========================================================

class BenchmarkRunner:
    """
    Runs performance benchmarks on models or pipelines.

    Supports:
        - latency measurement
        - throughput
        - GPU memory tracking
        - custom metrics
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        warmup_steps: int = 10,
        measure_steps: int = 50,
    ) -> None:

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps

    # =====================================================
    # MEMORY UTILS
    # =====================================================

    def _get_memory_usage(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)  # MB
        return 0.0

    def _reset_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # =====================================================
    # LATENCY
    # =====================================================

    def _measure_latency(
        self,
        fn: Callable,
        inputs: List[Any],
    ) -> List[float]:

        timings = []

        for inp in inputs:
            start = time.perf_counter()
            fn(inp)
            end = time.perf_counter()
            timings.append(end - start)

        return timings

    # =====================================================
    # MAIN BENCHMARK
    # =====================================================

    def run(
        self,
        task_name: str,
        fn: Callable,
        inputs: List[Any],
        extra_metrics_fn: Optional[Callable[[List[Any]], Dict[str, float]]] = None,
    ) -> BenchmarkResult:

        logger.info(f"[BENCHMARK] Running: {task_name}")

        # Warmup
        for i in range(min(self.warmup_steps, len(inputs))):
            fn(inputs[i])

        # Reset memory stats
        self._reset_memory()

        # Measure latency
        timings = self._measure_latency(
            fn,
            inputs[: self.measure_steps],
        )

        timings_np = np.asarray(timings)

        latency_mean = float(np.mean(timings_np))
        latency_std = float(np.std(timings_np))

        total_time = float(np.sum(timings_np))
        throughput = float(len(timings_np) / max(total_time, 1e-12))

        memory_usage = float(self._get_memory_usage())

        extra_metrics = {}
        if extra_metrics_fn is not None:
            try:
                extra_metrics = extra_metrics_fn(inputs)
            except Exception as e:
                logger.warning(f"Extra metrics failed: {e}")

        result = BenchmarkResult(
            task_name=task_name,
            latency_mean=latency_mean,
            latency_std=latency_std,
            throughput=throughput,
            memory_usage=memory_usage,
            num_samples=len(timings_np),
            extra_metrics=extra_metrics,
        )

        logger.info(f"[BENCHMARK] Done: {task_name}")

        return result

    # =====================================================
    # MULTI-TASK BENCHMARK
    # =====================================================

    def run_multiple(
        self,
        tasks: Dict[str, Dict[str, Any]],
    ) -> Dict[str, BenchmarkResult]:
        """
        tasks format:
        {
            "task_name": {
                "fn": callable,
                "inputs": [...],
                "extra_metrics_fn": optional
            }
        }
        """

        results = {}

        for name, cfg in tasks.items():
            result = self.run(
                task_name=name,
                fn=cfg["fn"],
                inputs=cfg["inputs"],
                extra_metrics_fn=cfg.get("extra_metrics_fn"),
            )
            results[name] = result

        return results

    # =====================================================
    # SUMMARY
    # =====================================================

    @staticmethod
    def summarize(results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:

        summary = {}

        for name, res in results.items():
            summary[name] = res.to_dict()

        return summary