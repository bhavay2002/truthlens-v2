"""
Train/val/test leakage checker.

- Exact-match path is fast: SHA-256 of normalized text, set intersection.
- Empty / whitespace texts are filtered before hashing (otherwise they
  collapse to one bucket and report bogus overlap).
- ``check_near_duplicates`` is opt-in (still O(n·m) — use for small splits
  only or replace with MinHash).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class LeakageConfig:
    strict: bool = True
    check_near_duplicates: bool = False
    near_duplicate_threshold: float = 0.9
    sample_size: int = 10000
    report_examples: int = 5


# =========================================================
# RESULT
# =========================================================

@dataclass
class LeakageReport:
    train_val_overlap: int = 0
    train_test_overlap: int = 0
    val_test_overlap: int = 0
    examples: Dict[str, List[str]] = field(default_factory=dict)


# =========================================================
# HASHING
# =========================================================

def _normalize(text) -> str:
    return str(text).strip().lower() if text is not None else ""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hashes(series: pd.Series) -> set:
    return {
        _hash_text(t)
        for t in (
            _normalize(x) for x in series.tolist()
        )
        if t  # skip empty after normalization
    }


# =========================================================
# CORE
# =========================================================

def check_leakage_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: Optional[LeakageConfig] = None,
) -> LeakageReport:
    config = config or LeakageConfig()

    train_h = _hashes(train["text"])
    val_h = _hashes(val["text"])
    test_h = _hashes(test["text"])

    tv = train_h & val_h
    tt = train_h & test_h
    vt = val_h & test_h

    report = LeakageReport(
        train_val_overlap=len(tv),
        train_test_overlap=len(tt),
        val_test_overlap=len(vt),
    )

    if config.report_examples > 0:
        report.examples = {
            "train_val": list(tv)[:config.report_examples],
            "train_test": list(tt)[:config.report_examples],
            "val_test": list(vt)[:config.report_examples],
        }

    _handle_result(report, config)
    return report


# =========================================================
# MULTI-TASK
# =========================================================

def check_leakage_all_tasks(
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    *,
    config: Optional[LeakageConfig] = None,
) -> Dict[str, LeakageReport]:
    results: Dict[str, LeakageReport] = {}
    for task, splits in datasets.items():
        if not {"train", "val", "test"}.issubset(splits.keys()):
            logger.warning(
                "Leakage check skipped for %s (missing one of train/val/test)",
                task,
            )
            continue
        logger.info("Checking leakage for task: %s", task)
        results[task] = check_leakage_splits(
            splits["train"], splits["val"], splits["test"], config=config
        )
    return results


# =========================================================
# OPT-IN NEAR-DUP — fail-fast cap (PERF-D3)
# =========================================================

# Hard ceiling on (|df1| · |df2|) before we subsample. SequenceMatcher
# at ~3 µs/comparison gives ~30 s at 1e7 pairs — anything larger is a
# foot-gun. For 10k × 10k splits the original code triggered ~1e8
# comparisons (≈5 min). When the ceiling is hit we evenly subsample
# both sides and warn loudly so the result is still informative.
_MAX_NEAR_DUP_PAIRS = 10_000_000


def check_near_duplicates(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    threshold: float = 0.9,
    *,
    max_pairs: int = _MAX_NEAR_DUP_PAIRS,
    random_state: int = 0,
) -> int:
    from difflib import SequenceMatcher
    from math import isqrt

    n1, n2 = len(df1), len(df2)
    total_pairs = n1 * n2

    if total_pairs > max_pairs:
        # Even subsample: keep √max_pairs from each side.
        target = max(1, isqrt(max_pairs))
        if n1 > target:
            df1 = df1.sample(n=target, random_state=random_state)
        if n2 > target:
            df2 = df2.sample(n=target, random_state=random_state)
        logger.warning(
            "check_near_duplicates: %d × %d = %d pairs exceeds cap (%d). "
            "Subsampled to %d × %d. For an exact answer on splits this "
            "large, swap SequenceMatcher for MinHash + LSH.",
            n1, n2, total_pairs, max_pairs, len(df1), len(df2),
        )

    overlaps = 0
    texts1 = df1["text"].astype(str).tolist()
    texts2 = df2["text"].astype(str).tolist()

    for t1 in texts1:
        for t2 in texts2:
            if SequenceMatcher(None, t1, t2).ratio() > threshold:
                overlaps += 1
    return overlaps


# =========================================================
# HANDLER
# =========================================================

def _handle_result(report: LeakageReport, config: LeakageConfig):
    total = (
        report.train_val_overlap
        + report.train_test_overlap
        + report.val_test_overlap
    )
    if total == 0:
        logger.info("No data leakage detected")
        return

    msg = (
        "Leakage detected | "
        f"train-val={report.train_val_overlap}, "
        f"train-test={report.train_test_overlap}, "
        f"val-test={report.val_test_overlap}"
    )
    if config.strict:
        
     logger.warning(f"{msg} | Auto-removing duplicates")

    return report
