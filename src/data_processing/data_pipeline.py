"""
TruthLens data pipeline orchestrator.

Order of operations:
    1. resolve paths
    2. load + validate + clean (raw)
    3. multitask validation + label analysis
    4. **leakage check on raw splits** (before augmentation)
    5. augmentation (train only)
    6. cache write
    7. profiling
    8. (optional) build datasets + dataloaders

Cache key now incorporates tokenizer / max_length / cleaning /
augmentation config so changing any of them invalidates stale entries.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, Optional

import pandas as pd

from src.data_processing.data_contracts import get_contract, DEFAULT_MAX_LENGTH
from src.data_processing.data_resolver import resolve_data_config
from src.data_processing.data_loader import load_dataframe
from src.data_processing.data_validator import validate_dataframe
from src.data_processing.data_cleaning import clean_for_task, DataCleaningConfig
from src.data_processing.data_augmentation import augment_dataset, AugmentationConfig
from src.data_processing.data_profiler import profile_dataframe
from src.data_processing.leakage_checker import check_leakage_all_tasks

from src.data_processing.dataset_factory import build_all_datasets
from src.data_processing.dataloader_factory import (
    build_all_dataloaders,
    DataLoaderConfig,
)

from src.analysis.label_analysis import analyze_labels, assert_label_health
from src.analysis.multitask_validator import (
    validate_multitask_dataframe,
    assert_multitask_health,
)

from src.data_processing.data_cache import (
    get_cache_key,
    load_cached_datasets,
    save_cached_datasets,
)

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

class DataPipelineConfig:

    def __init__(
        self,
        enable_cleaning: bool = True,
        enable_validation: bool = True,
        enable_augmentation: bool = False,
        enable_profiling: bool = True,
        enable_leakage_check: bool = True,
        enable_multitask_validation: bool = True,
        enable_label_analysis: bool = True,
        enable_cache: bool = True,
        force_rebuild: bool = False,
        max_length: int = DEFAULT_MAX_LENGTH,
        return_offsets_mapping: bool = False,
        cleaning_config: Optional[DataCleaningConfig] = None,
        augmentation_config: Optional[AugmentationConfig] = None,
        dataloader_config: Optional[DataLoaderConfig] = None,
    ):
        self.enable_cleaning = enable_cleaning
        self.enable_validation = enable_validation
        self.enable_augmentation = enable_augmentation
        self.enable_profiling = enable_profiling
        self.enable_leakage_check = enable_leakage_check
        self.enable_multitask_validation = enable_multitask_validation
        self.enable_label_analysis = enable_label_analysis
        self.enable_cache = enable_cache
        self.force_rebuild = force_rebuild
        self.max_length = max_length
        self.return_offsets_mapping = return_offsets_mapping
        self.cleaning_config = cleaning_config or DataCleaningConfig()
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.dataloader_config = dataloader_config or DataLoaderConfig()


# =========================================================
# CACHE-KEY EXTRA
# =========================================================

def _to_plain(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return obj


def _cache_extra(config: DataPipelineConfig, tokenizer) -> Dict[str, Any]:
    extra: Dict[str, Any] = {
        "max_length": config.max_length,
        "cleaning": _to_plain(config.cleaning_config) if config.enable_cleaning else None,
        "augmentation": _to_plain(config.augmentation_config) if config.enable_augmentation else None,
        "validation": config.enable_validation,
        "multitask_validation": config.enable_multitask_validation,
        "label_analysis": config.enable_label_analysis,
    }
    if tokenizer is not None:
        extra["tokenizer"] = {
            "name": getattr(tokenizer, "name_or_path", str(type(tokenizer))),
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "cls": tokenizer.__class__.__name__,
        }
    return extra


# =========================================================
# CORE
# =========================================================

def run_data_pipeline(
    *,
    data_config: Dict[str, Dict[str, str]],
    tokenizer=None,
    build_dataloaders: bool = False,
    config: Optional[DataPipelineConfig] = None,
):
    config = config or DataPipelineConfig()

    # 1) resolve paths first (needed for cache key)
    resolved_paths = resolve_data_config(data_config)

    # 2) cache key includes tokenizer + cleaning + augmentation + max_length
    cache_key = get_cache_key(
        data_config,
        resolved_paths,
        extra=_cache_extra(config, tokenizer),
    )

    raw_datasets: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None

    if config.enable_cache and not config.force_rebuild:
        raw_datasets = load_cached_datasets(cache_key)
        if raw_datasets is not None:
            logger.info(
                "Using cached dataset (augmentation step skipped on cache hit)"
            )

    # =========================================================
    # BUILD FROM SCRATCH
    # =========================================================
    if raw_datasets is None:
        logger.info("Building dataset from scratch")
        raw_datasets = _build_raw_datasets(resolved_paths, config)

        # always run leakage check on RAW (pre-augmentation) splits
        if config.enable_leakage_check:
            check_leakage_all_tasks(raw_datasets)

        # augmentation runs AFTER leakage check (train only)
        if config.enable_augmentation:
            for task, splits in raw_datasets.items():
                if "train" in splits:
                    val_df = splits.get("val")
                    test_df = splits.get("test")
                    splits["train"] = augment_dataset(
                        splits["train"],
                        task=task,
                        config=config.augmentation_config,
                        # Pre-filter augmented candidates against val/test text hashes
                        # so an augmentation op that mutates a train row into a
                        # near-duplicate of a val/test row is rejected and resampled.
                        # (CRIT-D5 / LEAK-D1)
                        held_out_dfs=[d for d in (val_df, test_df) if d is not None],
                    )

            # Defence in depth: re-run the cheap exact-match leakage check on
            # the post-augmentation splits so any candidate that slipped past
            # the per-row pre-filter is still caught before the cache is
            # written. Ensures the cached frames are guaranteed leak-free.
            if config.enable_leakage_check:
                check_leakage_all_tasks(raw_datasets)

        if config.enable_cache:
            save_cached_datasets(raw_datasets, cache_key)
            logger.info("Dataset cached")

        if config.enable_profiling:
            for task in raw_datasets:
                profile_dataframe(raw_datasets[task]["train"], task=task)
    else:
        # cache hit: still run leakage check (cheap; protects against poisoned cache)
        if config.enable_leakage_check:
            check_leakage_all_tasks(raw_datasets)

    # =========================================================
    # RETURN
    # =========================================================
    if not build_dataloaders:
        return raw_datasets

    if tokenizer is None:
        raise ValueError("Tokenizer required for dataloaders")

    datasets = build_all_datasets(
        datasets=raw_datasets,
        tokenizer=tokenizer,
        max_length=config.max_length,
        return_offsets_mapping=config.return_offsets_mapping,
    )

    return build_all_dataloaders(
        datasets=datasets,
        raw_dfs=raw_datasets,
        config=config.dataloader_config,
        tokenizer=tokenizer,
    )


# =========================================================
# RAW BUILD STEP
# =========================================================

def _build_raw_datasets(
    resolved_paths: Dict[str, Dict[str, Any]],
    config: DataPipelineConfig,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    raw_datasets: Dict[str, Dict[str, pd.DataFrame]] = {}

    for task, splits in resolved_paths.items():
        logger.info("Processing task: %s", task)
        contract = get_contract(task)
        raw_datasets[task] = {}

        # analysis modules expect a flat {task: column_name} dict — for
        # multilabel tasks, register one entry per label column.
        if contract.task_type == "classification":
            task_columns = {task: contract.label_columns[0]}
        else:
            task_columns = {
                f"{task}__{c}": c for c in contract.label_columns
            }

        for split, path in splits.items():
            df = load_dataframe(path)

            if config.enable_validation:
                validate_dataframe(df, task=task)

            if config.enable_cleaning:
                df = clean_for_task(df, task, config=config.cleaning_config)

            if config.enable_multitask_validation:
                df, mt_result = validate_multitask_dataframe(
                    df, task_columns=task_columns
                )
                assert_multitask_health(mt_result)

            if config.enable_label_analysis:
                label_result = analyze_labels(df, task_columns=task_columns)
                assert_label_health(
                    label_result,
                    fail_on_imbalance=False,
                    fail_on_rare=False,
                )

            raw_datasets[task][split] = df

    return raw_datasets
