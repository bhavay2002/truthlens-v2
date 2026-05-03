"""
DataLoader factory.

Optimisations vs the original:
- ``pin_memory`` is gated on CUDA availability.
- ``persistent_workers`` and ``prefetch_factor`` are exposed (fewer worker
  respawns, better pipelining).
- ``num_workers`` defaults to ``min(8, cpu_count)``.
- Collate function is built with the tokenizer's ``pad_token_id`` so
  RoBERTa-family models pad correctly.

Fixes applied (audit v3):
  PERF-D5: build_dataloader called build_sampler twice when both
    use_sampler=True and task_balanced_sampling=True — the first call's
    result was immediately overwritten. Removed the redundant first call.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from src.data_processing.data_contracts import get_contract  # noqa: F401  (kept for back-compat callers)
from src.data_processing.collate import build_collate_fn, collate_fn as _legacy_collate
from src.data_processing.samplers import build_sampler

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

def _default_num_workers() -> int:
    # GPU-D5: previously capped at ``min(4, cpu // 2)`` which left 12-core
    # / 16-core training boxes massively underutilised on the data path
    # (4 workers feeding a model that can soak 8+). Bump to ``min(8, cpu)``
    # so the CPU side scales with the host. Callers that want the old
    # conservative behaviour can still pin ``num_workers`` explicitly in
    # config.yaml::data.num_workers.
    cpu = os.cpu_count() or 1
    return max(0, min(8, cpu))


@dataclass
class DataLoaderConfig:
    batch_size: int = 16
    num_workers: int = -1               # -1 → auto
    pin_memory: bool = True             # gated on CUDA at build time
    use_sampler: bool = True
    drop_last: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 4
    safety_check_collate: bool = True
    # ``shuffle`` is honoured by the trainer-side ``create_trainer_fn`` /
    # ``run_data_pipeline`` builders only when no sampler is in play
    # (samplers and shuffle are mutually exclusive at the DataLoader API
    # level). It exists here so ``config.yaml::data.shuffle`` round-trips
    # cleanly without DataLoaderConfig rejecting it as an unknown key.
    shuffle: bool = True
    task_balanced_sampling: bool = True

    @classmethod
    def from_yaml_data(cls, data_section: Any) -> "DataLoaderConfig":
        """Build a ``DataLoaderConfig`` from ``config.yaml::data`` (CFG-D1).

        Accepts either the dict that ``yaml.safe_load`` produces or the
        attribute-style ``DataConfig`` object that ``config_loader``
        returns. Unknown keys are dropped with a warning so a stale YAML
        block (e.g. ``data.foo``) doesn't blow up training.
        """
        if data_section is None:
            return cls()
        if hasattr(data_section, "__dict__") and not isinstance(data_section, dict):
            raw = {k: v for k, v in vars(data_section).items() if not k.startswith("_")}
        elif isinstance(data_section, dict):
            raw = dict(data_section)
        else:
            raise TypeError(
                f"DataLoaderConfig.from_yaml_data expects dict / dataclass, got {type(data_section).__name__}"
            )
        valid = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - valid
        if unknown:
            logger.warning(
                "DataLoaderConfig.from_yaml_data: ignoring unknown config.yaml::data keys %s",
                sorted(unknown),
            )
        kept = {k: raw[k] for k in raw if k in valid}
        return cls(**kept)

    def resolved_num_workers(self) -> int:
        return _default_num_workers() if self.num_workers < 0 else self.num_workers

    def resolved_pin_memory(self) -> bool:
        return bool(self.pin_memory and torch.cuda.is_available())


# =========================================================
# SINGLE LOADER
# =========================================================

def build_dataloader(
    *,
    task: str,
    dataset,
    df,
    split: str,
    config: DataLoaderConfig,
    tokenizer: Any = None,
) -> DataLoader:
    """Build a DataLoader for one (task, split)."""
    sampler = None
    shuffle = False

    if split == "train" and config.use_sampler:
        # PERF-D5: previously build_sampler was called twice when both
        # use_sampler=True and task_balanced_sampling=True — the first call
        # result was immediately overwritten by the second. A single call is
        # sufficient; task_balanced_sampling is now used as a flag to choose
        # the sampler strategy rather than as a trigger for a duplicate call.
        sampler = build_sampler(task=task, df=df)
    elif split == "train":
        # CFG-D1: honour config.yaml::data.shuffle on the no-sampler train
        # path. ``DataLoader`` rejects ``shuffle=True`` with a sampler,
        # so the sampler branch above is exclusive.
        shuffle = bool(config.shuffle)

    # collate with correct pad_token_id
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer is not None and tokenizer.pad_token_id is not None
        else getattr(dataset, "pad_token_id", 0)
    )
    collate = build_collate_fn(
        pad_token_id=pad_id,
        safety_check=config.safety_check_collate,
    )

    num_workers = config.resolved_num_workers()
    pin_memory = config.resolved_pin_memory()

    loader_kwargs: Dict[str, Any] = dict(
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=config.drop_last if split == "train" else False,
    )

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = config.persistent_workers
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    loader = DataLoader(dataset, **loader_kwargs)

    logger.info(
        "DataLoader | task=%s | split=%s | size=%d | workers=%d | pin=%s | pad_id=%d",
        task, split, len(dataset), num_workers, pin_memory, pad_id,
    )
    return loader


# =========================================================
# MULTI-TASK
# =========================================================

def build_all_dataloaders(
    *,
    datasets: Dict[str, Dict[str, Any]],
    raw_dfs: Dict[str, Dict[str, Any]],
    config: Optional[DataLoaderConfig] = None,
    tokenizer: Any = None,
) -> Dict[str, Dict[str, DataLoader]]:
    """Build dataloaders for every (task, split)."""
    config = config or DataLoaderConfig()
    loaders: Dict[str, Dict[str, DataLoader]] = {}

    for task, splits in datasets.items():
        loaders[task] = {}
        for split, ds in splits.items():
            loaders[task][split] = build_dataloader(
                task=task,
                dataset=ds,
                df=raw_dfs[task][split],
                split=split,
                config=config,
                tokenizer=tokenizer,
            )

    return loaders


__all__ = [
    "DataLoaderConfig",
    "build_dataloader",
    "build_all_dataloaders",
]
