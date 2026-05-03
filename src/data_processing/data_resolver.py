"""
Resolve relative dataset paths against an environment-configurable base
directory. Splits are configurable (defaults to train/val/test).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence


DEFAULT_REQUIRED_SPLITS = ("train", "val", "test")


def resolve_data_config(
    config: Dict[str, Dict[str, str]],
    *,
    env_var: str = "DATA_DIR",
    strict: bool = True,
    required_splits: Sequence[str] = DEFAULT_REQUIRED_SPLITS,
) -> Dict[str, Dict[str, Path]]:
    """
    Resolve dataset paths for all tasks and splits.

    Args:
        config: ``{"bias": {"train": "...", "val": "...", "test": "..."}, ...}``
        env_var: env var holding the base directory (optional)
        strict: raise if any file is missing
        required_splits: splits that MUST be present (default train/val/test)
    """
    base_dir = os.environ.get(env_var, "")
    base_path = Path(base_dir) if base_dir else None

    resolved: Dict[str, Dict[str, Path]] = {}

    for task, split_map in config.items():
        if not isinstance(split_map, dict):
            raise ValueError(f"{task} config must be dict")

        resolved[task] = {}
        for split in required_splits:
            if split not in split_map:
                raise ValueError(f"{task} missing required split: {split}")

            raw_path = Path(split_map[split])
            path = (
                (base_path / raw_path).resolve()
                if base_path
                else raw_path.resolve()
            )

            if strict and not path.exists():
                raise FileNotFoundError(
                    f"[{task}][{split}] File not found: {path}"
                )

            resolved[task][split] = path

    return resolved


def resolve_path(
    path: str | Path,
    *,
    env_var: str = "DATA_DIR",
    strict: bool = True,
) -> Path:
    base_dir = os.environ.get(env_var, "")
    base_path = Path(base_dir) if base_dir else None

    p = Path(path)
    resolved = (base_path / p).resolve() if base_path else p.resolve()

    if strict and not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    return resolved


# NOTE: ``pretty_print_config`` was removed (UNUSED-D5 in audit pass v3).
# It was a debug helper that called ``print`` (not the logger), had no
# call-sites in production code, and confused readers into thinking it
# was an officially-supported way to dump config. If you need to inspect
# a resolved data config interactively, just ``logger.info(resolved)``
# from a REPL — ``Path`` objects render cleanly under repr().
