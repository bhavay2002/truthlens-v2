from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.checkpointing.checkpoint_manager import CheckpointManager, get_last_checkpoint


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _mkdir_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 1, "model_state_dict": {}}, path / "checkpoint.pt")


def test_list_checkpoints_sorts_and_filters_valid_names(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _mkdir_checkpoint(checkpoint_dir / "checkpoint-20")
    _mkdir_checkpoint(checkpoint_dir / "checkpoint-5")
    _mkdir(checkpoint_dir / "checkpoint-invalid")
    _mkdir(checkpoint_dir / "something-else")

    manager = CheckpointManager(checkpoint_dir)
    checkpoints = manager.list_checkpoints()

    assert [path.name for path in checkpoints] == ["checkpoint-5", "checkpoint-20"]


def test_get_latest_checkpoint_returns_highest_step(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _mkdir_checkpoint(checkpoint_dir / "checkpoint-100")
    _mkdir_checkpoint(checkpoint_dir / "checkpoint-250")

    manager = CheckpointManager(checkpoint_dir)
    latest = manager.get_latest_checkpoint()

    assert latest is not None
    assert latest.name == "checkpoint-250"
    assert get_last_checkpoint(checkpoint_dir) == latest


def test_cleanup_old_checkpoints_keeps_latest_n(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _mkdir_checkpoint(checkpoint_dir / "checkpoint-10")
    _mkdir_checkpoint(checkpoint_dir / "checkpoint-20")
    _mkdir_checkpoint(checkpoint_dir / "checkpoint-30")

    manager = CheckpointManager(checkpoint_dir)
    manager.cleanup_old_checkpoints(max_checkpoints=2)

    remaining = [path.name for path in manager.list_checkpoints()]
    assert remaining == ["checkpoint-20", "checkpoint-30"]


def test_cleanup_old_checkpoints_rejects_invalid_limit(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path / "checkpoints")

    with pytest.raises(ValueError):
        manager.cleanup_old_checkpoints(max_checkpoints=0)

