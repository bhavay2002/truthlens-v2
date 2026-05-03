from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.training.checkpointing import CHECKPOINT_FILE, list_checkpoints, load_checkpoint


def _touch_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 1}, path / CHECKPOINT_FILE)


def test_list_checkpoints_orders_hf_style_steps_numerically(tmp_path: Path) -> None:
    root = tmp_path / "ckpts"
    _touch_checkpoint(root / "checkpoint-10")
    _touch_checkpoint(root / "checkpoint-2")
    _touch_checkpoint(root / "checkpoint-100")

    ordered = list_checkpoints(root)

    assert [p.name for p in ordered] == ["checkpoint-2", "checkpoint-10", "checkpoint-100"]


def test_load_checkpoint_falls_back_to_latest_nested_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "ckpts"
    model = nn.Linear(4, 2)
    saved_state = {k: v.clone() for k, v in model.state_dict().items()}

    _touch_checkpoint(root / "checkpoint-2")
    latest = root / "checkpoint-10"
    latest.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": saved_state,
            "epoch": 3,
            "step": 120,
        },
        latest / CHECKPOINT_FILE,
    )

    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)

    restored = load_checkpoint(model, checkpoint_dir=root)

    assert restored["epoch"] == 3
    assert restored["step"] == 120
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved_state[key])


def test_load_checkpoint_rejects_missing_model_state_dict(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "flat"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 1, "step": 5}, checkpoint_dir / CHECKPOINT_FILE)

    model = nn.Linear(4, 2)

    with pytest.raises(RuntimeError, match="Checkpoint missing required key: model_state_dict"):
        load_checkpoint(model, checkpoint_dir=checkpoint_dir)
