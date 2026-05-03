import torch

from src.training.checkpointing import load_checkpoint, list_checkpoints, save_checkpoint


def test_save_and_load_uncompressed(tmp_path):
    model = torch.nn.Linear(4, 2)
    save_dir = tmp_path / "checkpoint-1"
    path = save_checkpoint(model, checkpoint_dir=save_dir, epoch=3, step=10)
    assert path.exists()
    out = load_checkpoint(model, checkpoint_dir=save_dir)
    assert out["epoch"] == 3
    assert out["step"] == 10


def test_list_checkpoints_includes_compressed(tmp_path):
    model = torch.nn.Linear(2, 1)
    save_checkpoint(
        model, checkpoint_dir=tmp_path / "checkpoint-1", use_compression=True
    )
    found = list_checkpoints(tmp_path)
    assert len(found) == 1
