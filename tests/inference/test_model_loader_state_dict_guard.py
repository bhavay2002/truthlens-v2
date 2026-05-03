import pytest
from pathlib import Path

from src.inference.model_loader import ModelLoader


def test_load_torch_model_rejects_state_dict(monkeypatch, tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_file = models_dir / "bias_model.pt"
    model_file.write_bytes(b"x")

    loader = ModelLoader(str(models_dir), device="cpu")

    monkeypatch.setattr(
        "src.inference.model_loader.torch.load",
        lambda *args, **kwargs: {"weight": 1},
    )

    with pytest.raises(RuntimeError, match="State dict found"):
        loader._load_torch_model(model_file)
