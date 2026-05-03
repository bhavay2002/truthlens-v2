from src.models.utils.model_utils import save_model, load_model
from pathlib import Path


def test_model_save_load(tmp_path: Path):

    model = {"type": "test"}

    path = tmp_path / "model.pkl"

    save_model(model, path)
    loaded = load_model(path)

    assert loaded == model