from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.utils.model_utils import load_model, save_model
from src.utils.device_utils import (
    device_summary,
    get_device,
    move_to_device,
)
from src.utils.helper_functions import create_folder
from src.utils.json_utils import append_json, load_json, save_json
from src.utils.seed_utils import get_seed_state, seed_worker, set_seed
from src.utils.settings import load_settings
from src.utils.time_utils import Timer, measure_runtime, timestamp


class TestSettings:
    def test_load_settings_has_expected_types(self) -> None:
        settings = load_settings()
        assert isinstance(settings.model.path, Path)
        assert isinstance(settings.paths.training_log_path, Path)
        assert settings.training.cross_validation_splits >= 2
        assert settings.training.optuna_trials >= 1

    def test_api_settings_are_strings(self) -> None:
        settings = load_settings()
        assert isinstance(settings.api.title, str)
        assert isinstance(settings.api.version, str)
        assert len(settings.api.title) > 0

    def test_inference_settings_present(self) -> None:
        settings = load_settings()
        assert settings.inference.batch_size > 0
        assert settings.inference.device in {"auto", "cpu", "cuda", "mps"}

    def test_training_settings_values(self) -> None:
        settings = load_settings()
        assert settings.training.batch_size > 0
        assert settings.training.epochs >= 1
        assert 0.0 < settings.training.learning_rate < 1.0


class TestFilesystemUtils:
    def test_create_folder_returns_path(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "folder"
        created = create_folder(target)
        assert created == target
        assert target.exists()
        assert target.is_dir()

    def test_create_folder_idempotent(self, tmp_path: Path) -> None:
        target = tmp_path / "folder"
        create_folder(target)
        create_folder(target)
        assert target.exists()


class TestModelSerialization:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.joblib"
        payload = {"a": 1, "b": [1, 2, 3]}
        save_path = save_model(payload, model_file)
        loaded = load_model(save_path)
        assert save_path == model_file
        assert loaded == payload

    def test_save_returns_path(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.joblib"
        result = save_model({"key": "value"}, model_file)
        assert isinstance(result, Path)

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            load_model(tmp_path / "nonexistent.joblib")


class TestJsonUtils:
    def test_save_and_load_json_roundtrip(self, tmp_path: Path) -> None:
        data = {"key": "value", "number": 42}
        path = tmp_path / "data.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_save_json_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "file.json"
        save_json({"x": 1}, path)
        assert path.exists()

    def test_save_json_non_dict_raises_type_error(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            save_json([1, 2, 3], tmp_path / "file.json")  # type: ignore[arg-type]

    def test_load_json_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "missing.json")

    def test_append_json_creates_list_file(self, tmp_path: Path) -> None:
        path = tmp_path / "entries.json"
        append_json({"entry": 1}, path)
        append_json({"entry": 2}, path)
        with path.open() as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_append_json_non_dict_raises_type_error(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            append_json("not_a_dict", tmp_path / "file.json")  # type: ignore[arg-type]


class TestSeedUtils:
    def test_set_seed_produces_reproducible_numpy(self) -> None:
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_set_seed_produces_reproducible_torch(self) -> None:
        set_seed(99)
        a = torch.rand(4)
        set_seed(99)
        b = torch.rand(4)
        assert torch.allclose(a, b)

    def test_set_seed_non_integer_raises(self) -> None:
        with pytest.raises((TypeError, RuntimeError)):
            set_seed(3.14)  # type: ignore[arg-type]

    def test_get_seed_state_returns_expected_keys(self) -> None:
        set_seed(0)
        state = get_seed_state()
        assert "python_hash_seed" in state
        assert "torch_seed" in state
        assert "cuda_available" in state

    def test_seed_worker_runs_without_error(self) -> None:
        set_seed(42)
        seed_worker(0)


class TestTimeUtils:
    def test_timestamp_format(self) -> None:
        ts = timestamp()
        assert len(ts) == 19
        assert ts[4] == "-" and ts[7] == "-"

    def test_measure_runtime_returns_result_and_elapsed(self) -> None:
        result, elapsed = measure_runtime(lambda: 42)
        assert result == 42
        assert elapsed >= 0.0

    def test_measure_runtime_propagates_exceptions(self) -> None:
        def bad_func():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            measure_runtime(bad_func)

    def test_timer_measures_elapsed_correctly(self) -> None:
        t = Timer()
        t.start()
        time.sleep(0.05)
        elapsed = t.stop()
        assert elapsed >= 0.04

    def test_timer_stop_without_start_raises(self) -> None:
        t = Timer()
        with pytest.raises(RuntimeError, match="not started"):
            t.stop()


class TestDeviceUtils:
    def test_get_device_returns_torch_device(self) -> None:
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_prefer_gpu_false_gives_cpu(self) -> None:
        device = get_device(prefer_gpu=False)
        assert device.type == "cpu"

    def test_device_summary_returns_expected_keys(self) -> None:
        summary = device_summary()
        for key in ("device", "device_name", "gpu_count", "cuda_available"):
            assert key in summary

    def test_move_to_device_tensor(self) -> None:
        device = torch.device("cpu")
        tensor = torch.ones(3)
        moved = move_to_device(tensor, device)
        assert moved.device.type == "cpu"

    def test_move_to_device_dict_of_tensors(self) -> None:
        device = torch.device("cpu")
        batch = {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
        moved = move_to_device(batch, device)
        assert moved["input_ids"].device.type == "cpu"

    def test_move_to_device_none_returns_none(self) -> None:
        assert move_to_device(None, torch.device("cpu")) is None

    def test_move_to_device_list_of_tensors(self) -> None:
        device = torch.device("cpu")
        tensors = [torch.ones(2), torch.zeros(3)]
        moved = move_to_device(tensors, device)
        assert all(t.device.type == "cpu" for t in moved)
