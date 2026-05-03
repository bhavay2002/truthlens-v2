import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.utils.json_utils import append_json, load_json, save_json


def test_save_load_json_roundtrip(tmp_path: Path):
    target = tmp_path / "data.json"
    payload = {"a": 1, "b": "x"}
    save_json(payload, target)
    loaded = load_json(target)
    assert loaded == payload


def test_append_json_new_file(tmp_path: Path):
    target = tmp_path / "items.json"
    append_json({"id": 1}, target)
    append_json({"id": 2}, target)
    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 2


def test_append_json_requires_list_file(tmp_path: Path):
    target = tmp_path / "items.json"
    target.write_text(json.dumps({"not": "a_list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a list"):
        append_json({"id": 1}, target)


def test_append_json_concurrent_smoke(tmp_path: Path):
    target = tmp_path / "concurrent.json"

    def worker(i: int):
        append_json({"id": i}, target)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(worker, range(30)))

    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ids = sorted(item["id"] for item in data)
    assert ids == list(range(30))
