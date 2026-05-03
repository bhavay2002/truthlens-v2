"""
test_loader.py
--------------
Loads the 6 real test datasets from data/test/ into a unified format
consumed by the evaluation pipeline.

Expected files (CSV or JSON/JSONL):

  bias.{csv,json,jsonl}
    columns: text, bias_label (0|1)

  ideology.{csv,json,jsonl}
    columns: text, ideology_label (0|1|2)

  propaganda.{csv,json,jsonl}
    columns: text, propaganda_label (0|1)

  frame.{csv,json,jsonl}
    columns: text, CO, EC, HI, MO, RE (each 0|1)

  narrative.{csv,json,jsonl}
    columns: text, hero, villain, victim (each 0|1)
             hero_entities, villain_entities, victim_entities (text, ignored)

  emotion.{csv,json,jsonl}
    columns: text, emotion_0 … emotion_10 (each 0|1)

Returns
-------
Each loader returns:
    texts  : list[str]
    labels : dict[str, np.ndarray]
               key   = TASK_CONFIG task name
               value = (N,)  for multiclass  OR  (N, C)  for multilabel
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# COLUMN SCHEMAS
# ─────────────────────────────────────────────────────────────

# Maps: dataset filename stem → (task_name, schema_type, column_spec)
#
# schema_type:
#   "multiclass" → single label column, returns (N,) int array
#   "multilabel" → ordered list of binary columns, returns (N, C) int array
#
DATASET_SCHEMAS: Dict[str, dict] = {
    "bias": {
        "task": "bias",
        "type": "multiclass",
        "label_col": "bias_label",
    },
    "ideology": {
        "task": "ideology",
        "type": "multiclass",
        "label_col": "ideology_label",
    },
    "propaganda": {
        "task": "propaganda",
        "type": "multiclass",
        "label_col": "propaganda_label",
    },
    "frame": {
        "task": "narrative_frame",
        "type": "multilabel",
        "label_cols": ["CO", "EC", "HI", "MO", "RE"],
    },
    "narrative": {
        "task": "narrative",
        "type": "multilabel",
        "label_cols": ["hero", "villain", "victim"],
        "ignore_cols": ["hero_entities", "villain_entities", "victim_entities"],
    },
    "emotion": {
        "task": "emotion",
        "type": "multilabel",
        "label_cols": [f"emotion_{i}" for i in range(11)],
    },
}


# ─────────────────────────────────────────────────────────────
# FILE READER  (CSV / JSON / JSONL)
# ─────────────────────────────────────────────────────────────

def _read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".json", ".jsonl"):
        try:
            with path.open() as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                return pd.DataFrame([data])
        except json.JSONDecodeError:
            return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported file format: {path.suffix}  (expected .csv / .json / .jsonl)")


def _find_file(directory: Path, stem: str) -> Optional[Path]:
    for ext in (".csv", ".json", ".jsonl"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ─────────────────────────────────────────────────────────────
# SINGLE-DATASET LOADER
# ─────────────────────────────────────────────────────────────

def load_dataset(
    path: Path,
    schema: dict,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Load one test file and return (texts, labels)."""
    df = _read_file(path)

    if "text" not in df.columns:
        raise ValueError(f"{path.name}: missing 'text' column. Found: {list(df.columns)}")

    texts: List[str] = df["text"].astype(str).tolist()
    task = schema["task"]

    if schema["type"] == "multiclass":
        col = schema["label_col"]
        if col not in df.columns:
            raise ValueError(
                f"{path.name}: missing label column '{col}'. Found: {list(df.columns)}"
            )
        labels_arr = df[col].astype(int).values
        return texts, {task: labels_arr}

    if schema["type"] == "multilabel":
        cols = schema["label_cols"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path.name}: missing multilabel columns {missing}. Found: {list(df.columns)}"
            )
        labels_arr = df[cols].astype(int).values
        return texts, {task: labels_arr}

    raise ValueError(f"Unknown schema type: {schema['type']}")


# ─────────────────────────────────────────────────────────────
# MULTI-DATASET LOADER  (loads all 6 at once)
# ─────────────────────────────────────────────────────────────

class TestDataLoader:
    """
    Load all 6 test datasets from a directory.

    Usage
    -----
    loader = TestDataLoader("data/test")
    datasets = loader.load_all()
    # datasets = {
    #   "bias":           (texts, {"bias": (N,)}),
    #   "ideology":       (texts, {"ideology": (N,)}),
    #   "propaganda":     (texts, {"propaganda": (N,)}),
    #   "narrative_frame":(texts, {"narrative_frame": (N,5)}),
    #   "narrative":      (texts, {"narrative": (N,3)}),
    #   "emotion":        (texts, {"emotion": (N,11)}),
    # }
    """

    def __init__(self, test_dir: str | Path):
        self.test_dir = Path(test_dir)
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")

    def load_all(self) -> Dict[str, Tuple[List[str], Dict[str, np.ndarray]]]:
        results = {}
        for stem, schema in DATASET_SCHEMAS.items():
            path = _find_file(self.test_dir, stem)
            if path is None:
                logger.warning("Dataset file not found: %s/{%s}.{csv,json,jsonl}", self.test_dir, stem)
                continue
            try:
                texts, labels = load_dataset(path, schema)
                task = schema["task"]
                results[task] = (texts, labels)
                label_shape = {t: arr.shape for t, arr in labels.items()}
                logger.info("Loaded %-20s  %d samples  labels=%s", task, len(texts), label_shape)
                print(f"  ✓ {stem:<12} → task={task:<18} samples={len(texts)}  label_shape={label_shape}")
            except Exception as exc:
                logger.error("Failed to load %s: %s", path, exc)
                print(f"  ✗ {stem:<12} → ERROR: {exc}")
        return results

    def load_one(self, stem: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
        schema = DATASET_SCHEMAS.get(stem)
        if schema is None:
            raise KeyError(f"Unknown dataset '{stem}'. Valid: {list(DATASET_SCHEMAS)}")
        path = _find_file(self.test_dir, stem)
        if path is None:
            raise FileNotFoundError(
                f"No file for '{stem}' in {self.test_dir}. "
                f"Expected one of: {stem}.csv / {stem}.json / {stem}.jsonl"
            )
        return load_dataset(path, schema)

    def summary(self) -> None:
        """Print what files are present / missing in the test directory."""
        print(f"\nTest directory: {self.test_dir}")
        print("-" * 50)
        for stem, schema in DATASET_SCHEMAS.items():
            path = _find_file(self.test_dir, stem)
            status = f"FOUND  ({path.name})" if path else "MISSING"
            print(f"  {stem:<12} → {schema['task']:<20}  {status}")
        print()
