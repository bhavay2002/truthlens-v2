from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


# =========================================================
# PATH SETTINGS
# =========================================================

@dataclass(frozen=True)
class PathSettings:
    project_root: Path
    data_dir: Path
    artifacts_dir: Path
    models_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    cache_dir: Path  

    training_log_path: Path
    evaluation_results_path: Path

    def ensure_dirs(self) -> None:
        """
        Ensure all required directories exist.
        """
        for p in [
            self.data_dir,
            self.artifacts_dir,
            self.models_dir,
            self.logs_dir,
            self.checkpoints_dir,
            self.cache_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


# =========================================================
# DATA PATHS (18 FILES)
# =========================================================

@dataclass(frozen=True)
class DataPaths:
    tasks: Dict[str, Dict[str, Path]]

    def get(self, task: str, split: str) -> Path:
        return self.tasks[task][split]

    def validate(self) -> None:
        """
        Ensure all dataset files exist.
        """
        missing = []

        for task, splits in self.tasks.items():
            for split, path in splits.items():
                if not path.exists():
                    missing.append(str(path))

        if missing:
            raise FileNotFoundError(
                "Missing dataset files:\n" + "\n".join(missing)
            )


# =========================================================
# RUNTIME FLAGS
# =========================================================

@dataclass(frozen=True)
class RuntimeFlags:
    require_gpu: bool = False
    debug: bool = False


# =========================================================
# GLOBAL SETTINGS
# =========================================================

@dataclass(frozen=True)
class Settings:
    paths: PathSettings
    data: DataPaths
    runtime: RuntimeFlags = field(default_factory=RuntimeFlags)


# =========================================================
# HELPERS
# =========================================================

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_path(key: str, default: Path) -> Path:
    val = os.environ.get(key)
    return Path(val).expanduser().resolve() if val else default.resolve()


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y"}


# =========================================================
# DATA PATH BUILDER (SPLIT-FIRST STRUCTURE)
# =========================================================

def _build_data_paths(data_dir: Path) -> DataPaths:

    def p(split: str, task: str) -> Path:
        return data_dir / split / f"{task}.csv"

    tasks = {
        "bias": {
            "train": p("train", "bias"),
            "val": p("val", "bias"),
            "test": p("test", "bias"),
        },
        "ideology": {
            "train": p("train", "ideology"),
            "val": p("val", "ideology"),
            "test": p("test", "ideology"),
        },
        "propaganda": {
            "train": p("train", "propaganda"),
            "val": p("val", "propaganda"),
            "test": p("test", "propaganda"),
        },
        "narrative": {
            "train": p("train", "narrative"),
            "val": p("val", "narrative"),
            "test": p("test", "narrative"),
        },
        "narrative_frame": {
            "train": p("train", "frame"),
            "val": p("val", "frame"),
            "test": p("test", "frame"),
        },
        "emotion": {
            "train": p("train", "emotion"),
            "val": p("val", "emotion"),
            "test": p("test", "emotion"),
        },
    }

    return DataPaths(tasks=tasks)


# =========================================================
# MAIN
# =========================================================

def load_settings(*, validate_data: bool = False) -> Settings:
    """Load runtime settings.

    The ``data/`` directory is built lazily — datasets only need to exist
    when the train pipeline actually runs. Pass ``validate_data=True`` from
    the train entry point to assert all 18 split files are present; the
    default leaves them unchecked so ``--mode infer`` and the API can boot
    without any CSVs on disk.
    """

    root = _project_root()

    # -----------------------------------------------------
    # BASE DIRECTORIES
    # -----------------------------------------------------

    artifacts_dir = _env_path("TRUTHLENS_ARTIFACTS_DIR", root / "artifacts")
    data_dir = _env_path("TRUTHLENS_DATA_DIR", root / "data")

    models_dir = _env_path("TRUTHLENS_MODELS_DIR", artifacts_dir / "models")
    logs_dir = _env_path("TRUTHLENS_LOGS_DIR", artifacts_dir / "logs")
    checkpoints_dir = _env_path("TRUTHLENS_CKPT_DIR", artifacts_dir / "checkpoints")
    cache_dir = _env_path("TRUTHLENS_CACHE_DIR", artifacts_dir / "cache")  # 🔥 NEW

    # -----------------------------------------------------
    # PATH SETTINGS
    # -----------------------------------------------------

    paths = PathSettings(
        project_root=root,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        cache_dir=cache_dir,
        training_log_path=logs_dir / "training.log",
        evaluation_results_path=models_dir / "evaluation.json",
    )

    #  CREATE ALL DIRECTORIES
    paths.ensure_dirs()

    # -----------------------------------------------------
    # DATA PATHS
    # -----------------------------------------------------

    data_paths = _build_data_paths(data_dir)

    # Opt-in CSV-presence check. Train flows pass ``validate_data=True``;
    # inference / API flows do not need the data directory populated and
    # would otherwise crash at boot time on a fresh checkout.
    if validate_data:
        data_paths.validate()

    # -----------------------------------------------------
    # RUNTIME FLAGS
    # -----------------------------------------------------

    runtime = RuntimeFlags(
        require_gpu=_env_bool("TRUTHLENS_REQUIRE_GPU", False),
        debug=_env_bool("TRUTHLENS_DEBUG", False),
    )

    return Settings(
        paths=paths,
        data=data_paths,
        runtime=runtime,
    )