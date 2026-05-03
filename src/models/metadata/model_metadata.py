from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =========================================================
# VALIDATION
# =========================================================

def _validate_non_empty(value: str, name: str):
    if not value or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# CORE COMPONENTS
# =========================================================

@dataclass
class ModelIdentity:
    model_name: str
    version: str
    architecture: str
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        _validate_non_empty(self.model_name, "model_name")
        _validate_non_empty(self.version, "version")
        _validate_non_empty(self.architecture, "architecture")


@dataclass
class TrainingProvenance:
    dataset_name: str
    dataset_version: Optional[str]
    experiment_name: Optional[str]
    run_id: Optional[str]
    framework: str
    seed: Optional[int] = None

    def __post_init__(self):
        _validate_non_empty(self.dataset_name, "dataset_name")
        _validate_non_empty(self.framework, "framework")


@dataclass
class ArtifactPaths:
    model_weights: Optional[str]
    config_file: Optional[str]
    tokenizer_path: Optional[str]
    training_logs: Optional[str]
    checkpoint_directory: Optional[str]

    def validate(self):
        for p in [
            self.model_weights,
            self.config_file,
            self.tokenizer_path,
            self.training_logs,
            self.checkpoint_directory,
        ]:
            if p and not Path(p).exists():
                logger.warning("Missing artifact: %s", p)


@dataclass
class RuntimeEnvironment:
    python_version: str
    framework_version: Optional[str]
    cuda_version: Optional[str]
    hardware: Optional[str]
    device_count: Optional[int]

    def __post_init__(self):
        _validate_non_empty(self.python_version, "python_version")


# =========================================================
# MAIN METADATA
# =========================================================

@dataclass
class ModelMetadata:
    identity: ModelIdentity
    provenance: TrainingProvenance
    artifacts: ArtifactPaths
    runtime: RuntimeEnvironment
    metrics: Optional[Dict[str, float]] = None
    tags: Optional[Dict[str, str]] = None
    extra: Optional[Dict[str, Any]] = None

    # =====================================================
    # SERIALIZATION
    # =====================================================

    def to_dict(self) -> Dict[str, Any]:
        try:
            return asdict(self)
        except Exception as e:
            logger.exception("Serialization failed")
            raise RuntimeError from e

    # =====================================================
    # SAVE
    # =====================================================

    def save_json(self, path: str | Path) -> Path:

        path = Path(path)
        _ensure_dir(path)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

        logger.info("Metadata saved: %s", path)
        return path

    # =====================================================
    # LOAD
    # =====================================================

    @classmethod
    def load_json(cls, path: str | Path) -> "ModelMetadata":

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = cls(
            identity=ModelIdentity(**data["identity"]),
            provenance=TrainingProvenance(**data["provenance"]),
            artifacts=ArtifactPaths(**data["artifacts"]),
            runtime=RuntimeEnvironment(**data["runtime"]),
            metrics=data.get("metrics"),
            tags=data.get("tags"),
            extra=data.get("extra"),
        )

        metadata.artifacts.validate()

        return metadata