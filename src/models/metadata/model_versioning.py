from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================================================
# UTILS
# =========================================================

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _validate_non_empty(value: str, name: str):
    if not value or not value.strip():
        raise ValueError(f"{name} cannot be empty")


# =========================================================
# DATA CLASS
# =========================================================

@dataclass
class ModelVersionInfo:
    model_name: str
    version: str
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    description: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    artifact_path: Optional[str] = None
    tags: Optional[List[str]] = None

    def __post_init__(self):
        _validate_non_empty(self.model_name, "model_name")
        _validate_non_empty(self.version, "version")

        if self.metrics:
            for k, v in self.metrics.items():
                if not isinstance(v, (int, float)):
                    raise ValueError(f"Invalid metric type for {k}")


# =========================================================
# REGISTRY
# =========================================================

class ModelVersionRegistry:

    REGISTRY_FILE = "model_registry.json"

    def __init__(self, registry_dir: str | Path):

        self.registry_dir = Path(registry_dir)
        _ensure_dir(self.registry_dir)

        self.registry_path = self.registry_dir / self.REGISTRY_FILE
        self._lock = threading.Lock()

        if not self.registry_path.exists():
            self._init_registry()

    # =====================================================
    # INTERNAL
    # =====================================================

    def _init_registry(self):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump({"models": {}}, f, indent=4)

    def _load(self) -> Dict:
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, data: Dict):
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # =====================================================
    # REGISTER
    # =====================================================

    def register_version(self, info: ModelVersionInfo) -> Path:

        version_dir = (
            self.registry_dir / info.model_name / f"version_{info.version}"
        )

        _ensure_dir(version_dir)

        info.artifact_path = str(version_dir)

        with self._lock:

            registry = self._load()

            registry["models"].setdefault(info.model_name, [])
            registry["models"][info.model_name].append(asdict(info))

            self._save(registry)

        metadata_file = version_dir / "version.json"

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(asdict(info), f, indent=4)

        logger.info("Registered model version: %s %s", info.model_name, info.version)

        return version_dir

    # =====================================================
    # QUERY
    # =====================================================

    def list_versions(self, model_name: str) -> List[ModelVersionInfo]:

        _validate_non_empty(model_name, "model_name")

        with self._lock:
            registry = self._load()

        entries = registry["models"].get(model_name, [])

        return [ModelVersionInfo(**e) for e in entries]

    def get_latest_version(self, model_name: str) -> Optional[ModelVersionInfo]:

        versions = self.list_versions(model_name)

        if not versions:
            return None

        return sorted(
            versions,
            key=lambda v: v.created_at,
            reverse=True,
        )[0]

    def get_version(
        self,
        model_name: str,
        version: str,
    ) -> Optional[ModelVersionInfo]:

        for v in self.list_versions(model_name):
            if v.version == version:
                return v

        return None

    # =====================================================
    # DELETE / CLEANUP
    # =====================================================

    def delete_version(self, model_name: str, version: str) -> bool:

        with self._lock:
            registry = self._load()

            versions = registry["models"].get(model_name, [])

            new_versions = [v for v in versions if v["version"] != version]

            if len(new_versions) == len(versions):
                return False

            registry["models"][model_name] = new_versions
            self._save(registry)

        version_dir = self.registry_dir / model_name / f"version_{version}"

        if version_dir.exists():
            import shutil
            shutil.rmtree(version_dir)

        logger.info("Deleted version: %s %s", model_name, version)

        return True

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self) -> Dict[str, Any]:

        with self._lock:
            registry = self._load()

        return {
            model: len(versions)
            for model, versions in registry["models"].items()
        }