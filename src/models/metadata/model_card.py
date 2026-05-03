from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# DATA CLASSES
# =========================================================

@dataclass
class ModelDetails:
    name: str
    version: str
    architecture: str
    description: str
    author: str
    license: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self):
        _validate_non_empty(self.name, "name")
        _validate_non_empty(self.version, "version")
        _validate_non_empty(self.architecture, "architecture")
        _validate_non_empty(self.description, "description")
        _validate_non_empty(self.author, "author")


@dataclass
class DatasetInfo:
    name: str
    source: Optional[str] = None
    preprocessing: Optional[str] = None
    size: Optional[int] = None
    features: Optional[List[str]] = None

    def __post_init__(self):
        _validate_non_empty(self.name, "dataset name")


@dataclass
class TrainingConfig:
    framework: str
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    hardware: Optional[str] = None

    def __post_init__(self):
        _validate_non_empty(self.framework, "framework")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class EvaluationResults:
    metrics: Dict[str, float]
    validation_dataset: Optional[str] = None
    test_dataset: Optional[str] = None

    def __post_init__(self):
        if not self.metrics:
            raise ValueError("metrics cannot be empty")


@dataclass
class EthicalConsiderations:
    intended_use: str
    limitations: Optional[str] = None
    biases: Optional[str] = None
    risks: Optional[str] = None
    mitigation_strategies: Optional[str] = None

    def __post_init__(self):
        _validate_non_empty(self.intended_use, "intended_use")


@dataclass
class ModelArtifacts:
    model_weights: Optional[str] = None
    tokenizer: Optional[str] = None
    config_file: Optional[str] = None
    training_logs: Optional[str] = None
    checkpoint_dir: Optional[str] = None


# =========================================================
# MAIN MODEL CARD
# =========================================================

@dataclass
class ModelCard:
    model_details: ModelDetails
    datasets: List[DatasetInfo]
    training: TrainingConfig
    evaluation: EvaluationResults
    ethics: EthicalConsiderations
    artifacts: ModelArtifacts
    tags: Optional[List[str]] = None
    references: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None

    # =====================================================
    # SERIALIZATION
    # =====================================================

    def to_dict(self) -> Dict[str, Any]:
        try:
            data = asdict(self)
            if self.extra:
                data["extra"] = self.extra
            return data
        except Exception as e:
            logger.exception("ModelCard serialization failed")
            raise RuntimeError from e

    # =====================================================
    # SAVE
    # =====================================================

    def save_json(self, path: str | Path) -> Path:

        path = Path(path)
        _ensure_dir(path)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

        logger.info("ModelCard saved JSON: %s", path)
        return path

    def save_markdown(self, path: str | Path) -> Path:

        path = Path(path)
        _ensure_dir(path)

        data = self.to_dict()

        with open(path, "w", encoding="utf-8") as f:

            f.write(f"# Model Card: {data['model_details']['name']}\n\n")

            # DETAILS
            f.write("## Model Details\n")
            for k, v in data["model_details"].items():
                f.write(f"- **{k}**: {v}\n")

            # DATASETS
            f.write("\n## Datasets\n")
            for d in data["datasets"]:
                f.write(f"- **{d['name']}**\n")
                for k, v in d.items():
                    if k != "name" and v is not None:
                        f.write(f"  - {k}: {v}\n")

            # TRAINING
            f.write("\n## Training\n")
            for k, v in data["training"].items():
                f.write(f"- **{k}**: {v}\n")

            # EVAL
            f.write("\n## Evaluation\n")
            for k, v in data["evaluation"]["metrics"].items():
                f.write(f"- **{k}**: {v}\n")

            # ETHICS
            f.write("\n## Ethics\n")
            for k, v in data["ethics"].items():
                if v:
                    f.write(f"- **{k}**: {v}\n")

            # TAGS
            if data.get("tags"):
                f.write("\n## Tags\n")
                for t in data["tags"]:
                    f.write(f"- {t}\n")

            # REFERENCES
            if data.get("references"):
                f.write("\n## References\n")
                for r in data["references"]:
                    f.write(f"- {r}\n")

            # EXTRA
            if data.get("extra"):
                f.write("\n## Extra\n")
                for k, v in data["extra"].items():
                    f.write(f"- **{k}**: {v}\n")

        logger.info("ModelCard saved Markdown: %s", path)
        return path

    # =====================================================
    # LOAD
    # =====================================================

    @staticmethod
    def load_json(path: str | Path) -> "ModelCard":

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return ModelCard(
            model_details=ModelDetails(**data["model_details"]),
            datasets=[DatasetInfo(**d) for d in data["datasets"]],
            training=TrainingConfig(**data["training"]),
            evaluation=EvaluationResults(**data["evaluation"]),
            ethics=EthicalConsiderations(**data["ethics"]),
            artifacts=ModelArtifacts(**data["artifacts"]),
            tags=data.get("tags"),
            references=data.get("references"),
            extra=data.get("extra"),
        )