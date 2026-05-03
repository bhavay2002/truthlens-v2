"""
End-to-end smoke test for TruthLens against the 10-row datasets.

Stages:
    1. Load + validate + clean + leakage-check via ``run_data_pipeline``
       (no training — verifies the data layer).
    2. Run the analysis + features + aggregation pipeline on a sample
       text via ``TruthLensPipeline`` (no predictor — verifies the
       inference fan-out works without a trained checkpoint).

Run:
    .pythonlibs/bin/python scripts/smoke_e2e.py
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

from src.config.config_loader import load_config  # noqa: E402
from src.config.settings_loader import load_settings  # noqa: E402
from src.data_processing.data_pipeline import (  # noqa: E402
    DataPipelineConfig,
    run_data_pipeline,
)
from src.pipelines.truthlens_pipeline import TruthLensPipeline  # noqa: E402
from src.utils.logging_utils import configure_logging  # noqa: E402

configure_logging()
logger = logging.getLogger("smoke")


def stage_data(tokenizer) -> dict:
    settings = load_settings()
    data_config = {
        "bias": {
            "train": settings.data.tasks["bias"]["train"],
            "val": settings.data.tasks["bias"]["val"],
            "test": settings.data.tasks["bias"]["test"],
        },
        "ideology": {
            "train": settings.data.tasks["ideology"]["train"],
            "val": settings.data.tasks["ideology"]["val"],
            "test": settings.data.tasks["ideology"]["test"],
        },
        "propaganda": {
            "train": settings.data.tasks["propaganda"]["train"],
            "val": settings.data.tasks["propaganda"]["val"],
            "test": settings.data.tasks["propaganda"]["test"],
        },
        "narrative": {
            "train": settings.data.tasks["narrative"]["train"],
            "val": settings.data.tasks["narrative"]["val"],
            "test": settings.data.tasks["narrative"]["test"],
        },
        "narrative_frame": {
            "train": settings.data.tasks["narrative_frame"]["train"],
            "val": settings.data.tasks["narrative_frame"]["val"],
            "test": settings.data.tasks["narrative_frame"]["test"],
        },
        "emotion": {
            "train": settings.data.tasks["emotion"]["train"],
            "val": settings.data.tasks["emotion"]["val"],
            "test": settings.data.tasks["emotion"]["test"],
        },
    }
    cfg = DataPipelineConfig(
        enable_cache=False,         # always rebuild for the smoke test
        enable_augmentation=False,  # 10-row dataset doesn't need augmentation
        enable_profiling=False,     # skip the heavy EDA writes
    )
    t0 = time.time()
    datasets = run_data_pipeline(
        data_config=data_config,
        tokenizer=tokenizer,
        build_dataloaders=False,
        config=cfg,
    )
    dt = time.time() - t0
    logger.info("[stage_data] OK in %.2fs", dt)
    for task, splits in datasets.items():
        for split, df in splits.items():
            logger.info("  %s/%s -> rows=%d cols=%d", task, split, len(df), len(df.columns))
    return datasets


def stage_inference(tokenizer) -> None:
    config = load_config(ROOT / "config" / "config.yaml")
    pipeline = TruthLensPipeline(
        tokenizer=tokenizer,
        model_version=getattr(config.model, "version", config.model.encoder),
        enable_explainability=False,
        enable_evaluation=False,
        parallel_stages=False,
    )
    samples = [
        "Senator Brooks praised the federal budget proposal as a balanced and forward-looking decision, "
        "highlighting transparency and the long-term impact on ordinary citizens.",
        "Opposition leaders condemned the controversial energy bill as a reckless overreach, "
        "warning of severe consequences for rural communities and small businesses.",
    ]
    t0 = time.time()
    batch = pipeline.analyze_batch(samples)
    dt = time.time() - t0
    logger.info(
        "[stage_inference] OK in %.2fs  n=%d  model=%s",
        dt,
        batch["batch_metadata"]["n_articles"],
        batch["batch_metadata"]["model_version"],
    )
    for i, article in enumerate(batch["articles"]):
        logger.info(
            "  article %d  scores=%s  errors=%s",
            i + 1,
            article.get("scores"),
            article.get("errors"),
        )
    pipeline.close()


def main() -> int:
    logger.info("== TruthLens 10-row smoke test ==")
    config = load_config(ROOT / "config" / "config.yaml")
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder)

    failures: list[str] = []

    for name, fn in [("data", lambda: stage_data(tokenizer)),
                     ("inference", lambda: stage_inference(tokenizer))]:
        logger.info("---- stage: %s ----", name)
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            logger.error("stage %s FAILED: %s", name, exc)
            traceback.print_exc()
            failures.append(name)

    if failures:
        logger.error("FAILURES: %s", failures)
        return 1
    logger.info("ALL STAGES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
