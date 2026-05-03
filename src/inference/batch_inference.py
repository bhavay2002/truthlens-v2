from __future__ import annotations

import argparse
import json 
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

from src.inference.constants import DEFAULT_INFERENCE_BATCH_SIZE
from src.inference.model_loader import ModelLoader
from src.inference.feature_preparer import (
    FeaturePreparer,
    FeaturePreparationConfig,
)
from src.inference.inference_pipeline import (
    PredictionPipeline,
    PredictionPipelineConfig,
)
from src.inference.report_generator import ReportGenerator
from src.inference.result_formatter import ResultFormatter
from src.analysis.integration_runner import AnalysisIntegrationRunner
from src.features.base.base_feature import FeatureContext
from src.features.pipelines.feature_pipeline import FeaturePipeline
from src.graph.graph_pipeline import GraphPipeline, get_default_pipeline

logger = logging.getLogger(__name__)


# =========================================================
# HELPER
# =========================================================

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class BatchInferenceConfig:
    dataset_path: str
    text_column: str = "text"
    output_path: str = "batch_predictions.json"
    # CFG-5: keep the default in lockstep with the engine's
    # ``InferenceConfig.batch_size`` via the shared constant.
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE
    models_dir: str = "models"
    num_workers: int = 0
    # LAT-6: opt-in for torch.compile, propagated to ModelLoader.
    use_torch_compile: bool = False


# =========================================================
# ENGINE
# =========================================================

class BatchInferenceEngine:

    def __init__(self, config: BatchInferenceConfig):

        self.config = config
        self.model_loader = ModelLoader(
            config.models_dir,
            use_torch_compile=config.use_torch_compile,
        )
        self.artifacts = self.model_loader.load_all()

        # ---------------- FEATURE PREPARER ----------------
        schema = self.artifacts.feature_schema
        feature_schema = (
            list(schema.keys()) if isinstance(schema, dict)
            else schema if isinstance(schema, list)
            else ["text_length"]
        )

        self.feature_preparer = FeaturePreparer(
            FeaturePreparationConfig(
                feature_schema=feature_schema,
                return_tensor=True,
            ),
            scaler=self.artifacts.feature_scaler,
            selector=self.artifacts.feature_selector,
        )

        # CRIT-6: previously ``_process_batch`` fed the preparer with a stub
        # ``[{"text": t, "text_length": len(t)}]`` dict, which left every
        # other slot in the schema (bias_*, framing_*, ideological_*, …)
        # at zero. We now run the real ``FeaturePipeline`` on each input.
        self.feature_pipeline = FeaturePipeline()
        self.feature_pipeline.initialize()

        # ---------------- MODEL PIPELINE ----------------
        self.prediction_pipeline = PredictionPipeline(
            config=PredictionPipelineConfig(
                device=str(self.model_loader.device),
                return_probabilities=True,  # 🔥 IMPORTANT
                return_logits=True,         # 🔥 NEW
            ),
            bias_model=self.artifacts.bias_model,
            ideology_model=self.artifacts.ideology_model,
            emotion_model=self.artifacts.emotion_model,
        )

        self.report_generator = ReportGenerator()
        self.formatter = ResultFormatter()
        self.analysis_runner = AnalysisIntegrationRunner()
        self.graph_pipeline = get_default_pipeline()  # G-R1: shared singleton

    # =====================================================
    # DATA
    # =====================================================

    def _load_dataset(self):
        df = pd.read_csv(self.config.dataset_path)
        if self.config.text_column not in df.columns:
            raise ValueError("Text column missing")
        return df

    # =====================================================
    # BATCH PROCESSING (UPDATED 🔥)
    # =====================================================

    def _process_batch(self, texts: List[str]):

        # CRIT-6: build the real per-text feature dict via FeaturePipeline
        # so the preparer receives every named slot the schema expects
        # (instead of an empty stub that produced silently-wrong inputs).
        contexts = [FeatureContext(text=t) for t in texts]
        try:
            extracted = self.feature_pipeline.batch_extract(contexts)
        except Exception:
            extracted = [self.feature_pipeline.extract(c) for c in contexts]

        features = []
        for t, fdict in zip(texts, extracted):
            row = dict(fdict) if isinstance(fdict, dict) else {}
            row.setdefault("text", t)
            row.setdefault("text_length", float(len(t)))
            features.append(row)

        prepared = self.feature_preparer.prepare_batch(features)
        if not torch.is_tensor(prepared):
            prepared = torch.tensor(prepared, dtype=torch.float32)

        device_type = str(self.prediction_pipeline.device.type)
        amp_enabled = device_type == "cuda"

        with torch.inference_mode():
            if amp_enabled:
                with torch.autocast(device_type=device_type):
                    output = self.prediction_pipeline.predict(prepared)
            else:
                output = self.prediction_pipeline.predict(prepared)

        # output keys: "bias" (list), "ideology" (list),
        # "propaganda_probability" (list), "emotion" (list)

        results = []

        for i, text in enumerate(texts):

            bias_val = output.get("bias", [None] * (i + 1))[i]
            ideology_val = output.get("ideology", [None] * (i + 1))[i]
            prop_val = output.get("propaganda_probability", [None] * (i + 1))[i]
            emotion_val = output.get("emotion", [None] * (i + 1))[i]

            preds = {
                "bias": bias_val,
                "ideology": ideology_val,
                "propaganda_probability": prop_val,
                "emotion": emotion_val,
            }

            report = self.report_generator.generate_report(
                article_text=text,
                bias_analysis={"bias": bias_val},
                emotion_analysis={"emotion": emotion_val},
                credibility_score=None,
            )

            results.append({
                "text": text,
                "predictions": preds,
                "report": report,
            })

        return results

    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        df = self._load_dataset()
        results = []

        for i in tqdm(range(0, len(df), self.config.batch_size)):

            batch = df.iloc[i:i + self.config.batch_size]
            texts = batch[self.config.text_column].fillna("").tolist()

            batch_results = self._process_batch(texts)
            results.extend(batch_results)

        return results

    # =====================================================
    # SAVE
    # =====================================================

    def save_results(self, results):
        with open(self.config.output_path, "w") as f:
            json.dump(results, f, indent=4)


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="predictions.json")

    args = parser.parse_args()

    engine = BatchInferenceEngine(
        BatchInferenceConfig(
            dataset_path=args.dataset,
            output_path=args.output,
        )
    )

    results = engine.run()
    engine.save_results(results)


if __name__ == "__main__":
    main()