from __future__ import annotations

# TOKENIZERS-FORK-FIX: must be set BEFORE ``transformers`` is imported.
# See main.py for the full rationale.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
from pathlib import Path
from typing import Any, List, Optional

import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from src.inference.predict_api import predict, predict_batch
from src.analysis.argument_mining import ArgumentMiningAnalyzer
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.graph.graph_pipeline import GraphPipeline, get_default_pipeline
from src.graph.graph_embeddings import GraphEmbeddingGenerator
from src.graph.temporal_graph import TemporalGraphAnalyzer
from src.analysis.context_omission_detector import ContextOmissionDetector
from src.analysis.discourse_coherence_analyzer import DiscourseCoherenceAnalyzer
from src.analysis.emotion_target_analysis import EmotionTargetAnalyzer
from src.analysis.framing_analysis import FramingAnalyzer
from src.analysis.ideological_language_detector import IdeologicalLanguageDetector
from src.analysis.information_density_analyzer import InformationDensityAnalyzer
from src.analysis.information_omission_detector import InformationOmissionDetector
from src.analysis.narrative_conflict import NarrativeConflictAnalyzer
from src.analysis.narrative_propagation import NarrativePropagationAnalyzer
from src.analysis.narrative_role_extractor import NarrativeRoleExtractor
from src.analysis.narrative_temporal_analyzer import NarrativeTemporalAnalyzer
from src.analysis.propaganda_pattern_detector import PropagandaPatternDetector
from src.analysis.rhetorical_device_detector import RhetoricalDeviceDetector
from src.analysis.source_attribution_analyzer import SourceAttributionAnalyzer
from src.features.bias.bias_lexicon import compute_bias_features
from src.features.emotion.emotion_lexicon import EmotionLexiconAnalyzer
from src.explainability.emotion_explainer import explain_emotion
from src.explainability.lime_explainer import explain_prediction
from src.explainability.explainability_pipeline import (
    run_explainability_pipeline,
    ExplainabilityConfig,
)
from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.utils import ensure_non_empty_text, ensure_non_empty_text_list
from src.utils.logging_utils import configure_logging
from src.utils.settings import load_settings

# ── src.inference integration ──────────────────────────────────────────────────
from src.inference.inference_cache import InferenceCache, InferenceCacheConfig
from src.inference.inference_logger import InferenceLogger
from src.inference.result_formatter import ResultFormatter
from src.inference.report_generator import ReportGenerator, ReportConfig
from src.inference.inference_engine import (
    InferenceEngine,
    InferenceConfig as EngineConfig,
)

# ── src.models.calibration integration ────────────────────────────────────────
from src.models.calibration.calibration_metrics import (
    CalibrationMetrics,
    CalibrationMetricConfig,
)
from src.models.calibration.temperature_scaling import (
    TemperatureScaler,
    TemperatureScalingConfig,
)
from src.models.calibration.isotonic_calibration import (
    IsotonicCalibrator,
    IsotonicCalibrationConfig,
)

# ── src.models.ensemble integration ───────────────────────────────────────────
from src.models.ensemble.ensemble_model import EnsembleConfig
from src.models.ensemble.weighted_ensemble import WeightedEnsembleConfig

# ── src.models.export integration ─────────────────────────────────────────────
from src.models.export.onnx_export import ONNXExporter, ONNXExportConfig
from src.models.export.torchscript_export import (
    TorchScriptExporter,
    TorchScriptExportConfig,
)
from src.models.export.quantization import QuantizationEngine, QuantizationConfig

configure_logging()
logger = logging.getLogger(__name__)
SETTINGS = load_settings()
MODEL_PATH = SETTINGS.model.path
VECTORIZER_PATH = SETTINGS.paths.tfidf_vectorizer_path
TRAINING_TEXT_COLUMN = SETTINGS.training.text_column
APP_TITLE = SETTINGS.api.title
APP_DESCRIPTION = SETTINGS.api.description
APP_VERSION = SETTINGS.api.version
TEXT_PREVIEW_CHARS = max(int(SETTINGS.api.text_preview_chars), 1)
INFERENCE_ALLOW_RAW_TEXT_FALLBACK = bool(SETTINGS.inference.allow_raw_text_fallback)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SUBPACKAGES = (
    "emotion",
    "encoder",
    "ideology",
    "multitask",
    "narrative",
    "propaganda",
)
LIME_NUM_SAMPLES = 16

# ── Singleton analyzers ────────────────────────────────────────────────────────
EMOTION_ANALYZER = EmotionLexiconAnalyzer()
ARGUMENT_ANALYZER = ArgumentMiningAnalyzer()
BIAS_PROFILE_BUILDER = BiasProfileBuilder()
CONTEXT_OMISSION_DETECTOR = ContextOmissionDetector()
DISCOURSE_ANALYZER = DiscourseCoherenceAnalyzer()
EMOTION_TARGET_ANALYZER = EmotionTargetAnalyzer()
FRAMING_ANALYZER = FramingAnalyzer()
IDEOLOGICAL_DETECTOR = IdeologicalLanguageDetector()
INFO_DENSITY_ANALYZER = InformationDensityAnalyzer()
INFO_OMISSION_DETECTOR = InformationOmissionDetector()
NARRATIVE_CONFLICT_ANALYZER = NarrativeConflictAnalyzer()
NARRATIVE_PROPAGATION_ANALYZER = NarrativePropagationAnalyzer()
NARRATIVE_ROLE_EXTRACTOR = NarrativeRoleExtractor()
NARRATIVE_TEMPORAL_ANALYZER = NarrativeTemporalAnalyzer()
PROPAGANDA_PATTERN_DETECTOR = PropagandaPatternDetector()
RHETORICAL_DETECTOR = RhetoricalDeviceDetector()
SOURCE_ATTRIBUTION_ANALYZER = SourceAttributionAnalyzer()
GRAPH_PIPELINE = get_default_pipeline()  # G-R1: process-wide singleton
GRAPH_EMBEDDING_GENERATOR = GraphEmbeddingGenerator()
TEMPORAL_GRAPH_ANALYZER = TemporalGraphAnalyzer()

# ── src.inference singletons ───────────────────────────────────────────────────
INFERENCE_CACHE = InferenceCache(
    InferenceCacheConfig(
        cache_dir="cache/inference",
        enable_disk_cache=False,
        enable_memory_cache=True,
        ttl_seconds=3600,
    )
)

INFERENCE_LOGGER = InferenceLogger(service_name="truthlens-api", enable_json_logs=True)

RESULT_FORMATTER = ResultFormatter()

REPORT_GENERATOR = ReportGenerator(
    ReportConfig(include_timestamp=True, pretty_json=False, validate_fields=True)
)

AGGREGATION_PIPELINE = AggregationPipeline()

# InferenceEngine is initialised lazily so a missing model directory does not
# crash the server at startup.
_INFERENCE_ENGINE: Optional[InferenceEngine] = None


def _get_inference_engine() -> Optional[InferenceEngine]:
    """Return the shared InferenceEngine, or None if the model is not yet trained."""
    global _INFERENCE_ENGINE
    if _INFERENCE_ENGINE is None and MODEL_PATH.exists():
        try:
            _INFERENCE_ENGINE = InferenceEngine(
                EngineConfig(
                    model_path=str(MODEL_PATH),
                    tokenizer_path=str(MODEL_PATH),
                    max_length=SETTINGS.model.max_length,
                    device=SETTINGS.inference.device,
                )
            )
            logger.info("InferenceEngine initialised from %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("InferenceEngine could not be initialised: %s", exc)
    return _INFERENCE_ENGINE


# ── src.models.calibration singletons ─────────────────────────────────────────
CALIBRATION_METRICS = CalibrationMetrics()

# TemperatureScaler and IsotonicCalibrator require validation data to fit, so
# they are exposed only through info/metrics endpoints rather than as eager
# singletons that would call .fit() at startup.

# ── src.models.export singletons ──────────────────────────────────────────────
ONNX_EXPORTER = ONNXExporter(ONNXExportConfig(verify_export=False))
TORCHSCRIPT_EXPORTER = TorchScriptExporter(TorchScriptExportConfig(verify_export=False))

# QuantizationEngine raises ValueError if the requested backend is not
# available (e.g. 'fbgemm' on some ARM builds), so it is initialised lazily.
_QUANTIZATION_ENGINE: Optional[QuantizationEngine] = None


def _get_quantization_engine(method: str = "dynamic") -> Optional[QuantizationEngine]:
    """Return a QuantizationEngine for the given method, or None if unavailable."""
    global _QUANTIZATION_ENGINE
    if _QUANTIZATION_ENGINE is None:
        try:
            _QUANTIZATION_ENGINE = QuantizationEngine(
                QuantizationConfig(method=method, backend="fbgemm")
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("QuantizationEngine not available: %s", exc)
    return _QUANTIZATION_ENGINE


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)


# ── Startup hook: prune the on-disk feature cache (audit fix #1.5) ────────────
#
# Without an eviction policy the on-disk feature cache grew unbounded and
# orphan tempfiles from killed processes were never cleaned up.  On every
# server start we now sweep every namespace under the configured cache
# root, dropping anything older than `FEATURE_CACHE_MAX_AGE_DAYS` and
# (after that) trimming each namespace down to `FEATURE_CACHE_MAX_BYTES`
# by oldest-first eviction.  Both knobs are env-tunable; sensible
# production defaults below.

FEATURE_CACHE_MAX_AGE_DAYS = float(
    getattr(SETTINGS, "feature_cache_max_age_days", 0) or 14.0
)
FEATURE_CACHE_MAX_BYTES = int(
    getattr(SETTINGS, "feature_cache_max_bytes", 0) or (512 * 1024 * 1024)
)


@app.on_event("startup")
def _prune_feature_cache_on_startup() -> None:
    try:
        from src.features.cache.cache_manager import CacheManager

        cache_root = getattr(getattr(SETTINGS, "paths", None), "cache_dir", None)
        manager = CacheManager(base_cache_dir=Path(cache_root) if cache_root else None)
        results = manager.prune_all(
            max_bytes_per_namespace=FEATURE_CACHE_MAX_BYTES,
            max_age_days=FEATURE_CACHE_MAX_AGE_DAYS,
        )
        if results:
            total_removed = sum(
                int(r.get("removed_age", 0)) + int(r.get("removed_size", 0))
                for r in results.values()
            )
            logger.info(
                "Feature cache prune complete | namespaces=%d removed=%d",
                len(results),
                total_removed,
            )
    except Exception as exc:
        # Pruning must NEVER block the server from starting.
        logger.warning("Feature cache prune skipped: %s", exc)


# ── Request / response models ──────────────────────────────────────────────────

class NewsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Breaking news: Scientists discover new species in Amazon rainforest."
            }
        }
    )
    text: str = Field(..., min_length=10, max_length=10_000, description="News article text to analyze")


class BatchNewsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "First news article text here.",
                    "Second news article text here.",
                ]
            }
        }
    )
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of news article texts to analyze (max 50 items)",
    )


class NewsResponse(BaseModel):
    text: str
    fake_probability: float = Field(..., ge=0, le=1, description="Probability of being fake news (0-1)")
    prediction: str
    confidence: float


class BatchNewsResponse(BaseModel):
    results: list[NewsResponse]
    total: int
    cache_hits: int


class AnalysisResponse(BaseModel):
    text: str
    prediction: str
    fake_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    bias: dict[str, Any]
    emotion: dict[str, Any]
    narrative: dict[str, Any]
    framing: dict[str, Any]
    rhetoric: dict[str, Any]
    discourse: dict[str, Any]
    propaganda_analysis: dict[str, Any]
    credibility_profile: dict[str, Any]
    graph_analysis: dict[str, Any]
    explainability: dict[str, Any]


class ReportResponse(BaseModel):
    article_summary: dict[str, Any]
    bias_analysis: dict[str, Any]
    emotion_analysis: dict[str, Any]
    narrative_structure: dict[str, Any]
    entity_graph: dict[str, Any]
    credibility_score: Optional[float]


class ModelInfoResponse(BaseModel):
    available: bool
    model_path: str
    device: Optional[str]
    num_parameters: Optional[int]
    num_trainable_parameters: Optional[int]
    label_map: Optional[dict[str, Any]]


# ── Calibration models ─────────────────────────────────────────────────────────

class CalibrationMetricsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "probabilities": [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]],
                "labels": [0, 1, 0],
                "n_bins": 15,
            }
        }
    )
    probabilities: List[List[float]] = Field(
        ...,
        description="Per-sample probability distributions [[p_real, p_fake], ...] for each article",
    )
    labels: List[int] = Field(
        ...,
        description="Ground-truth class indices (0=real, 1=fake) for each article",
    )
    n_bins: int = Field(default=15, ge=2, le=100, description="Number of calibration bins (2–100)")


class CalibrationMetricsResponse(BaseModel):
    ece: float = Field(..., description="Expected Calibration Error (lower is better)")
    mce: float = Field(..., description="Maximum Calibration Error (lower is better)")
    brier_score: float = Field(..., description="Brier Score (lower is better)")
    nll: float = Field(..., description="Negative Log-Likelihood (lower is better)")
    n_samples: int
    n_bins: int


# ── Ensemble models ────────────────────────────────────────────────────────────

class EnsemblePredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_probabilities": [
                    [0.7, 0.3],
                    [0.4, 0.6],
                    [0.6, 0.4],
                ],
                "weights": [0.5, 0.3, 0.2],
                "strategy": "weighted_average",
            }
        }
    )
    model_probabilities: List[List[float]] = Field(
        ...,
        description=(
            "Probability vectors from each model [[p_real, p_fake], ...]. "
            "Each inner list must have exactly 2 values that sum to 1."
        ),
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="Per-model weights for 'weighted_average' strategy. Must match length of model_probabilities.",
    )
    strategy: str = Field(
        default="average",
        description="Combination strategy: 'average', 'weighted_average', or 'majority_vote'",
    )


class EnsemblePredictResponse(BaseModel):
    strategy: str
    ensemble_probabilities: List[float] = Field(
        ..., description="Combined [p_real, p_fake] probability pair"
    )
    prediction: str
    fake_probability: float
    confidence: float
    num_models: int


# ── Export models ──────────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"output_path": "exports/model.onnx"}}
    )
    output_path: str = Field(
        ...,
        description="Destination file path for the exported model artifact",
    )


class ExportResponse(BaseModel):
    format: str
    output_path: str
    success: bool
    message: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _preview_text(text: str) -> str:
    if len(text) <= TEXT_PREVIEW_CHARS:
        return text
    return text[:TEXT_PREVIEW_CHARS] + "..."


def _safe_run(fn, *args, **kwargs) -> dict:
    """Call an analysis function; return empty dict on any error so other
    sections of the /analyze response are unaffected."""
    try:
        result = fn(*args, **kwargs)
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        name = getattr(fn, "__qualname__", type(fn).__name__)
        logger.warning("Analysis step '%s' failed: %s", name, exc)
        return {}


def _serialize_graph_result(result: dict) -> dict:
    """Convert any numpy arrays in a graph pipeline result to Python lists
    so the response is JSON-serializable."""
    out = {}
    for k, v in result.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _build_project_view() -> dict[str, Any]:
    src_dir = PROJECT_ROOT / "src"
    model_dir = src_dir / "models"

    model_subpackages = {}
    for subpackage in MODEL_SUBPACKAGES:
        package_dir = model_dir / subpackage
        model_subpackages[subpackage] = {
            "directory_exists": package_dir.exists(),
            "package_init_exists": (package_dir / "__init__.py").exists(),
        }

    return {
        "project_root": str(PROJECT_ROOT),
        "api": {
            "title": APP_TITLE,
            "version": APP_VERSION,
            "description": APP_DESCRIPTION,
        },
        "config": {
            "model_name": SETTINGS.model.name,
            "model_path": str(MODEL_PATH),
            "training_text_column": TRAINING_TEXT_COLUMN,
            "vectorizer_path": str(VECTORIZER_PATH),
        },
        "structure": {
            "src_exists": src_dir.exists(),
            "api_exists": (PROJECT_ROOT / "api").exists(),
            "config_exists": (PROJECT_ROOT / "config").exists(),
            "tests_exists": (PROJECT_ROOT / "tests").exists(),
            "models_package_init_exists": (model_dir / "__init__.py").exists(),
            "model_subpackages": model_subpackages,
        },
    }


def _decode_prediction_result(prediction_result) -> tuple[float, str, float]:
    """Normalise the output of models.inference.predictor.predict into
    (fake_probability, prediction_label, confidence)."""
    if isinstance(prediction_result, dict):
        prob = float(prediction_result.get("fake_probability", 0.0))
        prediction = str(prediction_result.get("label", "Fake")).upper()
        confidence = float(prediction_result.get("confidence", max(prob, 1 - prob)))
    else:
        prob = float(prediction_result)
        prediction = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else (1 - prob)
    return prob, prediction, confidence


def _heuristic_predict_fn(text: str) -> dict:
    """Model-free fallback predict function using lexicon-based features.

    Used when the ML model has not been trained yet. Combines bias score and
    emotion intensity to produce a ``fake_probability`` estimate so that the
    explainability pipeline (LIME, aggregation, etc.) can still run.
    """
    try:
        bias = compute_bias_features(text)
        bias_score = float(getattr(bias, "bias_score", 0.0))
    except Exception:
        bias_score = 0.0
    try:
        emo = EMOTION_ANALYZER.analyze(text)
        emo_scores: dict = getattr(emo, "emotion_scores", {}) or {}
        emo_intensity = (
            sum(emo_scores.values()) / len(emo_scores) if emo_scores else 0.0
        )
    except Exception:
        emo_intensity = 0.0

    fake_prob = min(max(0.5 * bias_score + 0.3 * emo_intensity + 0.1, 0.05), 0.95)
    return {
        "fake_probability": round(fake_prob, 4),
        "label": "FAKE" if fake_prob > 0.5 else "REAL",
        "prediction": "FAKE" if fake_prob > 0.5 else "REAL",
        "confidence": round(max(fake_prob, 1.0 - fake_prob), 4),
        "source": "heuristic_fallback",
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "message": APP_TITLE,
        "status": "online",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "analyze": "/analyze",
            "explain": "/explain",
            "report": "/report",
            "inference_model_info": "/inference/model-info",
            "cache_clear": "/cache/clear",
            "calibration_info": "/calibration/info",
            "calibration_metrics": "/calibration/metrics",
            "ensemble_info": "/ensemble/info",
            "ensemble_predict": "/ensemble/predict",
            "export_info": "/export/info",
            "export_onnx": "/export/onnx",
            "export_torchscript": "/export/torchscript",
            "health": "/health",
            "project_view": "/project-view",
            "docs": "/docs",
        },
    }


@app.get("/project-view")
def project_view():
    """Project-level view of API metadata, configuration, and package layout."""
    return _build_project_view()


@app.post("/predict", response_model=NewsResponse)
def predict_news(request: NewsRequest):
    """
    Predict whether a news article is fake or real.

    Results are cached in memory for one hour — repeated submissions of
    identical text are served instantly without rerunning the model.
    """
    try:
        text = ensure_non_empty_text(request.text, name="request.text")
        logger.info("Received /predict request (text length: %d)", len(text))
        timer_start = INFERENCE_LOGGER.start_timer()

        # ── Cache lookup ───────────────────────────────────────────────────────
        cached = INFERENCE_CACHE.get(text)
        if cached is not None:
            logger.debug("Cache hit for /predict")
            return NewsResponse(**cached)

        # ── Model inference ────────────────────────────────────────────────────
        prediction_result = predict(text)
        prob, prediction, confidence = _decode_prediction_result(prediction_result)

        # ── Format via ResultFormatter ─────────────────────────────────────────
        raw_prediction = {
            "bias": None,
            "ideology": None,
            "propaganda_probability": None,
            "credibility_score": round(1.0 - prob, 4),
        }
        RESULT_FORMATTER.format_api_response(raw_prediction)  # validates schema

        response_data = {
            "text": _preview_text(text),
            "fake_probability": round(prob, 4),
            "prediction": prediction,
            "confidence": round(confidence, 4),
        }

        # ── Store in cache ─────────────────────────────────────────────────────
        INFERENCE_CACHE.set(text, response_data)

        # ── Structured inference log ───────────────────────────────────────────
        INFERENCE_LOGGER.log_prediction(
            article_id=None,
            start_time=timer_start,
            model_versions={"roberta": SETTINGS.model.name},
            feature_count=0,
            prediction_confidence=round(confidence, 4),
        )

        logger.info("Prediction: %s (confidence: %.4f)", prediction, confidence)
        return NewsResponse(**response_data)

    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        raise HTTPException(status_code=503, detail="Model not available. Please train the model first.")
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.post("/batch-predict", response_model=BatchNewsResponse)
def batch_predict_news(request: BatchNewsRequest):
    """
    Predict fake/real for a batch of news articles (up to 50 at a time).

    Each text is individually checked against the cache before running the
    model, so cached entries are served without additional inference cost.
    """
    try:
        normalized_texts = ensure_non_empty_text_list(request.texts, name="request.texts")
        logger.info("Received /batch-predict request (%d texts)", len(normalized_texts))

        results: list[NewsResponse] = []
        cache_hits = 0
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # ── Separate cache hits from texts that need inference ─────────────────
        for i, text in enumerate(normalized_texts):
            cached = INFERENCE_CACHE.get(text)
            if cached is not None:
                results.append(NewsResponse(**cached))
                cache_hits += 1
            else:
                results.append(None)  # placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # ── Run batch inference on uncached texts ──────────────────────────────
        if uncached_texts:
            timer_start = INFERENCE_LOGGER.start_timer()

            engine = _get_inference_engine()

            if engine is not None:
                # Use InferenceEngine for batch inference
                engine_results = engine.predict(uncached_texts)
                for idx, engine_result in zip(uncached_indices, engine_results):
                    text = normalized_texts[idx]
                    probs = engine_result.probabilities or [0.5, 0.5]
                    fake_index = 1
                    prob = round(float(probs[fake_index]), 4)
                    confidence = round(max(probs), 4)
                    prediction = "FAKE" if prob > 0.5 else "REAL"
                    response_data = {
                        "text": _preview_text(text),
                        "fake_probability": prob,
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                    INFERENCE_CACHE.set(text, response_data)
                    results[idx] = NewsResponse(**response_data)
            else:
                # Fall back to existing predict_batch function
                batch_probs = predict_batch(uncached_texts)
                for idx, probs, text in zip(uncached_indices, batch_probs, uncached_texts):
                    prob_real, prob_fake = float(probs[0]), float(probs[1])
                    prob = round(prob_fake, 4)
                    confidence = round(max(prob_real, prob_fake), 4)
                    prediction = "FAKE" if prob > 0.5 else "REAL"
                    response_data = {
                        "text": _preview_text(text),
                        "fake_probability": prob,
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                    INFERENCE_CACHE.set(text, response_data)
                    results[idx] = NewsResponse(**response_data)

            INFERENCE_LOGGER.log_prediction(
                article_id=None,
                start_time=timer_start,
                model_versions={"roberta": SETTINGS.model.name},
                feature_count=0,
                prediction_confidence=None,
            )

        return BatchNewsResponse(
            results=results,
            total=len(normalized_texts),
            cache_hits=cache_hits,
        )

    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        raise HTTPException(status_code=503, detail="Model not available. Please train the model first.")
    except ValueError as exc:
        logger.error("Invalid batch input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")


@app.get("/health")
def health_check():
    """Detailed health check."""
    try:
        model_exists = MODEL_PATH.exists()
        vectorizer_required = TRAINING_TEXT_COLUMN == "engineered_text"
        vectorizer_exists = (not vectorizer_required) or VECTORIZER_PATH.exists()
        vectorizer_fallback_enabled = INFERENCE_ALLOW_RAW_TEXT_FALLBACK
        vectorizer_effective_ready = (
            vectorizer_exists
            if not vectorizer_required
            else (vectorizer_exists or vectorizer_fallback_enabled)
        )

        required_files = ["config.json", "tokenizer.json"]
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        has_weight_file = any((MODEL_PATH / f).exists() for f in weight_files) if model_exists else False
        model_files_exist = (
            all((MODEL_PATH / f).exists() for f in required_files) and has_weight_file
            if model_exists
            else False
        )

        cache_size = len(INFERENCE_CACHE.memory_cache)

        return {
            "status": (
                "healthy"
                if model_exists and model_files_exist and vectorizer_effective_ready
                else "degraded"
            ),
            "model_path": str(MODEL_PATH),
            "model_exists": model_exists,
            "model_files_complete": model_files_exist,
            "training_text_column": TRAINING_TEXT_COLUMN,
            "vectorizer_required": vectorizer_required,
            "vectorizer_exists": vectorizer_exists,
            "vectorizer_fallback_enabled": vectorizer_fallback_enabled,
            "vectorizer_effective_ready": vectorizer_effective_ready,
            "vectorizer_path": str(VECTORIZER_PATH),
            "inference_cache_entries": cache_size,
        }
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return {
            "status": "unhealthy",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_news(request: NewsRequest):
    """
    Unified deep-analysis endpoint.

    Returns model prediction plus the full suite of linguistic, narrative,
    framing, rhetoric, discourse, propaganda-pattern, and credibility-profile
    analyses.

    Full analysis results are cached for one hour so repeated requests for
    the same text are served without re-running all subsystems.
    """
    try:
        text = ensure_non_empty_text(request.text, name="request.text")
        logger.info("Received /analyze request (text length: %d)", len(text))
        timer_start = INFERENCE_LOGGER.start_timer()

        # ── Cache lookup ───────────────────────────────────────────────────────
        cache_key = f"analyze:{text}"
        cached = INFERENCE_CACHE.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for /analyze")
            return AnalysisResponse(**cached)

        # ── 1. Model prediction ────────────────────────────────────────────────
        # Use the ML model when available; fall back to a lexicon-based
        # heuristic so every downstream linguistic / explainability step
        # still runs even before the model has been trained.
        _model_unavailable = False
        try:
            prediction_result = predict(text)
            fake_probability, prediction, confidence = _decode_prediction_result(prediction_result)
            _analyze_predict_fn = predict_batch
        except FileNotFoundError:
            _model_unavailable = True
            _fallback = _heuristic_predict_fn(text)
            fake_probability = _fallback["fake_probability"]
            prediction = _fallback["prediction"]
            confidence = _fallback["confidence"]
            _analyze_predict_fn = _heuristic_predict_fn

        # ── 2. Bias + emotion (lexicon-based) ─────────────────────────────────
        bias_result = compute_bias_features(text)
        emotion_result = EMOTION_ANALYZER.analyze(text)
        emotion_scores: dict[str, float] = getattr(emotion_result, "emotion_scores", {})

        # ── 3. Narrative analysis ──────────────────────────────────────────────
        narrative_roles: dict = _safe_run(NARRATIVE_ROLE_EXTRACTOR.analyze, request.text)
        hero_entities: list = narrative_roles.get("hero_entities", [])
        villain_entities: list = narrative_roles.get("villain_entities", [])
        victim_entities: list = narrative_roles.get("victim_entities", [])

        narrative_conflict: dict = _safe_run(
            NARRATIVE_CONFLICT_ANALYZER.analyze,
            request.text,
            hero_entities=hero_entities,
            villain_entities=villain_entities,
            victim_entities=victim_entities,
        )
        narrative_propagation: dict = _safe_run(
            NARRATIVE_PROPAGATION_ANALYZER.analyze,
            request.text,
            hero_entities=hero_entities,
            villain_entities=villain_entities,
            victim_entities=victim_entities,
        )
        narrative_temporal: dict = _safe_run(NARRATIVE_TEMPORAL_ANALYZER.analyze, request.text)

        # ── 4. Framing ─────────────────────────────────────────────────────────
        framing: dict = _safe_run(FRAMING_ANALYZER.analyze, request.text)

        # ── 5. Rhetoric + argument structure ──────────────────────────────────
        rhetorical: dict = _safe_run(RHETORICAL_DETECTOR.analyze, request.text)
        argument: dict = _safe_run(ARGUMENT_ANALYZER.analyze, request.text)

        # ── 6. Discourse-level analyses ────────────────────────────────────────
        info_density: dict = _safe_run(INFO_DENSITY_ANALYZER.analyze, request.text)
        info_omission: dict = _safe_run(INFO_OMISSION_DETECTOR.analyze, request.text)
        context_omission: dict = _safe_run(CONTEXT_OMISSION_DETECTOR.analyze, request.text)
        discourse_coherence: dict = _safe_run(DISCOURSE_ANALYZER.analyze, request.text)
        ideological: dict = _safe_run(IDEOLOGICAL_DETECTOR.analyze, request.text)
        emotion_target: dict = _safe_run(EMOTION_TARGET_ANALYZER.analyze, request.text)
        source_attribution: dict = _safe_run(SOURCE_ATTRIBUTION_ANALYZER.analyze, request.text)

        # ── 7. Propaganda pattern detection ───────────────────────────────────
        combined_narrative: dict = {**narrative_conflict, **narrative_propagation, **narrative_temporal}
        combined_info: dict = {**info_density, **info_omission}
        propaganda_patterns: dict = _safe_run(
            PROPAGANDA_PATTERN_DETECTOR.analyze,
            emotion_features=emotion_scores,
            narrative_features=combined_narrative,
            rhetorical_features=rhetorical,
            argument_features=argument,
            information_features=combined_info,
        )

        # ── 8. Credibility profile ─────────────────────────────────────────────
        combined_discourse: dict = {
            **discourse_coherence,
            **context_omission,
            **info_density,
            **info_omission,
            **source_attribution,
        }
        credibility_profile: dict = _safe_run(
            BIAS_PROFILE_BUILDER.build_profile,
            bias={"bias_score": float(bias_result.bias_score)},
            emotion=emotion_scores,
            narrative=combined_narrative,
            discourse=combined_discourse,
            ideology=ideological,
        )

        # ── 9. Graph analysis ──────────────────────────────────────────────────
        raw_graph_result: dict = _safe_run(GRAPH_PIPELINE.run, request.text)
        graph_result: dict = _serialize_graph_result(raw_graph_result)

        entity_graph: dict = graph_result.get("entity_graph", {})
        entity_embeddings: list = []
        if entity_graph:
            try:
                embedding_arr = GRAPH_EMBEDDING_GENERATOR.generate_embedding(entity_graph)
                entity_embeddings = embedding_arr.tolist()
            except Exception as emb_err:
                logger.warning("Entity graph embedding failed: %s", emb_err)

        raw_temporal = _safe_run(
            lambda t: TEMPORAL_GRAPH_ANALYZER.analyze(t).to_dict(), request.text
        )

        graph_analysis: dict = {
            "entity_graph": entity_graph,
            "entity_graph_metrics": graph_result.get("entity_graph_metrics", {}),
            "entity_embeddings": entity_embeddings,
            "narrative_graph": graph_result.get("narrative_graph", {}),
            "narrative_graph_metrics": graph_result.get("narrative_graph_metrics", {}),
            "graph_features": graph_result.get("graph_features", {}),
            "temporal_graph": raw_temporal,
        }

        # ── 10. Explainability ─────────────────────────────────────────────────
        emotion_explanation = _safe_run(explain_emotion, request.text)
        try:
            lime_result = explain_prediction(
                _analyze_predict_fn,
                request.text,
                num_features=8,
                num_samples=LIME_NUM_SAMPLES,
            )
        except Exception as lime_error:
            logger.warning("LIME explanation unavailable: %s", lime_error)
            lime_result = {
                "text": request.text,
                "important_features": [],
                "error": "lime_unavailable",
            }

        # ── 11. Build response dict ────────────────────────────────────────────
        response_data: dict[str, Any] = {
            "text": _preview_text(request.text),
            "prediction": prediction,
            "fake_probability": round(fake_probability, 4),
            "confidence": round(confidence, 4),
            "bias": {
                "bias_score": round(float(bias_result.bias_score), 4),
                "media_bias": bias_result.media_bias,
                "biased_tokens": bias_result.biased_tokens,
                "sentence_heatmap": bias_result.sentence_heatmap,
            },
            "emotion": {
                "dominant_emotion": emotion_result.dominant_emotion,
                "emotion_scores": emotion_scores,
                "emotion_distribution": emotion_scores,
            },
            "narrative": {
                "roles": narrative_roles,
                "conflict": narrative_conflict,
                "propagation": narrative_propagation,
                "temporal": narrative_temporal,
            },
            "framing": framing,
            "rhetoric": {
                "rhetorical_devices": rhetorical,
                "argument_structure": argument,
            },
            "discourse": {
                "coherence": discourse_coherence,
                "context_omission": context_omission,
                "information_density": info_density,
                "information_omission": info_omission,
                "source_attribution": source_attribution,
                "ideological_language": ideological,
                "emotion_targets": emotion_target,
            },
            "propaganda_analysis": propaganda_patterns,
            "credibility_profile": credibility_profile,
            "graph_analysis": graph_analysis,
            "explainability": {
                "emotion_explanation": emotion_explanation,
                "lime": lime_result,
            },
        }

        # ── Cache the full analysis result ─────────────────────────────────────
        INFERENCE_CACHE.set(cache_key, response_data)

        # ── Structured inference log ───────────────────────────────────────────
        credibility_score = credibility_profile.get("credibility_score")
        INFERENCE_LOGGER.log_prediction(
            article_id=None,
            start_time=timer_start,
            model_versions={"roberta": SETTINGS.model.name},
            feature_count=len(combined_info) + len(combined_narrative),
            prediction_confidence=round(confidence, 4),
        )

        return AnalysisResponse(**response_data)

    except FileNotFoundError as exc:
        logger.error("Model not found during analysis: %s", exc)
        raise HTTPException(status_code=503, detail="Model not available. Please train the model first.")
    except ValueError as exc:
        logger.error("Invalid analysis input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Analysis error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during analysis")


@app.post("/explain")
def explain_article(request: NewsRequest):
    """
    Full explainability pipeline with credibility aggregation.

    Runs LIME token attribution, emotion explanation, graph-based explanation,
    and the ExplanationAggregator end-to-end.  The aggregated token importances
    are then fed into the TruthLens credibility AggregationPipeline to produce
    a complete risk + credibility profile alongside the token-level explanation.

    Works with or without a trained ML model — when the model is absent a
    lexicon-based heuristic predict function is used automatically.
    """
    try:
        text = ensure_non_empty_text(request.text, name="request.text")
        logger.info("Received /explain request (text length: %d)", len(text))

        # ── Choose predict_fn (ML model or heuristic fallback) ─────────────────
        engine = _get_inference_engine()
        if engine is not None:
            def _model_predict(t: str) -> dict:
                try:
                    result = predict(t)
                    return result if isinstance(result, dict) else {"fake_probability": float(result)}
                except Exception:
                    return _heuristic_predict_fn(t)
            predict_fn = _model_predict
            predict_source = "model"
        else:
            predict_fn = _heuristic_predict_fn
            predict_source = "heuristic_fallback"

        logger.info("Explainability predict_fn source: %s", predict_source)

        # ── Run explainability pipeline ─────────────────────────────────────────
        expl_config = ExplainabilityConfig(
            enabled=True,
            use_lime=True,
            use_shap=False,
            use_bias_emotion=False,
            use_attention_rollout=False,
            use_graph_explainer=True,
            use_aggregation=True,
            use_consistency=True,
            use_explanation_metrics=True,
            cache_enabled=False,
        )

        expl_result = run_explainability_pipeline(
            text=text,
            predict_fn=predict_fn,
            config=expl_config,
        )

        # ── Credibility aggregation pipeline ────────────────────────────────────
        bias_result = compute_bias_features(text)
        emotion_result = EMOTION_ANALYZER.analyze(text)
        emotion_scores: dict = getattr(emotion_result, "emotion_scores", {}) or {}

        narrative_features: dict = _safe_run(NARRATIVE_CONFLICT_ANALYZER.analyze, text)
        discourse_features: dict = _safe_run(DISCOURSE_ANALYZER.analyze, text)

        credibility_profile: dict = _safe_run(
            BIAS_PROFILE_BUILDER.build_profile,
            bias={"bias_score": float(bias_result.bias_score)},
            emotion=emotion_scores,
            narrative=narrative_features,
            discourse=discourse_features,
            ideology={},
        )

        aggregation_result: dict = {}
        try:
            aggregation_result = AGGREGATION_PIPELINE.run(
                profile=credibility_profile or {},
                text=text,
            )
        except Exception as agg_err:
            logger.warning("Credibility aggregation failed in /explain: %s", agg_err)

        # ── Serialize explainability result ────────────────────────────────────
        def _ser(obj: Any) -> Any:
            if obj is None:
                return None
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "__dict__"):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            return obj

        base_pred = predict_fn(text)

        return {
            "text": _preview_text(text),
            "prediction": base_pred,
            "predict_source": predict_source,
            "explainability": {
                "lime": _ser(expl_result.lime_explanation),
                "aggregated": _ser(expl_result.aggregated_explanation),
                "consistency_metrics": expl_result.consistency_metrics,
                "explanation_metrics": expl_result.explanation_metrics,
                "explanation_quality_score": expl_result.explanation_quality_score,
                "emotion_explanation": _ser(expl_result.emotion_explanation),
                "module_failures": expl_result.module_failures,
                "metadata": expl_result.metadata,
            },
            "aggregation": aggregation_result,
        }

    except ValueError as exc:
        logger.error("Invalid /explain input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Explain pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Explainability pipeline error: {exc}")


@app.post("/report", response_model=ReportResponse)
def generate_report(request: NewsRequest):
    """
    Generate a structured analysis report for a news article.

    Runs bias and emotion analysis then packages the results into a
    standardised report using ReportGenerator.  Lighter than /analyze —
    no graph, LIME, or discourse sub-systems are invoked.
    """
    try:
        text = ensure_non_empty_text(request.text, name="request.text")
        logger.info("Received /report request (text length: %d)", len(text))

        # ── Cache lookup ───────────────────────────────────────────────────────
        cache_key = f"report:{text}"
        cached = INFERENCE_CACHE.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for /report")
            return ReportResponse(**cached)

        # ── Lightweight analysis ───────────────────────────────────────────────
        bias_result = compute_bias_features(text)
        emotion_result = EMOTION_ANALYZER.analyze(text)
        emotion_scores: dict[str, float] = getattr(emotion_result, "emotion_scores", {})

        narrative_roles: dict = _safe_run(NARRATIVE_ROLE_EXTRACTOR.analyze, text)
        combined_narrative: dict = {**_safe_run(NARRATIVE_CONFLICT_ANALYZER.analyze, text)}

        combined_discourse: dict = {**_safe_run(DISCOURSE_ANALYZER.analyze, text)}

        credibility_profile: dict = _safe_run(
            BIAS_PROFILE_BUILDER.build_profile,
            bias={"bias_score": float(bias_result.bias_score)},
            emotion=emotion_scores,
            narrative=combined_narrative,
            discourse=combined_discourse,
            ideology={},
        )

        credibility_score: Optional[float] = credibility_profile.get("credibility_score")

        # ── Generate report via ReportGenerator ────────────────────────────────
        report = REPORT_GENERATOR.generate_report(
            article_text=text,
            title=None,
            source=None,
            analysis={
                "bias": {
                    "bias_score": round(float(bias_result.bias_score), 4),
                    "media_bias": bias_result.media_bias,
                    "biased_tokens": bias_result.biased_tokens,
                },
                "emotion": {
                    "dominant_emotion": emotion_result.dominant_emotion,
                    "emotion_scores": emotion_scores,
                },
                "narrative": narrative_roles,
                "credibility_score": credibility_score,
            },
        )

        response_data = {
            "article_summary": report.get("article_summary", {}),
            "bias_analysis": report.get("bias_analysis", {}),
            "emotion_analysis": report.get("emotion_analysis", {}),
            "narrative_structure": report.get("narrative_structure", {}),
            "entity_graph": report.get("entity_graph", {}),
            "credibility_score": report.get("credibility_score"),
        }

        INFERENCE_CACHE.set(cache_key, response_data)

        return ReportResponse(**response_data)

    except ValueError as exc:
        logger.error("Invalid report input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Report generation error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during report generation")


@app.get("/inference/model-info", response_model=ModelInfoResponse)
def inference_model_info():
    """
    Return metadata about the InferenceEngine's loaded model.

    If the model directory does not yet exist (i.e. the model has not been
    trained), returns available=false with a descriptive message.
    """
    engine = _get_inference_engine()
    if engine is None:
        return ModelInfoResponse(
            available=False,
            model_path=str(MODEL_PATH),
            device=None,
            num_parameters=None,
            num_trainable_parameters=None,
            label_map=None,
        )

    try:
        info = engine.get_model_info()
        label_map = (
            {str(k): v for k, v in engine.label_map.items()}
            if engine.label_map
            else None
        )
        return ModelInfoResponse(
            available=True,
            model_path=info["model_path"],
            device=info["device"],
            num_parameters=info["num_parameters"],
            num_trainable_parameters=info["num_trainable_parameters"],
            label_map=label_map,
        )
    except Exception as exc:
        logger.error("Failed to retrieve model info: %s", exc)
        raise HTTPException(status_code=500, detail="Could not retrieve model information")


@app.post("/cache/clear")
def clear_inference_cache():
    """
    Clear all cached inference results.

    Useful after retraining the model to ensure stale predictions are
    not served from cache.
    """
    try:
        INFERENCE_CACHE.clear()
        logger.info("Inference cache cleared via /cache/clear")
        return {"message": "Inference cache cleared successfully"}
    except Exception as exc:
        logger.error("Cache clear failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to clear inference cache")


# ── Calibration endpoints ──────────────────────────────────────────────────────

@app.get("/calibration/info")
def calibration_info():
    """
    Describe the calibration strategies available in TruthLens.

    Returns metadata about each method — no ground-truth labels required.
    To evaluate your model's calibration, call POST /calibration/metrics.
    """
    return {
        "methods": {
            "temperature_scaling": {
                "class": "TemperatureScaler",
                "description": (
                    "Learns a single scalar temperature T that divides logits before "
                    "softmax.  T > 1 softens (reduces over-confidence), T < 1 sharpens. "
                    "Very fast to fit; requires only a validation set of logits + labels."
                ),
                "reference": "Guo et al. (2017) — 'On Calibration of Modern Neural Networks'",
                "parameters": {
                    "lr": TemperatureScalingConfig().lr,
                    "max_iter": TemperatureScalingConfig().max_iter,
                    "tolerance": TemperatureScalingConfig().tolerance,
                },
            },
            "isotonic_regression": {
                "class": "IsotonicCalibrator",
                "description": (
                    "Fits a non-parametric monotonically non-decreasing function per class "
                    "using scikit-learn IsotonicRegression.  More flexible than temperature "
                    "scaling but requires more calibration data.  Uses a one-vs-rest strategy."
                ),
                "reference": "Zadrozny & Elkan (2002) — 'Transforming Classifier Scores into Accurate Multiclass Probability Estimates'",
                "parameters": {
                    "out_of_bounds": IsotonicCalibrationConfig().out_of_bounds,
                    "increasing": IsotonicCalibrationConfig().increasing,
                },
            },
        },
        "metrics_endpoint": "POST /calibration/metrics",
        "note": (
            "Both calibration methods require a held-out validation set with ground-truth "
            "labels. They should be applied after training — not during live inference."
        ),
    }


@app.post("/calibration/metrics", response_model=CalibrationMetricsResponse)
def compute_calibration_metrics(request: CalibrationMetricsRequest):
    """
    Compute calibration quality metrics for a set of model predictions.

    Accepts probability distributions and corresponding ground-truth labels,
    then returns ECE, MCE, Brier Score, and NLL.  Useful for evaluating how
    well-calibrated the current model is before deciding whether to apply
    temperature scaling or isotonic regression.

    **probabilities** — one row per sample: [p_real, p_fake]
    **labels**        — ground-truth index per sample (0=real, 1=fake)
    """
    try:
        n = len(request.probabilities)
        if n == 0:
            raise HTTPException(status_code=400, detail="probabilities list must not be empty")
        if len(request.labels) != n:
            raise HTTPException(
                status_code=400,
                detail=f"probabilities has {n} rows but labels has {len(request.labels)} entries",
            )

        metrics_obj = CalibrationMetrics(CalibrationMetricConfig(n_bins=request.n_bins))

        probs_tensor = torch.tensor(request.probabilities, dtype=torch.float32)
        labels_tensor = torch.tensor(request.labels, dtype=torch.long)

        metrics = metrics_obj.compute_all_metrics(probs_tensor, labels_tensor)

        logger.info(
            "Calibration metrics computed for %d samples: ECE=%.4f MCE=%.4f",
            n,
            metrics["ece"],
            metrics["mce"],
        )

        return CalibrationMetricsResponse(
            ece=round(metrics["ece"], 6),
            mce=round(metrics["mce"], 6),
            brier_score=round(metrics["brier_score"], 6),
            nll=round(metrics["nll"], 6),
            n_samples=n,
            n_bins=request.n_bins,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.error("Calibration metrics input error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Calibration metrics computation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during calibration metric computation")


# ── Ensemble endpoints ─────────────────────────────────────────────────────────

@app.get("/ensemble/info")
def ensemble_info():
    """
    Describe the ensemble strategies supported by TruthLens.

    Because ensemble models require multiple pre-loaded PyTorch modules,
    they are built at training/evaluation time — not during live inference.
    This endpoint documents what's available and how to use it via the
    POST /ensemble/predict convenience endpoint.
    """
    return {
        "strategies": {
            "average": {
                "class": "EnsembleModel",
                "description": "Averages logits from all member models before applying softmax.",
                "requires_weights": False,
            },
            "weighted_average": {
                "class": "EnsembleModel / WeightedEnsembleModel",
                "description": (
                    "Each model's logits are multiplied by its assigned weight before "
                    "summation.  Weights are automatically normalised to sum to 1."
                ),
                "requires_weights": True,
            },
            "majority_vote": {
                "class": "EnsembleModel",
                "description": (
                    "Each model votes for a class; the class with the most votes wins. "
                    "Final probabilities are the vote counts divided by total votes."
                ),
                "requires_weights": False,
            },
            "stacking": {
                "class": "StackingEnsembleModel",
                "description": (
                    "Base models produce intermediate probability vectors which are "
                    "concatenated and fed to a trainable meta-learner for the final prediction."
                ),
                "requires_weights": False,
                "requires_meta_model": True,
            },
        },
        "predict_endpoint": "POST /ensemble/predict",
        "note": (
            "POST /ensemble/predict accepts raw probability vectors from multiple sources "
            "and combines them without needing physical model instances. "
            "For full nn.Module-level ensembling, use EnsembleModel or WeightedEnsembleModel "
            "directly in your training pipeline."
        ),
    }


@app.post("/ensemble/predict", response_model=EnsemblePredictResponse)
def ensemble_predict(request: EnsemblePredictRequest):
    """
    Combine probability predictions from multiple model sources.

    Accepts a list of [p_real, p_fake] probability vectors — one per model —
    and combines them using the chosen strategy.  No physical PyTorch models
    are required; this endpoint operates on already-computed probabilities,
    making it useful for model-agnostic ensembling at inference time.

    **strategies**: average, weighted_average, majority_vote
    """
    try:
        valid_strategies = {"average", "weighted_average", "majority_vote"}
        if request.strategy not in valid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy '{request.strategy}'. Must be one of {sorted(valid_strategies)}.",
            )

        n_models = len(request.model_probabilities)
        if n_models == 0:
            raise HTTPException(status_code=400, detail="model_probabilities must not be empty")

        for i, probs in enumerate(request.model_probabilities):
            if len(probs) != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"model_probabilities[{i}] must have exactly 2 values [p_real, p_fake]",
                )

        probs_tensor = torch.tensor(request.model_probabilities, dtype=torch.float32)

        if request.strategy == "average":
            combined = torch.mean(probs_tensor, dim=0)

        elif request.strategy == "weighted_average":
            if request.weights is None:
                raise HTTPException(
                    status_code=400,
                    detail="weights must be provided for 'weighted_average' strategy",
                )
            if len(request.weights) != n_models:
                raise HTTPException(
                    status_code=400,
                    detail=f"weights length ({len(request.weights)}) must match number of models ({n_models})",
                )
            weights_tensor = torch.tensor(request.weights, dtype=torch.float32)
            weight_sum = weights_tensor.sum()
            if weight_sum <= 0:
                raise HTTPException(status_code=400, detail="Sum of weights must be positive")
            weights_tensor = weights_tensor / weight_sum
            combined = torch.sum(probs_tensor * weights_tensor.unsqueeze(1), dim=0)

        else:  # majority_vote
            votes = torch.argmax(probs_tensor, dim=1)
            n_classes = probs_tensor.shape[1]
            vote_counts = torch.zeros(n_classes)
            for v in votes:
                vote_counts[v] += 1
            combined = vote_counts / vote_counts.sum()

        combined_list = combined.tolist()
        fake_prob = round(combined_list[1], 4)
        confidence = round(float(combined.max().item()), 4)
        prediction = "FAKE" if fake_prob > 0.5 else "REAL"

        logger.info(
            "Ensemble predict (%s, %d models): %s (fake_prob=%.4f)",
            request.strategy,
            n_models,
            prediction,
            fake_prob,
        )

        return EnsemblePredictResponse(
            strategy=request.strategy,
            ensemble_probabilities=[round(p, 4) for p in combined_list],
            prediction=prediction,
            fake_probability=fake_prob,
            confidence=confidence,
            num_models=n_models,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.error("Ensemble predict input error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Ensemble predict failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during ensemble prediction")


# ── Export endpoints ───────────────────────────────────────────────────────────

@app.get("/export/info")
def export_info():
    """
    Describe the model export and quantization formats supported by TruthLens.

    Exporting requires the model to be trained and available at the configured
    path.  Use POST /export/onnx or POST /export/torchscript to trigger an
    export once the model is ready.
    """
    engine = _get_inference_engine()
    model_ready = engine is not None

    return {
        "model_ready": model_ready,
        "model_path": str(MODEL_PATH),
        "formats": {
            "onnx": {
                "class": "ONNXExporter",
                "endpoint": "POST /export/onnx",
                "description": (
                    "Exports the model to ONNX (Open Neural Network Exchange) format. "
                    "Enables deployment on ONNX Runtime, TensorRT, CoreML, and other "
                    "hardware-accelerated inference engines."
                ),
                "config": {
                    "opset_version": ONNXExportConfig().opset_version,
                    "dynamic_batch": ONNXExportConfig().dynamic_batch,
                    "verify_export": False,
                },
                "dependencies": ["onnx", "onnxruntime"],
            },
            "torchscript": {
                "class": "TorchScriptExporter",
                "endpoint": "POST /export/torchscript",
                "description": (
                    "Exports the model to TorchScript (.pt) format using tracing. "
                    "Enables Python-free deployment via the PyTorch C++ runtime, "
                    "mobile environments, and high-performance inference services."
                ),
                "config": {
                    "method": TorchScriptExportConfig().method,
                    "verify_export": False,
                },
            },
        },
        "quantization": {
            "class": "QuantizationEngine",
            "methods": {
                "dynamic": "Quantizes weights at load time; activations at run time.  No calibration data needed.",
                "static": "Quantizes both weights and activations using calibration data. Requires a representative dataset.",
                "qat": "Quantization Aware Training — inserts fake-quantization nodes before fine-tuning.",
            },
            "note": (
                "Quantization is applied to the exported model artifact, not the live "
                "serving model. Integrate QuantizationEngine into your training/export pipeline."
            ),
        },
    }


@app.post("/export/onnx", response_model=ExportResponse)
def export_onnx(request: ExportRequest):
    """
    Export the loaded TruthLens model to ONNX format.

    Requires the model to be trained and the InferenceEngine to be available.
    The exported .onnx file is written to the path specified in output_path.
    """
    try:
        engine = _get_inference_engine()
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="Model not available. Train the model first before exporting.",
            )

        model = engine.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded in InferenceEngine")

        max_length = SETTINGS.model.max_length
        example_input = torch.zeros(1, max_length, dtype=torch.long)

        output_path = ONNX_EXPORTER.export(model, example_input, request.output_path)

        logger.info("ONNX export completed: %s", output_path)
        return ExportResponse(
            format="onnx",
            output_path=str(output_path),
            success=True,
            message=f"Model successfully exported to ONNX at '{output_path}'",
        )

    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.error("ONNX export failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"ONNX export failed: {exc}")
    except Exception as exc:
        logger.error("Unexpected ONNX export error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during ONNX export")


@app.post("/export/torchscript", response_model=ExportResponse)
def export_torchscript(request: ExportRequest):
    """
    Export the loaded TruthLens model to TorchScript (.pt) format.

    Requires the model to be trained and the InferenceEngine to be available.
    The exported .pt file is written to the path specified in output_path.
    """
    try:
        engine = _get_inference_engine()
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="Model not available. Train the model first before exporting.",
            )

        model = engine.model
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded in InferenceEngine")

        max_length = SETTINGS.model.max_length
        example_input = torch.zeros(1, max_length, dtype=torch.long)

        output_path = TORCHSCRIPT_EXPORTER.export(model, example_input, request.output_path)

        logger.info("TorchScript export completed: %s", output_path)
        return ExportResponse(
            format="torchscript",
            output_path=str(output_path),
            success=True,
            message=f"Model successfully exported to TorchScript at '{output_path}'",
        )

    except HTTPException:
        raise
    except RuntimeError as exc:
        logger.error("TorchScript export failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"TorchScript export failed: {exc}")
    except Exception as exc:
        logger.error("Unexpected TorchScript export error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during TorchScript export")
