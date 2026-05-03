from __future__ import annotations

import importlib


def test_model_submodules_are_importable() -> None:
    modules = [
        "src.models.ideology.ideology_classifier",
        "src.models.narrative.narrative_detector",
        "src.models.propaganda.propaganda_detector",
        "src.models.multitask.multitask_truthlens_model",
        "src.models.emotion.load_emotion_model",
        "src.models.emotion.train_emotion_model",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name) is not None


def test_model_subpackage_re_exports() -> None:
    from src.models.emotion import EmotionModelLoader, EmotionTrainer
    from src.models.encoder import TransformerEncoder
    from src.models.ideology import IdeologyClassifier
    from src.models.multitask import MultiTaskTruthLensModel, TaskHead
    from src.models.narrative import NarrativeDetector
    from src.models.propaganda import PropagandaDetector

    assert EmotionModelLoader is not None
    assert EmotionTrainer is not None
    assert TransformerEncoder is not None
    assert IdeologyClassifier is not None
    assert MultiTaskTruthLensModel is not None
    assert TaskHead is not None
    assert NarrativeDetector is not None
    assert PropagandaDetector is not None

