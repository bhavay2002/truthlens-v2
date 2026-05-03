"""Compatibility exports for emotion model helpers."""

from .load_emotion_model import EmotionModelLoader
from .train_emotion_model import EmotionTrainer

__all__ = ["EmotionModelLoader", "EmotionTrainer"]
