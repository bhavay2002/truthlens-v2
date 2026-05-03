from __future__ import annotations

import pytest

from src.models.multitask.multitask_truthlens_model import (
    MultiTaskTruthLensConfig,
    MultiTaskTruthLensModel,
)


class TestMultiTaskModelClassAttributes:
    def test_bias_labels_are_binary(self) -> None:
        assert len(MultiTaskTruthLensModel.BIAS_LABELS) == MultiTaskTruthLensModel.NUM_BIAS
        assert MultiTaskTruthLensModel.NUM_BIAS == 2

    def test_ideology_labels_are_ternary(self) -> None:
        assert len(MultiTaskTruthLensModel.IDEOLOGY_LABELS) == MultiTaskTruthLensModel.NUM_IDEOLOGY
        assert MultiTaskTruthLensModel.NUM_IDEOLOGY == 3

    def test_propaganda_labels_are_binary(self) -> None:
        assert len(MultiTaskTruthLensModel.PROPAGANDA_LABELS) == MultiTaskTruthLensModel.NUM_PROPAGANDA
        assert MultiTaskTruthLensModel.NUM_PROPAGANDA == 2

    def test_narrative_labels_are_three(self) -> None:
        assert len(MultiTaskTruthLensModel.NARRATIVE_LABELS) == MultiTaskTruthLensModel.NUM_NARRATIVE
        assert MultiTaskTruthLensModel.NUM_NARRATIVE == 3

    def test_frame_labels_match_count(self) -> None:
        assert len(MultiTaskTruthLensModel.FRAME_LABELS) == MultiTaskTruthLensModel.NUM_NARRATIVE_FRAMES

    def test_emotion_count_is_twenty(self) -> None:
        assert MultiTaskTruthLensModel.NUM_EMOTIONS == 20

    def test_bias_labels_contain_expected_values(self) -> None:
        labels = [l.lower() for l in MultiTaskTruthLensModel.BIAS_LABELS]
        assert "bias" in labels or "non_bias" in labels

    def test_ideology_labels_contain_expected_values(self) -> None:
        labels = [l.lower() for l in MultiTaskTruthLensModel.IDEOLOGY_LABELS]
        assert "left" in labels
        assert "right" in labels

    def test_frame_labels_contain_expected_codes(self) -> None:
        frame_labels = MultiTaskTruthLensModel.FRAME_LABELS
        for code in ("RE", "HI", "CO", "MO", "EC"):
            assert code in frame_labels


class TestMultiTaskModelConfig:
    def test_default_config_values(self) -> None:
        config = MultiTaskTruthLensConfig()
        assert config.model_name == "roberta-base"
        assert config.pooling == "cls"
        assert 0.0 <= config.dropout <= 1.0

    def test_task_weights_default_to_one(self) -> None:
        config = MultiTaskTruthLensConfig()
        for weight in (
            config.bias_weight,
            config.ideology_weight,
            config.propaganda_weight,
            config.narrative_weight,
            config.emotion_weight,
        ):
            assert weight == pytest.approx(1.0)

    def test_custom_config_is_respected(self) -> None:
        config = MultiTaskTruthLensConfig(
            model_name="bert-base-uncased",
            pooling="mean",
            dropout=0.2,
        )
        assert config.model_name == "bert-base-uncased"
        assert config.pooling == "mean"
        assert config.dropout == pytest.approx(0.2)
