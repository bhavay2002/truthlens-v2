import pytest

from src.inference.feature_preparer import FeaturePreparer, FeaturePreparationConfig


def test_prepare_batch_empty_raises():
    preparer = FeaturePreparer(
        FeaturePreparationConfig(feature_schema=["text_length"], derive_graph_features=False)
    )
    with pytest.raises(ValueError):
        preparer.prepare_batch([])
