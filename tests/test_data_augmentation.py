import pytest
from src.data.data_augmentation import augment_dataset


def test_augment_dataset_multiplier_one_is_noop(sample_dataset):

    augmented = augment_dataset(
        sample_dataset,
        text_column="text",
        multiplier=1,
    )

    assert len(augmented) == len(sample_dataset)


def test_augment_dataset_invalid_multiplier(sample_dataset):

    with pytest.raises(ValueError):
        augment_dataset(
            sample_dataset,
            text_column="text",
            multiplier=0,
        )