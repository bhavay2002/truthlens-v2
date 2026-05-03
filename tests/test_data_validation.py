import pandas as pd
from src.data.validate_data import DataValidator


def test_validator_schema():

    validator = DataValidator(required_columns=['text', 'label'])

    df = pd.DataFrame({'text': ['sample'], 'label': [0]})

    assert validator.validate_schema(df)


def test_validator_label_specs_range_checks() -> None:

    validator = DataValidator(
        required_columns=["text", "bias", "emotion"],
        label_columns=["bias", "emotion"],
        label_specs={
            "bias": {"allowed_values": [0, 1]},
            # EMOTION-11: max valid emotion index is now 10 (was 19).
            "emotion": {"min_value": 0, "max_value": 10},
        },
    )

    df = pd.DataFrame(
        {
            "text": ["sample one", "sample two"],
            "bias": [0, 2],
            # 11 is out-of-range under the new 11-class schema.
            "emotion": [3, 11],
        }
    )

    results = validator.validate(df)

    assert not results["labels_valid"]
    assert any("Invalid values in 'bias'" in err for err in results["errors"])
    assert any("above max_value=10 in 'emotion'" in err for err in results["errors"])
