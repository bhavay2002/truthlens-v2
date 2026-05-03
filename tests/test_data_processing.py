import pandas as pd
from src.data.clean_data import clean_text, clean_dataframe


def test_clean_text():

    text = "Check this URL: https://example.com and some CAPS!"
    cleaned = clean_text(text)

    assert "https" not in cleaned
    assert cleaned.islower()


def test_clean_dataframe():

    df = pd.DataFrame({
        'text': ['Sample news 1', 'Sample news 2', 'Sample news 1', None],
        'label': [0, 1, 0, 1]
    })

    cleaned = clean_dataframe(df)

    assert cleaned['text'].notnull().all()