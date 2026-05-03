import pandas as pd
from sklearn.model_selection import train_test_split


def test_dataset_split_no_overlap():

    df = pd.DataFrame({
        "text": [f"text {i}" for i in range(50)],
        "label": [i % 2 for i in range(50)]
    })

    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train_set = set(train["text"])
    test_set = set(test["text"])

    assert train_set.intersection(test_set) == set()