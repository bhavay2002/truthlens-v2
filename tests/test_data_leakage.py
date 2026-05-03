from sklearn.model_selection import train_test_split


def test_no_data_leakage(feature_dataset):

    train, test = train_test_split(
        feature_dataset,
        test_size=0.2,
        random_state=42
    )

    train_texts = set(train["text"])
    test_texts = set(test["text"])

    intersection = train_texts.intersection(test_texts)

    assert len(intersection) == 0