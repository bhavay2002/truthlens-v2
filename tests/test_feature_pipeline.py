from src.features.pipelines.feature_pipeline import apply_feature_engineering


def test_apply_feature_engineering_adds_engineered_text(feature_dataset):

    df, vectorizer = apply_feature_engineering(
        feature_dataset,
        text_column="text",
        tfidf_max_features=20,
        top_terms_per_doc=2,
    )

    assert "engineered_text" in df.columns
    assert len(df) == len(feature_dataset)
    assert vectorizer is not None