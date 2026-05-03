def test_legacy_imports_resolve():
    import features.pipelines.feature_pipeline as fp
    import models.checkpointing.checkpoint_manager as cm
    import models.registry.model_registry as mr
    import models.utils.model_utils as mu
    import pipelines.prediction_pipeline as pp  # noqa: F401

    assert hasattr(fp, "FeaturePipeline")
    assert hasattr(fp, "apply_feature_engineering")
    assert hasattr(cm, "CheckpointManager")
    assert hasattr(cm, "get_last_checkpoint")
    assert hasattr(mr, "ModelRegistry")
    assert hasattr(mu, "load_model")
