# Source Structure

Below is the directory structure of `src/`, excluding all `__pycache__` folders.

```text
src/
|-- aggregation
|   |-- __init__.py
|   |-- aggregation_config.py
|   |-- aggregation_metrics.py
|   |-- aggregation_pipeline.py
|   |-- aggregation_validator.py
|   |-- calibration.py
|   |-- feature_mapper.py
|   |-- risk_assessment.py
|   |-- score_explainer.py
|   |-- score_normalizer.py
|   |-- score_schema.py
|   |-- truthlens_score_calculator.py
|   `-- weight_manager.py
|-- analysis
|   |-- __init__.py
|   |-- _text_features.py
|   |-- analysis_config.py
|   |-- analysis_pipeline.py
|   |-- analysis_registry.py
|   |-- argument_mining.py
|   |-- base_analyzer.py
|   |-- batch_processor.py
|   |-- bias_profile_builder.py
|   |-- context_omission_detector.py
|   |-- discourse_coherence_analyzer.py
|   |-- emotion_lexicon.py
|   |-- emotion_target_analysis.py
|   |-- feature_context.py
|   |-- feature_merger.py
|   |-- feature_schema.py
|   |-- framing_analysis.py
|   |-- ideological_language_detector.py
|   |-- information_density_analyzer.py
|   |-- information_omission_detector.py
|   |-- integration_runner.py
|   |-- label_analysis.py
|   |-- multitask_validator.py
|   |-- narrative_conflict.py
|   |-- narrative_propagation.py
|   |-- narrative_role_extractor.py
|   |-- narrative_temporal_analyzer.py
|   |-- orchestrator.py
|   |-- output_models.py
|   |-- preprocessing.py
|   |-- propaganda_pattern_detector.py
|   |-- rhetorical_device_detector.py
|   |-- source_attribution_analyzer.py
|   `-- spacy_loader.py
|-- config
|   |-- config_loader.py
|   |-- settings_loader.py
|   `-- task_config.py
|-- data
|   |-- class_balance.py
|   |-- collate.py
|   |-- data_augmentation.py
|   |-- data_cache.py
|   |-- data_cleaning.py
|   |-- data_contracts.py
|   |-- data_loader.py
|   |-- data_pipeline.py
|   |-- data_profiler.py
|   |-- data_resolver.py
|   |-- data_validator.py
|   |-- dataloader_factory.py
|   |-- dataset.py
|   |-- dataset_factory.py
|   |-- file_integrity.py
|   |-- leakage_checker.py
|   `-- samplers.py
|-- evaluation
|   |-- __init__.py
|   |-- advanced_analysis.py
|   |-- calibration.py
|   |-- error_analysis.py
|   |-- evaluate_model.py
|   |-- evaluate_saved_model.py
|   |-- evaluation_dashboard.py
|   |-- evaluation_engine.py
|   |-- evaluation_pipeline.py
|   |-- evaluator.py
|   |-- metrics_engine.py
|   |-- mlflow_tracker.py
|   |-- pdf_report.py
|   |-- prediction_collector.py
|   |-- reliability_diagram.py
|   |-- report_writer.py
|   |-- task_correlation.py
|   |-- threshold_optimizer.py
|   `-- uncertainty.py
|-- explainability
|   |-- __init__.py
|   |-- attention_rollout.py
|   |-- attention_visualizer.py
|   |-- bias_explainer.py
|   |-- common_schema.py
|   |-- emotion_explainer.py
|   |-- explainability_pipeline.py
|   |-- explanation_aggregator.py
|   |-- explanation_cache.py
|   |-- explanation_calibrator.py
|   |-- explanation_consistency.py
|   |-- explanation_metrics.py
|   |-- explanation_monitor.py
|   |-- explanation_report_generator.py
|   |-- explanation_visualizer.py
|   |-- lime_explainer.py
|   |-- model_explainer.py
|   |-- orchestrator.py
|   |-- propaganda_explainer.py
|   |-- shap_explainer.py
|   |-- token_alignment.py
|   `-- utils_validation.py
|-- features
|   |-- analysis
|   |   |-- __init__.py
|   |   `-- analysis_adapter_features.py
|   |-- base
|   |   |-- __init__.py
|   |   |-- base_feature.py
|   |   |-- feature_config.py
|   |   `-- feature_registry.py
|   |-- bias
|   |   |-- __init__.py
|   |   |-- bias_features.py
|   |   |-- bias_lexicon.py
|   |   |-- bias_lexicon_features.py
|   |   |-- framing_features.py
|   |   `-- ideological_features.py
|   |-- cache
|   |   |-- __init__.py
|   |   |-- cache_manager.py
|   |   `-- feature_cache.py
|   |-- discourse
|   |   |-- __init__.py
|   |   |-- argument_structure_features.py
|   |   `-- discourse_features.py
|   |-- emotion
|   |   |-- __init__.py
|   |   |-- emotion_features.py
|   |   |-- emotion_intensity_features.py
|   |   |-- emotion_lexicon.py
|   |   |-- emotion_lexicon_features.py
|   |   |-- emotion_schema.py
|   |   |-- emotion_target_features.py
|   |   `-- emotion_trajectory_features.py
|   |-- fusion
|   |   |-- __init__.py
|   |   |-- feature_fusion.py
|   |   |-- feature_merger.py
|   |   |-- feature_scaling.py
|   |   `-- feature_selection.py
|   |-- graph
|   |   |-- __init__.py
|   |   |-- entity_graph_features.py
|   |   `-- interaction_graph_features.py
|   |-- importance
|   |   |-- __init__.py
|   |   |-- feature_ablation.py
|   |   |-- permutation_importance.py
|   |   `-- shap_importance.py
|   |-- narrative
|   |   |-- __init__.py
|   |   |-- conflict_features.py
|   |   |-- narrative_features.py
|   |   |-- narrative_frame_features.py
|   |   `-- narrative_role_features.py
|   |-- pipelines
|   |   |-- __init__.py
|   |   |-- batch_feature_pipeline.py
|   |   |-- feature_engineering_pipeline.py
|   |   |-- feature_pipeline.py
|   |   `-- feature_schema.py
|   |-- propaganda
|   |   |-- __init__.py
|   |   |-- manipulation_patterns.py
|   |   |-- propaganda_features.py
|   |   `-- propaganda_lexicon_features.py
|   |-- text
|   |   |-- __init__.py
|   |   |-- lexical_features.py
|   |   |-- semantic_features.py
|   |   |-- syntactic_features.py
|   |   `-- token_features.py
|   |-- utills
|   |   `-- tfidf_engineering.py
|   |-- __init__.py
|   |-- dataset_feature_generator.py
|   |-- feature_bootstrap.py
|   |-- feature_pruning.py
|   |-- feature_report.py
|   |-- feature_schema.py
|   |-- feature_schema_validator.py
|   `-- feature_statistics.py
|-- graph
|   |-- __init__.py
|   |-- entity_graph.py
|   |-- graph_analysis.py
|   |-- graph_config.py
|   |-- graph_embeddings.py
|   |-- graph_explainer.py
|   |-- graph_features.py
|   |-- graph_pipeline.py
|   |-- graph_schema.py
|   |-- graph_utils.py
|   |-- graph_visualization.py
|   |-- narrative_graph_builder.py
|   `-- temporal_graph.py
|-- inference
|   |-- __init__.py
|   |-- analyze_article.py
|   |-- batch_inference.py
|   |-- drift_detection.py
|   |-- feature_preparer.py
|   |-- inference_cache.py
|   |-- inference_config.py
|   |-- inference_engine.py
|   |-- inference_logger.py
|   |-- inference_pipeline.py
|   |-- model_loader.py
|   |-- monitoring.py
|   |-- postprocessing.py
|   |-- predict_api.py
|   |-- prediction_service.py
|   |-- report_generator.py
|   |-- result_formatter.py
|   |-- run_inference.py
|   `-- schema.py
|-- models
|   |-- adapters
|   |   |-- adapter_config.py
|   |   |-- adapter_layer.py
|   |   `-- lora_adapter.py
|   |-- base
|   |   |-- __init__.py
|   |   |-- base_classifier.py
|   |   |-- base_model.py
|   |   `-- multitask_base_model.py
|   |-- benchmarking
|   |   |-- benchmark_runner.py
|   |   `-- dataset_benchmarks.py
|   |-- calibration
|   |   |-- __init__.py
|   |   |-- calibration_metrics.py
|   |   |-- isotonic_calibration.py
|   |   `-- temperature_scaling.py
|   |-- checkpointing
|   |   |-- __init__.py
|   |   |-- artifact_manager.py
|   |   |-- checkpoint_manager.py
|   |   |-- integrity.py
|   |   |-- io_utils.py
|   |   |-- loader_utils.py
|   |   |-- metadata.py
|   |   |-- model_loader.py
|   |   |-- resolver.py
|   |   |-- schema.py
|   |   |-- selection.py
|   |   `-- validator.py
|   |-- config
|   |   |-- __init__.py
|   |   `-- model_config.py
|   |-- distillation
|   |   `-- knowledge_distillation.py
|   |-- emotion
|   |   |-- __init__.py
|   |   |-- load_emotion_model.py
|   |   `-- train_emotion_model.py
|   |-- encoder
|   |   |-- __init__.py
|   |   |-- encoder_config.py
|   |   |-- encoder_factory.py
|   |   `-- transformer_encoder.py
|   |-- ensemble
|   |   |-- __init__.py
|   |   |-- _utils.py
|   |   |-- ensemble_model.py
|   |   |-- stacking_ensemble.py
|   |   `-- weighted_ensemble.py
|   |-- export
|   |   |-- __init__.py
|   |   |-- onnx_export.py
|   |   |-- quantization.py
|   |   `-- torchscript_export.py
|   |-- heads
|   |   |-- __init__.py
|   |   |-- classification_head.py
|   |   |-- multilabel_head.py
|   |   |-- multitask_head.py
|   |   `-- regression_head.py
|   |-- ideology
|   |   |-- __init__.py
|   |   `-- ideology_classifier.py
|   |-- inference
|   |   |-- __init__.py
|   |   |-- model_wrapper.py
|   |   |-- prediction_output.py
|   |   `-- predictor.py
|   |-- interpretability
|   |   |-- attention_extractor.py
|   |   `-- gradient_hooks.py
|   |-- loss
|   |   |-- base_balancer.py
|   |   |-- coverage_tracker.py
|   |   |-- gradnorm.py
|   |   |-- loss_normalizer.py
|   |   |-- multitask_loss.py
|   |   |-- task_loss_router.py
|   |   `-- uncertainty.py
|   |-- metadata
|   |   |-- __init__.py
|   |   |-- model_card.py
|   |   |-- model_metadata.py
|   |   `-- model_versioning.py
|   |-- monitoring
|   |   |-- embedding_drift.py
|   |   `-- prediction_drift.py
|   |-- multitask
|   |   |-- __init__.py
|   |   |-- multitask_model.py
|   |   |-- multitask_output.py
|   |   `-- multitask_truthlens_model.py
|   |-- narrative
|   |   |-- __init__.py
|   |   `-- narrative_detector.py
|   |-- optimization
|   |   |-- lr_scheduler.py
|   |   `-- optimizer_factory.py
|   |-- propaganda
|   |   |-- __init__.py
|   |   `-- propaganda_detector.py
|   |-- registry
|   |   |-- __init__.py
|   |   |-- model_factory.py
|   |   `-- model_registry.py
|   |-- regularization
|   |   |-- adversarial_training.py
|   |   |-- label_smoothing.py
|   |   `-- mixup.py
|   |-- representation
|   |   |-- attention_pooling.py
|   |   |-- cls_pooling.py
|   |   |-- mean_pooling.py
|   |   `-- pooling.py
|   |-- tasks
|   |   |-- bias
|   |   |   |-- __init__.py
|   |   |   `-- bias_classifier.py
|   |   |-- emotion
|   |   |   |-- __init__.py
|   |   |   `-- emotion_classifier.py
|   |   |-- ideology
|   |   |   |-- __init__.py
|   |   |   `-- ideology_classifier.py
|   |   |-- narrative
|   |   |   |-- __init__.py
|   |   |   `-- narrative_detector.py
|   |   |-- propaganda
|   |   |   |-- __init__.py
|   |   |   `-- propaganda_detector.py
|   |   `-- __init__.py
|   |-- uncertainty
|   |   |-- ensemble_uncertainty.py
|   |   |-- mc_dropout.py
|   |   `-- uncertainty_head.py
|   |-- utils
|   |   |-- __init__.py
|   |   |-- model_utils.py
|   |   |-- parameter_count.py
|   |   `-- weight_initialization.py
|   `-- __init__.py
|-- pipelines
|   |-- __init__.py
|   |-- baseline_training.py
|   `-- truthlens_pipeline.py
|-- training
|   |-- __init__.py
|   |-- create_trainer_fn.py
|   |-- cross_validation.py
|   |-- distributed_engine.py
|   |-- evaluation_engine.py
|   |-- experiment_tracker.py
|   |-- hyperparameter_tuning.py
|   |-- instrumentation.py
|   |-- loss_engine.py
|   |-- loss_functions.py
|   |-- lr_scheduler_engine.py
|   |-- monitor_engine.py
|   |-- step_engine.py
|   |-- task_scheduler.py
|   |-- trainer.py
|   |-- training_setup.py
|   |-- training_step.py
|   `-- training_utils.py
|-- utils
|   |-- __init__.py
|   |-- config_loader.py
|   |-- device_utils.py
|   |-- distributed_utils.py
|   |-- error_handling.py
|   |-- experiment_utils.py
|   |-- helper_functions.py
|   |-- input_validation.py
|   |-- json_utils.py
|   |-- logging_utils.py
|   |-- metrics_utils.py
|   |-- seed_utils.py
|   |-- settings.py
|   `-- time_utils.py
|-- visualization
|   |-- __init__.py
|   `-- visualize.py
`-- __init__.py
```
