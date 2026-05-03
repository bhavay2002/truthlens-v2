"""
TruthLens data layer.

Public surface:
    - run_data_pipeline, DataPipelineConfig
    - build_dataset, build_all_datasets
    - build_dataloader, build_all_dataloaders, DataLoaderConfig
    - collate_fn, build_collate_fn
    - get_contract, list_tasks, CONTRACTS
"""

from src.data_processing.data_contracts import (
    CONTRACTS,
    DataContract,
    DEFAULT_MAX_LENGTH,
    get_contract,
    list_tasks,
    is_classification,
    is_multilabel,
    get_num_classes,
    get_required_columns,
    get_optional_columns,
)

from src.data_processing.data_pipeline import (
    run_data_pipeline,
    DataPipelineConfig,
)

from src.data_processing.dataset_factory import (
    build_dataset,
    build_all_datasets,
    DatasetBuildConfig,
    validate_dataset_compatibility,
)

from src.data_processing.dataloader_factory import (
    build_dataloader,
    build_all_dataloaders,
    DataLoaderConfig,
)

from src.data_processing.collate import (
    collate_fn,
    build_collate_fn,
)

__all__ = [
    "CONTRACTS",
    "DataContract",
    "DEFAULT_MAX_LENGTH",
    "get_contract",
    "list_tasks",
    "is_classification",
    "is_multilabel",
    "get_num_classes",
    "get_required_columns",
    "get_optional_columns",
    "run_data_pipeline",
    "DataPipelineConfig",
    "build_dataset",
    "build_all_datasets",
    "DatasetBuildConfig",
    "validate_dataset_compatibility",
    "build_dataloader",
    "build_all_dataloaders",
    "DataLoaderConfig",
    "collate_fn",
    "build_collate_fn",
]
