# =========================================================
# DEVICE UTILITIES
# =========================================================
from .device_utils import (
    device_summary,
    device_name,
    get_device,
    get_gpu_count,
    gpu_memory_summary,
    is_primary_process,
    move_batch,
    move_to_device,
    set_cuda_device,
)

# =========================================================
# FILESYSTEM / PATH UTILITIES
# =========================================================
from .helper_functions import (
    create_folder,
    ensure_directories,
    ensure_file_exists,
    get_file_size,
    to_path,
)

# =========================================================
# INPUT VALIDATION
# =========================================================
from .input_validation import (
    ensure_dataframe,
    ensure_non_empty_text,
    ensure_non_empty_text_column,
    ensure_non_empty_text_list,
    ensure_positive_int,
)

# =========================================================
# JSON UTILITIES
# =========================================================
from .json_utils import (
    append_json,
    append_json_batch,   # NEW
    load_json,
    save_json,
)

# =========================================================
# LOGGING UTILITIES
# =========================================================
from .logging_utils import (
    configure_logging,
    log_event,            # NEW
    log_training_step,    # NEW
    log_epoch_summary,    # NEW
)

# =========================================================
# SEED / REPRODUCIBILITY
# =========================================================
from .seed_utils import (
    create_generator,
    get_seed_state,
    seed_worker,
    set_seed,
)

# =========================================================
# TIME UTILITIES
# =========================================================
from .time_utils import (
    current_datetime,
    measure_runtime,
    timestamp,
    Timer,               # NEW
)

# =========================================================
# DISTRIBUTED (NEW - CRITICAL)
# =========================================================
from .distributed_utils import (
    get_rank,
    get_world_size,
    is_distributed,
    is_primary,
    barrier,
)

# =========================================================
# METRICS (NEW)
# =========================================================
from .metrics_utils import (
    safe_mean,
    normalize_score,
)

# =========================================================
# PUBLIC API
# =========================================================

__all__ = [
    # Device
    "device_summary",
    "device_name",
    "get_device",
    "get_gpu_count",
    "gpu_memory_summary",
    "is_primary_process",
    "move_batch",
    "move_to_device",
    "set_cuda_device",

    # Filesystem
    "create_folder",
    "ensure_directories",
    "ensure_file_exists",
    "get_file_size",
    "to_path",

    # Validation
    "ensure_dataframe",
    "ensure_non_empty_text",
    "ensure_non_empty_text_column",
    "ensure_non_empty_text_list",
    "ensure_positive_int",

    # JSON
    "append_json",
    "append_json_batch",
    "load_json",
    "save_json",

    # Logging
    "configure_logging",
    "log_event",
    "log_training_step",
    "log_epoch_summary",

    # Seed
    "create_generator",
    "get_seed_state",
    "seed_worker",
    "set_seed",

    # Time
    "current_datetime",
    "measure_runtime",
    "timestamp",
    "Timer",

    # Distributed
    "get_rank",
    "get_world_size",
    "is_distributed",
    "is_primary",
    "barrier",

    # Metrics
    "safe_mean",
    "normalize_score",

    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
]