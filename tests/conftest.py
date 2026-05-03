from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


# Ensure tests can import project modules (e.g. `src`, `api`) when run from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_dataset():
    return pd.DataFrame(
        {
            "text": ["hello world", "another sample"],
            "label": [0, 1],
        }
    )


@pytest.fixture
def feature_dataset():
    return pd.DataFrame(
        {
            "title": ["T1", "T2", "T3"],
            "text": [
                "economy inflation jobs report",
                "football match championship",
                "space telescope galaxy mission",
            ],
            "label": [0, 1, 0],
        }
    )
