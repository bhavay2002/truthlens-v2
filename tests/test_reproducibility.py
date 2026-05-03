from __future__ import annotations

import numpy as np
import torch

from src.utils.seed_utils import set_seed


def test_numpy_reproducibility_with_set_seed() -> None:
    set_seed(42)
    a = np.random.rand(5)

    set_seed(42)
    b = np.random.rand(5)

    np.testing.assert_array_equal(a, b)


def test_torch_reproducibility_with_set_seed() -> None:
    set_seed(7)
    a = torch.rand(5)

    set_seed(7)
    b = torch.rand(5)

    assert torch.allclose(a, b)


def test_different_seeds_produce_different_outputs() -> None:
    set_seed(1)
    a = np.random.rand(5)

    set_seed(2)
    b = np.random.rand(5)

    assert not np.array_equal(a, b)


def test_set_seed_sets_python_hash_seed() -> None:
    import os
    set_seed(123)
    assert os.environ.get("PYTHONHASHSEED") == "123"


def test_deterministic_mode_does_not_raise() -> None:
    set_seed(42, deterministic=True)


def test_non_deterministic_mode_does_not_raise() -> None:
    set_seed(42, deterministic=False)
