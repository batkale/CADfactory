"""
Centralized seed management for reproducibility.

Sets random seeds across all RNG backends (stdlib, numpy, and optionally torch)
to ensure reproducible results in topology optimization, stochastic sampling,
and any future ML features.

Inspired by GenCAD's lack of reproducibility controls.
"""

import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all backends.

    Sets:
      - Python stdlib random
      - numpy
      - PYTHONHASHSEED env var
      - torch (if installed)
      - torch.cuda (if available)

    Args:
        seed: Integer seed value. Default 42.
    """
    # Python stdlib
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Hash seed for deterministic string hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (optional — may not be installed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Deterministic algorithms (may be slower)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.debug(f"Seed set for torch: {seed}")
    except ImportError:
        pass

    logger.debug(f"Global seed set: {seed}")


def get_seed_from_env(default: int = 42) -> int:
    """Read seed from CADFACTORY_SEED env var, or use default."""
    return int(os.environ.get("CADFACTORY_SEED", str(default)))


def reproducible_context(seed: Optional[int] = None):
    """
    Context manager for reproducible operations.
    Saves and restores RNG states after the block.

    Usage:
        with reproducible_context(seed=123):
            result = topology_optimize(...)
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        if seed is None:
            yield
            return

        # Save state
        py_state = random.getstate()
        np_state = np.random.get_state()

        set_seed(seed)
        try:
            yield
        finally:
            # Restore state
            random.setstate(py_state)
            np.random.set_state(np_state)

    return _ctx()
