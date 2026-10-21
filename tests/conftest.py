"""Shared pytest fixtures; puts the project root on sys.path so ``import src`` works."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture(scope="session")
def toy_series():
    """6 days of a deterministic 4-sensor series on a 5-minute grid.

    ``speed[t, n] = 10 + t / 100 + n`` (strictly increasing in t) with a few holes, so any
    leakage of future values changes a feature in a detectable way.
    """
    ts = pd.date_range("2012-03-01", periods=6 * 288, freq="5min")
    t = np.arange(len(ts), dtype=np.float32)[:, None]
    n = np.arange(4, dtype=np.float32)[None, :]
    speed = (10 + t / 100 + n).astype(np.float32)
    mask = np.ones_like(speed, dtype=bool)
    mask[5:9, 1] = False
    mask[100, :] = False
    mask[700:760, 3] = False
    speed = np.where(mask, speed, np.nan).astype(np.float32)
    return ts, speed, mask
