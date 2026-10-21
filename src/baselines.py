"""Non-neural forecasting baselines.

All baselines return arrays of shape ``[S, H, N]`` (samples, horizons, sensors) in mph
and **never use information after the forecast origin**, except the seasonal table,
which is built from *training timestamps only*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .preprocessing import calendar_arrays, window_anchor, window_index_arrays


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def persistence_forecast(anchor: np.ndarray, n_horizons: int) -> np.ndarray:
    """Predict every horizon with the latest observed speed in the input window.

    ``anchor`` is ``[S, N]`` as produced by :func:`preprocessing.window_anchor` (training
    sensor mean when the whole window is missing).
    """
    return np.repeat(anchor[:, None, :], n_horizons, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Rolling mean
# --------------------------------------------------------------------------- #
def rolling_mean_forecast(
    speed: np.ndarray, mask: np.ndarray, origins: np.ndarray, window: int,
    fallback: np.ndarray, n_horizons: int,
) -> np.ndarray:
    """Predict every horizon with the mean of the observed speeds in the last ``window`` steps."""
    x_idx, _ = window_index_arrays(origins, window, 1)
    vals = np.where(mask[x_idx], speed[x_idx], 0.0)          # [S, window, N]
    cnt = mask[x_idx].sum(axis=1)                            # [S, N]
    mean = np.where(cnt > 0, vals.sum(axis=1) / np.maximum(cnt, 1), fallback[None, :])
    return np.repeat(mean[:, None, :], n_horizons, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Seasonal (historical) average
# --------------------------------------------------------------------------- #
@dataclass
class SeasonalAverage:
    """Expected speed for a (day-of-week, 5-minute slot, sensor) cell from training history.

    Fallback hierarchy for cells with fewer than ``min_count`` training observations:
    1. (weekday-or-weekend, slot, sensor) average, 2. sensor mean over the training range.
    """

    table_dow: np.ndarray        # [7, 288, N]  NaN where fewer than min_count observations
    table_daytype: np.ndarray    # [2, 288, N]  weekday / weekend
    sensor_mean: np.ndarray      # [N]
    count_dow: np.ndarray        # [7, 288, N]  observations behind each cell
    min_count: int

    @classmethod
    def fit(
        cls, speed: np.ndarray, mask: np.ndarray, timestamps: pd.DatetimeIndex,
        train_range: tuple[int, int], min_count: int = 3,
    ) -> "SeasonalAverage":
        a, b = train_range
        cal = calendar_arrays(timestamps[a:b])
        dow, slot = cal["dow"], cal["slot"]
        daytype = (dow >= 5).astype(np.int32)
        obs = mask[a:b]
        vals = np.where(obs, speed[a:b], 0.0).astype(np.float64)
        n = speed.shape[1]
        n_slots = 24 * 60 // config.STEP_MINUTES

        sum_dow = np.zeros((7, n_slots, n))
        cnt_dow = np.zeros((7, n_slots, n))
        np.add.at(sum_dow, (dow, slot), vals)
        np.add.at(cnt_dow, (dow, slot), obs.astype(np.float64))
        sum_dt = np.zeros((2, n_slots, n))
        cnt_dt = np.zeros((2, n_slots, n))
        np.add.at(sum_dt, (daytype, slot), vals)
        np.add.at(cnt_dt, (daytype, slot), obs.astype(np.float64))

        with np.errstate(invalid="ignore", divide="ignore"):
            table_dow = np.where(cnt_dow >= min_count, sum_dow / cnt_dow, np.nan)
            table_dt = np.where(cnt_dt >= 1, sum_dt / cnt_dt, np.nan)
        total_cnt = obs.sum(axis=0)
        sensor_mean = np.where(total_cnt > 0, vals.sum(axis=0) / np.maximum(total_cnt, 1), np.nan)
        # a sensor with no training data at all gets the network mean (never zero)
        sensor_mean = np.where(np.isfinite(sensor_mean), sensor_mean, np.nanmean(sensor_mean))
        return cls(
            table_dow.astype(np.float32), table_dt.astype(np.float32),
            sensor_mean.astype(np.float32), cnt_dow.astype(np.int32), min_count,
        )

    def predict_at(self, timestamps: pd.DatetimeIndex, return_level: bool = False):
        """Seasonal expectation for each timestamp: ``[len(timestamps), N]``."""
        cal = calendar_arrays(timestamps)
        dow, slot = cal["dow"], cal["slot"]
        daytype = (dow >= 5).astype(np.int32)
        level1 = self.table_dow[dow, slot]
        level2 = self.table_daytype[daytype, slot]
        pred = np.where(np.isfinite(level1), level1, level2)
        level = np.where(np.isfinite(level1), 1, np.where(np.isfinite(level2), 2, 3))
        pred = np.where(np.isfinite(pred), pred, self.sensor_mean[None, :]).astype(np.float32)
        return (pred, level) if return_level else pred


def seasonal_forecast(
    seasonal: SeasonalAverage, timestamps: pd.DatetimeIndex, origins: np.ndarray, horizons: tuple[int, ...]
) -> np.ndarray:
    """Historical average at each target time ``origin + h``: ``[S, H, N]``."""
    out = []
    for h in horizons:
        tgt = np.asarray(origins) + h
        out.append(seasonal.predict_at(timestamps[tgt]))
    return np.stack(out, axis=1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Convenience wrapper
# --------------------------------------------------------------------------- #
def all_baseline_forecasts(
    speed: np.ndarray, mask: np.ndarray, timestamps: pd.DatetimeIndex,
    locf_value: np.ndarray, locf_age: np.ndarray, sensor_mean: np.ndarray,
    seasonal: SeasonalAverage, origins: np.ndarray,
    horizons: tuple[int, ...] = config.HORIZONS, input_len: int = config.INPUT_LEN,
) -> dict[str, np.ndarray]:
    """Persistence, rolling-mean and seasonal forecasts for the given origins."""
    anchor, _ = window_anchor(locf_value, locf_age, origins, input_len, sensor_mean)
    h = len(horizons)
    return {
        "persistence": persistence_forecast(anchor, h),
        "rolling_mean": rolling_mean_forecast(speed, mask, origins, input_len, sensor_mean, h),
        "seasonal_avg": seasonal_forecast(seasonal, timestamps, origins, horizons),
    }
