"""Leakage-free tabular features for the per-sensor (graph-agnostic) XGBoost model.

One row = one (forecast origin, sensor) pair; rows are ordered origin-major
(``row = s * N + n``).  Every feature uses only readings at or before the origin plus
calendar information about the forecast target time (which is known in advance).

Deliberately **excluded**: any feature built from *other* sensors (neighbour speeds,
network-wide means) and any statistic fitted on the rows being predicted (e.g. a
seasonal mean that contains the target itself).  This keeps the model an honest
"each sensor on its own" baseline for the research question.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .preprocessing import calendar_arrays, window_anchor, window_index_arrays

ROLL_MEAN_WINDOWS: tuple[int, ...] = (3, 6, 12)
ROLL_STD_WINDOWS: tuple[int, ...] = (6, 12)


def _nan_mean_std(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NaN-aware mean, std (needs >= 2 observations) and count over the last axis."""
    obs = np.isfinite(vals)
    cnt = obs.sum(axis=-1)
    v0 = np.where(obs, vals, 0.0)
    s1 = v0.sum(axis=-1, dtype=np.float64)
    s2 = (v0.astype(np.float64) ** 2).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt >= 2, s2 / np.maximum(cnt, 1) - mean ** 2, np.nan)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32), cnt


def build_base_features(
    speed: np.ndarray, locf_value: np.ndarray, locf_age: np.ndarray, sensor_mean: np.ndarray,
    origins: np.ndarray, lat: np.ndarray, lon: np.ndarray, input_len: int = config.INPUT_LEN,
) -> tuple[np.ndarray, list[str]]:
    """Horizon-independent features ``[S * N, F_base]`` and their names.

    ``speed`` is NaN where a reading is missing; lags therefore stay NaN (XGBoost handles
    missing values natively) instead of being zero-filled.
    """
    origins = np.asarray(origins, dtype=np.int64)
    n_samples, n = len(origins), speed.shape[1]
    x_idx, _ = window_index_arrays(origins, input_len, 1)
    window = speed[x_idx]                                   # [S, L, N] oldest -> newest
    lags = window[:, ::-1, :].transpose(0, 2, 1)            # [S, N, L], lag_0 = newest

    cols: list[np.ndarray] = []
    names: list[str] = []
    for k in range(input_len):
        cols.append(lags[:, :, k]); names.append(f"lag_{k}")
    for w in ROLL_MEAN_WINDOWS:
        mean, _, _ = _nan_mean_std(lags[:, :, :w])
        cols.append(mean); names.append(f"roll_mean_{w}")
    for w in ROLL_STD_WINDOWS:
        _, std, _ = _nan_mean_std(lags[:, :, :w])
        cols.append(std); names.append(f"roll_std_{w}")
    _, _, n_obs = _nan_mean_std(lags)
    cols.append(n_obs.astype(np.float32)); names.append("n_observed_window")
    recent, _, _ = _nan_mean_std(lags[:, :, :3])
    older, _, _ = _nan_mean_std(lags[:, :, 3:6])
    cols.append(recent - older); names.append("trend_3")

    anchor, _ = window_anchor(locf_value, locf_age, origins, input_len, sensor_mean)
    cols.append(anchor); names.append("last_observed")
    age = np.minimum(locf_age[origins], input_len).astype(np.float32)
    cols.append(age); names.append("age_last_observed")

    cols.append(np.broadcast_to(sensor_mean[None, :], (n_samples, n))); names.append("sensor_train_mean")
    cols.append(np.broadcast_to(lat[None, :].astype(np.float32), (n_samples, n))); names.append("latitude")
    cols.append(np.broadcast_to(lon[None, :].astype(np.float32), (n_samples, n))); names.append("longitude")

    stacked = np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=-1)   # [S, N, F]
    return stacked.reshape(n_samples * n, len(names)), names


def horizon_features(
    timestamps: pd.DatetimeIndex, origins: np.ndarray, horizon: int, n_sensors: int
) -> tuple[np.ndarray, list[str]]:
    """Calendar features of the forecast *target* time, repeated per sensor: ``[S * N, 5]``."""
    cal = calendar_arrays(timestamps[np.asarray(origins) + horizon])
    names = ["target_tod_sin", "target_tod_cos", "target_dow_sin", "target_dow_cos", "target_is_weekend"]
    mat = np.stack([cal["tod_sin"], cal["tod_cos"], cal["dow_sin"], cal["dow_cos"], cal["is_weekend"]], axis=1)
    return np.repeat(mat, n_sensors, axis=0).astype(np.float32), names


def sensor_index_column(n_samples: int, n_sensors: int) -> np.ndarray:
    """Sensor index for each row (origin-major order)."""
    return np.tile(np.arange(n_sensors, dtype=np.int32), n_samples)


def to_frame(
    base: np.ndarray, base_names: list[str], hz: np.ndarray, hz_names: list[str], n_sensors: int
) -> pd.DataFrame:
    """Assemble the XGBoost design matrix (``sensor_id`` as a categorical column)."""
    frame = pd.DataFrame(np.hstack([base, hz]), columns=base_names + hz_names)
    sid = sensor_index_column(len(base) // n_sensors, n_sensors)
    frame["sensor_id"] = pd.Categorical(sid, categories=np.arange(n_sensors))
    return frame
