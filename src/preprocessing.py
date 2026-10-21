"""Chronological splitting, training-only scaling, causal helpers and window generation.

Conventions used everywhere
---------------------------
* ``speed``  : float32 ``[T, N]`` in mph, ``NaN`` where the reading is missing.
* ``mask``   : bool ``[T, N]``, ``True`` where the reading is observed.
* An **origin** ``t`` is the index of the *last* input step.  A sample uses the
  inputs ``t-L+1 .. t`` and predicts the targets ``t+1 .. t+H`` (step ``h`` is the
  value ``h`` steps = ``5h`` minutes after the origin).
* A sample belongs to the split in which **all of its targets** fall.  Its inputs
  may reach back into earlier splits: that is legitimate past context that would be
  available in real operation.  Nothing after the origin is ever used as an input.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loader import CleanTraffic, load_clean_traffic

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Chronological splits
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitBoundaries:
    """Half-open index ranges ``[start, end)`` of the three chronological splits."""

    n_total: int
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    first_timestamp: dict[str, str]
    last_timestamp: dict[str, str]

    def range(self, name: str) -> tuple[int, int]:
        return getattr(self, name)

    def to_json(self, path: Path = config.SPLITS_JSON) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "n_total": self.n_total, "train": self.train, "val": self.val, "test": self.test,
            "first_timestamp": self.first_timestamp, "last_timestamp": self.last_timestamp,
        }, indent=2))

    @classmethod
    def from_json(cls, path: Path = config.SPLITS_JSON) -> "SplitBoundaries":
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path} not found - run notebook 02_data_preprocessing first.")
        d = json.loads(Path(path).read_text())
        return cls(d["n_total"], tuple(d["train"]), tuple(d["val"]), tuple(d["test"]),
                   d["first_timestamp"], d["last_timestamp"])


def make_chronological_splits(
    timestamps: pd.DatetimeIndex,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    align_to_day: bool = config.SPLIT_ALIGN_TO_DAY,
) -> SplitBoundaries:
    """Contiguous train / validation / test ranges with exact timestamp boundaries.

    With ``align_to_day`` each boundary snaps to the nearest midnight, so every split
    contains whole days (no day is shared between splits and weekly/day-time structure
    is not truncated mid-day).  The test fraction is the remainder.
    """
    n = len(timestamps)
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Need 0 < train_frac, val_frac and train_frac + val_frac < 1")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("Timestamps must be strictly increasing before splitting.")
    raw_train_end = int(round(train_frac * n))
    raw_val_end = int(round((train_frac + val_frac) * n))

    def snap(idx: int) -> int:
        if not align_to_day:
            return idx
        midnights = np.flatnonzero((timestamps.hour == 0) & (timestamps.minute == 0))
        if len(midnights) == 0:
            return idx
        return int(midnights[np.argmin(np.abs(midnights - idx))])

    train_end, val_end = snap(raw_train_end), snap(raw_val_end)
    if not (0 < train_end < val_end < n):
        raise ValueError(f"Degenerate split boundaries {train_end}, {val_end} for n={n}")
    ranges = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, n)}
    first = {k: str(timestamps[a]) for k, (a, b) in ranges.items()}
    last = {k: str(timestamps[b - 1]) for k, (a, b) in ranges.items()}
    return SplitBoundaries(n, ranges["train"], ranges["val"], ranges["test"], first, last)


# --------------------------------------------------------------------------- #
# Training-only scaling
# --------------------------------------------------------------------------- #
@dataclass
class SpeedScaler:
    """Global z-score scaler plus per-sensor training means, fitted on training rows only."""

    mean: float
    std: float
    sensor_mean: np.ndarray        # [N] mean observed training speed per sensor (mph)
    fit_range: tuple[int, int]     # index range the statistics were computed on

    @classmethod
    def fit(cls, speed: np.ndarray, mask: np.ndarray, fit_range: tuple[int, int]) -> "SpeedScaler":
        a, b = fit_range
        block, obs = speed[a:b], mask[a:b]
        if not obs.any():
            raise ValueError("No observed training values to fit the scaler on.")
        values = block[obs].astype(np.float64)
        counts = obs.sum(axis=0)
        sums = np.where(obs, block, 0.0).sum(axis=0, dtype=np.float64)
        global_mean = float(values.mean())
        # sensors with no training observation fall back to the global mean
        sensor_mean = np.where(counts > 0, sums / np.maximum(counts, 1), global_mean).astype(np.float32)
        return cls(global_mean, float(values.std()), sensor_mean, (int(a), int(b)))

    def transform(self, speed: np.ndarray) -> np.ndarray:
        return ((speed - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return (z * self.std + self.mean).astype(np.float32)

    def to_json(self, path: Path = config.SCALER_JSON) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "mean": self.mean, "std": self.std, "fit_range": list(self.fit_range),
            "sensor_mean": self.sensor_mean.tolist(),
        }, indent=2))

    @classmethod
    def from_json(cls, path: Path = config.SCALER_JSON) -> "SpeedScaler":
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path} not found - run notebook 02_data_preprocessing first.")
        d = json.loads(Path(path).read_text())
        return cls(d["mean"], d["std"], np.asarray(d["sensor_mean"], dtype=np.float32), tuple(d["fit_range"]))


# --------------------------------------------------------------------------- #
# Calendar features (known in advance, therefore legal at prediction time)
# --------------------------------------------------------------------------- #
def calendar_arrays(timestamps: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Minute-of-day, day-of-week (Mon=0) and cyclical encodings for every timestamp."""
    minute = (timestamps.hour * 60 + timestamps.minute).to_numpy().astype(np.float32)
    dow = timestamps.dayofweek.to_numpy().astype(np.float32)
    return {
        "minute_of_day": minute,
        "slot": (minute // config.STEP_MINUTES).astype(np.int32),
        "dow": dow.astype(np.int32),
        "tod_sin": np.sin(2 * np.pi * minute / 1440.0).astype(np.float32),
        "tod_cos": np.cos(2 * np.pi * minute / 1440.0).astype(np.float32),
        "dow_sin": np.sin(2 * np.pi * dow / 7.0).astype(np.float32),
        "dow_cos": np.cos(2 * np.pi * dow / 7.0).astype(np.float32),
        "is_weekend": (dow >= 5).astype(np.float32),
    }


def time_feature_matrix(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """``[T, 4]`` matrix ``(tod_sin, tod_cos, dow_sin, dow_cos)`` used as dynamic model input."""
    cal = calendar_arrays(timestamps)
    return np.stack([cal["tod_sin"], cal["tod_cos"], cal["dow_sin"], cal["dow_cos"]], axis=1)


# --------------------------------------------------------------------------- #
# Causal (past-only) helpers
# --------------------------------------------------------------------------- #
def locf_with_age(speed: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last observation carried forward, using **only past and current** rows.

    Returns ``(value [T, N], age [T, N])`` where ``age[t, n]`` is the number of steps
    since sensor ``n`` was last observed at or before ``t`` (0 when ``t`` itself is
    observed).  If the sensor has not been observed yet, ``value`` is NaN and ``age``
    is ``t + 1`` (larger than any window).
    """
    t_len, n = mask.shape
    rows = np.arange(t_len)[:, None]
    last_idx = np.maximum.accumulate(np.where(mask, rows, -1), axis=0)
    seen = last_idx >= 0
    safe = np.where(seen, last_idx, 0)
    value = np.where(seen, np.take_along_axis(speed, safe, axis=0), np.nan).astype(np.float32)
    age = np.where(seen, rows - last_idx, rows + 1).astype(np.int32)
    return value, age


def window_anchor(
    locf_value: np.ndarray, locf_age: np.ndarray, origins: np.ndarray, input_len: int,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Latest observed speed inside the input window of each origin.

    ``anchor[s, n]`` is the most recent observation among the ``input_len`` input steps
    of sample ``s`` at sensor ``n``.  If the whole window is missing the per-sensor
    ``fallback`` (training mean) is used and ``used_fallback`` is True.
    """
    value, age = locf_value[origins], locf_age[origins]
    in_window = (age <= input_len - 1) & np.isfinite(value)
    anchor = np.where(in_window, value, fallback[None, :]).astype(np.float32)
    return anchor, ~in_window


# --------------------------------------------------------------------------- #
# Window generation
# --------------------------------------------------------------------------- #
def valid_origins(
    split_range: tuple[int, int], input_len: int, max_horizon: int, n_total: int, split_name: str
) -> np.ndarray:
    """All origins whose targets ``t+1..t+max_horizon`` lie fully inside ``split_range``.

    The inputs ``t-input_len+1..t`` need only exist (index >= 0); for validation and test
    they may extend into the preceding split (past context).
    """
    start, end = split_range
    first = max(start - 1, input_len - 1)          # target t+1 must be >= start
    last = end - 1 - max_horizon                   # target t+max_horizon must be < end
    if last < first:
        return np.empty(0, dtype=np.int64)
    origins = np.arange(first, last + 1, dtype=np.int64)
    assert origins.max() + max_horizon < min(end, n_total), f"{split_name}: target leaves split"
    return origins


def window_index_arrays(origins: np.ndarray, input_len: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Index matrices ``x_idx [S, L]`` (inputs) and ``y_idx [S, n_steps]`` (targets)."""
    origins = np.asarray(origins, dtype=np.int64)
    x_idx = origins[:, None] + np.arange(-input_len + 1, 1)[None, :]
    y_idx = origins[:, None] + np.arange(1, n_steps + 1)[None, :]
    return x_idx, y_idx


def assert_no_leakage(
    origins: np.ndarray, split_range: tuple[int, int], input_len: int, n_steps: int, n_total: int
) -> None:
    """Hard checks on temporal ordering for a set of origins."""
    x_idx, y_idx = window_index_arrays(origins, input_len, n_steps)
    assert x_idx.min() >= 0, "inputs reach before the start of the data"
    assert (x_idx.max(axis=1) <= origins).all(), "an input step lies after its origin"
    assert (y_idx.min(axis=1) > origins).all(), "a target is not strictly in the future"
    assert y_idx.min() >= split_range[0] and y_idx.max() < split_range[1], "targets leave the split"
    assert y_idx.max() < n_total


def gather_model_inputs(
    speed_scaled_filled: np.ndarray, mask: np.ndarray, time_feats: np.ndarray,
    origins: np.ndarray, input_len: int,
) -> np.ndarray:
    """Graph-model input tensor ``[S, L, N, F]`` with ``F = 2 + time features``.

    Channels: (0) scaled speed (0 where missing, i.e. the training mean), (1) observed mask,
    (2..) calendar features of the input step broadcast over sensors.
    """
    x_idx, _ = window_index_arrays(origins, input_len, 1)
    values = speed_scaled_filled[x_idx]                      # [S, L, N]
    obs = mask[x_idx].astype(np.float32)                     # [S, L, N]
    n = values.shape[2]
    tf = time_feats[x_idx]                                   # [S, L, 4]
    tf = np.broadcast_to(tf[:, :, None, :], (*tf.shape[:2], n, tf.shape[2]))
    return np.concatenate([values[..., None], obs[..., None], tf], axis=-1).astype(np.float32)


def gather_targets(
    speed: np.ndarray, mask: np.ndarray, origins: np.ndarray, steps: tuple[int, ...] | list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Targets at the given 1-based ``steps``: ``(y [S, H, N] with NaN, observed [S, H, N])``."""
    origins = np.asarray(origins, dtype=np.int64)
    idx = origins[:, None] + np.asarray(steps, dtype=np.int64)[None, :]
    return speed[idx], mask[idx]


def fill_for_model(speed: np.ndarray, mask: np.ndarray, scaler: SpeedScaler) -> np.ndarray:
    """Scaled speeds with missing entries set to 0 (= training mean in scaled space).

    This uses no neighbouring or future values; the observed-mask channel tells the model
    which entries are real.
    """
    return np.where(mask, scaler.transform(np.nan_to_num(speed, nan=0.0)), 0.0).astype(np.float32)


def missingness_summary(mask: np.ndarray, splits: SplitBoundaries) -> pd.DataFrame:
    """Missing-value statistics per split."""
    rows = []
    for name in SPLIT_NAMES:
        a, b = splits.range(name)
        m = mask[a:b]
        rows.append({
            "split": name, "timesteps": b - a, "cells": m.size,
            "missing_cells": int((~m).sum()), "missing_pct": 100 * float((~m).mean()),
            "network_wide_outage_steps": int((~m).all(axis=1).sum()),
        })
    return pd.DataFrame(rows).set_index("split")


# --------------------------------------------------------------------------- #
# Shared experiment context (what every modelling notebook loads from disk)
# --------------------------------------------------------------------------- #
def evenly_spaced(indices: np.ndarray, cap: int | None) -> np.ndarray:
    """At most ``cap`` indices spread evenly over ``indices`` (all of them if ``cap`` is None)."""
    if cap is None or len(indices) <= cap:
        return indices
    return indices[np.linspace(0, len(indices) - 1, cap).round().astype(int)]


@dataclass
class ExperimentContext:
    """Everything notebooks 04-08 need, rebuilt from files written by notebooks 02 and 03."""

    clean: CleanTraffic
    splits: SplitBoundaries
    scaler: SpeedScaler
    locf_value: np.ndarray
    locf_age: np.ndarray
    origins: dict[str, np.ndarray]     # split -> forecast origins (identical for every model)

    @property
    def speed(self) -> np.ndarray:
        return self.clean.speed

    @property
    def mask(self) -> np.ndarray:
        return self.clean.mask

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self.clean.timestamps

    @property
    def sensor_ids(self) -> list[str]:
        return self.clean.sensor_ids


def load_context(
    input_len: int = config.INPUT_LEN, max_horizon: int = config.MAX_HORIZON,
) -> ExperimentContext:
    """Load the processed data and recompute the (deterministic) causal helper arrays."""
    clean = load_clean_traffic()
    splits = SplitBoundaries.from_json()
    scaler = SpeedScaler.from_json()
    if splits.n_total != len(clean.timestamps):
        raise ValueError("splits.json does not match traffic_clean.npz - rerun notebook 02.")
    if scaler.fit_range != splits.train:
        raise ValueError("The scaler was not fitted on the training range - rerun notebook 02.")
    locf_value, locf_age = locf_with_age(clean.speed, clean.mask)
    origins = {}
    for name in SPLIT_NAMES:
        full = valid_origins(splits.range(name), input_len, max_horizon, splits.n_total, name)
        assert_no_leakage(full, splits.range(name), input_len, max_horizon, splits.n_total)
        origins[name] = evenly_spaced(full, config.SMOKE_ORIGIN_CAP if config.IS_SMOKE else None)
    return ExperimentContext(clean, splits, scaler, locf_value, locf_age, origins)
