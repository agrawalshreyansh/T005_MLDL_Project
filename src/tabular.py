"""Shared assembly of the tabular design matrices used by every tree-based model (XGBoost, Random Forest, tuning, ablation).

One row = one (forecast origin, sensor) pair, ordered origin-major (``row = s * N + n``).  The matrices are built once per
origin set; the horizon-specific calendar columns are attached on demand.  Targets are *residuals*: the speed at ``t + h``
minus the last observed speed in the input window (the anchor), exactly as in notebook 05, so that all tree models are
scored on what they add beyond persistence.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, feature_engineering as fe
from .preprocessing import ExperimentContext, calendar_arrays

SENSOR_IDENTITY = ("sensor_train_mean", "latitude", "longitude")
ORIGIN_CALENDAR = ("origin_tod_sin", "origin_tod_cos", "origin_dow_sin", "origin_dow_cos", "origin_is_weekend")


@dataclass
class TabularData:
    ctx: ExperimentContext
    origin_sets: dict[str, np.ndarray]
    base: dict[str, np.ndarray]
    base_names: list[str]
    lat: np.ndarray
    lon: np.ndarray
    neighbours: dict[str, np.ndarray] | None = None
    neighbour_names: list[str] = field(default_factory=list)

    @property
    def n_sensors(self) -> int:
        return len(self.ctx.sensor_ids)

    @property
    def anchor_col(self) -> int:
        return self.base_names.index("last_observed")

    def design(self, split: str, h: int, with_neighbours: bool = False) -> pd.DataFrame:
        """XGBoost design matrix for horizon ``h``: base features + target-time calendar (+ neighbours) + categorical sensor id."""
        hz, hz_names = fe.horizon_features(self.ctx.timestamps, self.origin_sets[split], h, self.n_sensors)
        base, names = self.base[split], self.base_names
        if with_neighbours:
            if self.neighbours is None:
                raise ValueError("neighbour features were not built")
            base, names = np.hstack([base, self.neighbours[split]]), names + self.neighbour_names
        return fe.to_frame(base, names, hz, hz_names, self.n_sensors)

    def origin_calendar(self, split: str) -> tuple[np.ndarray, list[str]]:
        """Calendar features of the *origin* time (horizon-free, needed by a model that predicts all horizons at once)."""
        cal = calendar_arrays(self.ctx.timestamps[self.origin_sets[split]])
        mat = np.stack([cal["tod_sin"], cal["tod_cos"], cal["dow_sin"], cal["dow_cos"], cal["is_weekend"]], axis=1)
        return np.repeat(mat, self.n_sensors, axis=0).astype(np.float32), list(ORIGIN_CALENDAR)

    def flat_matrix(self, split: str) -> tuple[np.ndarray, list[str]]:
        """Horizon-free float matrix (no sensor-id column): base features + origin calendar."""
        cal, cal_names = self.origin_calendar(split)
        return np.hstack([self.base[split], cal]), self.base_names + cal_names

    def targets(self, split: str, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(residual, observed, anchor)`` for every row of ``split`` at horizon ``h``."""
        o = self.origin_sets[split]
        y = self.ctx.speed[o + h].reshape(-1)
        ok = self.ctx.mask[o + h].reshape(-1)
        anchor = self.base[split][:, self.anchor_col]
        return y - anchor, ok, anchor

    def multi_targets(self, split: str, horizons: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Residual targets ``[rows, H]``, a per-row flag that *all* horizons are observed, and the anchor."""
        res, ok, anchor = [], [], None
        for h in horizons:
            r, k, anchor = self.targets(split, h)
            res.append(r); ok.append(k)
        return np.stack(res, axis=1), np.stack(ok, axis=1).all(axis=1), anchor


def build_tabular(
    ctx: ExperimentContext, lat: np.ndarray, lon: np.ndarray, adjacency: np.ndarray | None = None,
    train_stride: int = config.XGB_TRAIN_STRIDE, val_stride: int = config.XGB_VAL_STRIDE,
    splits: tuple[str, ...] = ("train", "val_fit", "val", "test"),
) -> TabularData:
    """Build the feature matrices for the requested origin sets (the same sets notebook 05 uses)."""
    all_sets = {
        "train": ctx.origins["train"][::train_stride],
        "val_fit": ctx.origins["val"][::val_stride],       # early stopping only
        "val": ctx.origins["val"],
        "test": ctx.origins["test"],
    }
    origin_sets = {k: all_sets[k] for k in splits}
    base, nbr, base_names, nbr_names = {}, {}, None, []
    for name, o in origin_sets.items():
        base[name], base_names = fe.build_base_features(ctx.speed, ctx.locf_value, ctx.locf_age, ctx.scaler.sensor_mean, o, lat, lon)
        if adjacency is not None:
            nbr[name], nbr_names = fe.build_neighbour_features(ctx.speed, adjacency, o)
    return TabularData(ctx, origin_sets, base, base_names, lat, lon, nbr or None, nbr_names)
