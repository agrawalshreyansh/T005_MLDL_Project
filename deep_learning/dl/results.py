"""One place that names the models and loads their saved test predictions on a common mask.

Used by notebook 14 and the dashboard, so both always score the same points the same way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config, evaluation as ev
from src.preprocessing import ExperimentContext, gather_targets

# prediction file key -> (label, family); dict order = display order
MODELS = {
    "persistence": ("Persistence", "baseline"),
    "seasonal_avg": ("Seasonal average", "baseline"),
    "xgboost": ("XGBoost", "Phase 1 ML"),
    "xgboost_spatial": ("XGBoost + neighbours", "Phase 1 ML"),
    "dl_gru_s42": ("GRU (time only)", "deep learning"),
    "dl_st_gat_noedge_s42": ("ST-GAT, edges removed", "deep learning"),
    "dl_st_gat_s42": ("ST-GAT", "deep learning"),
    "dl_dcrnn_s42": ("DCRNN", "deep learning"),
}


def label(key: str) -> str:
    return MODELS.get(key, (key, ""))[0]


def available(split: str = "test") -> list[str]:
    return [k for k in MODELS if ev.prediction_path(k, split).is_file()]


@dataclass
class Scored:
    preds: dict[str, np.ndarray]   # key -> [S, H, N] mph, clipped to 0-70
    y: np.ndarray                  # [S, H, N] mph (NaN where missing)
    valid: np.ndarray              # [S, H, N] observed AND finite for every loaded model
    origins: np.ndarray            # [S]
    horizons: tuple[int, ...]

    def by_horizon(self) -> pd.DataFrame:
        t = ev.metrics_by_horizon(self.preds, self.y, self.valid, self.horizons)
        return t.assign(model=t.model.map(label))

    def by_peak(self, ctx: ExperimentContext) -> pd.DataFrame:
        peak = ev.peak_flags_sh(ctx.timestamps, self.origins, self.horizons)
        t = ev.metrics_by_group(self.preds, self.y, self.valid, self.horizons, {"peak": peak, "off-peak": ~peak})
        return t.assign(model=t.model.map(label))


def load_scored(ctx: ExperimentContext, keys: list[str] | None = None, split: str = "test") -> Scored:
    keys = keys or available(split)
    preds, y, ymask, origins, horizons, missing = ev.load_aligned(keys, split, ctx.speed, ctx.mask)
    if missing:
        raise FileNotFoundError(f"missing predictions: {missing}")
    return Scored(preds, y, ev.common_valid_mask(preds, ymask), origins, horizons)
