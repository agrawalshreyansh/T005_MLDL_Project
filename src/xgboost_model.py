"""Direct multi-horizon XGBoost forecaster (one global model per horizon, shared by all sensors).

Strategy
--------
* **Global** model: all 207 sensors share the trees; ``sensor_id`` (categorical), latitude,
  longitude and the sensor's training mean let the trees specialise per sensor without
  training 207 x 4 separate models on ~100k rows each.
* **Direct** multi-horizon: a separate model for each horizon in ``config.HORIZONS``
  (no error accumulation as in recursive forecasting; costs 4 fits instead of 1).
* **Residual target**: the model predicts ``speed(t+h) - last_observed`` and the anchor is
  added back.  The same anchor is used by persistence and by the graph model, so the
  comparison is about what each model learns *beyond* persistence.
* Early stopping uses the **validation** split; the test split is untouched until final
  prediction.

Limitations: sensors are modelled independently (by design), and a global model with ~12
rolling features cannot represent congestion propagating along the road network.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config


def fit_horizon_model(
    x_train: pd.DataFrame, y_residual_train: np.ndarray,
    x_val: pd.DataFrame, y_residual_val: np.ndarray, params: dict | None = None,
) -> xgb.XGBRegressor:
    """Fit one horizon with validation-based early stopping."""
    p = dict(config.XGB_PARAMS if params is None else params)
    model = xgb.XGBRegressor(**p, eval_metric="mae")
    model.fit(x_train, y_residual_train, eval_set=[(x_val, y_residual_val)], verbose=False)
    return model


def predict_speed(model: xgb.XGBRegressor, x: pd.DataFrame) -> np.ndarray:
    """Predicted speed in mph = anchor + predicted residual."""
    return (x["last_observed"].to_numpy(dtype=np.float32) + model.predict(x)).astype(np.float32)


def model_path(horizon: int, prefix: str = "xgb") -> Path:
    return config.MODELS_DIR / f"{config.tagged(prefix)}_h{horizon}.json"


def save_models(models: dict[int, xgb.XGBRegressor], extra: dict, prefix: str = "xgb") -> None:
    """Save each horizon as native XGBoost JSON (no pickle) plus a hyper-parameter record."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    record = {"params": {k: str(v) for k, v in config.XGB_PARAMS.items()}, "horizons": {}}
    for h, m in models.items():
        m.save_model(model_path(h, prefix))
        record["horizons"][str(h)] = {"best_iteration": int(m.best_iteration), "n_features": int(m.n_features_in_)}
    record.update(extra)
    (config.MODELS_DIR / f"{config.tagged(prefix)}_hyperparameters.json").write_text(json.dumps(record, indent=2))


def load_model(horizon: int, prefix: str = "xgb") -> xgb.XGBRegressor:
    path = model_path(horizon, prefix)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run notebook 05_xgboost_forecasting first.")
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model
