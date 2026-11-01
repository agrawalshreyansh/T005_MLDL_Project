"""Multi-output Random Forest forecaster (one forest predicts all horizons jointly).

Why a forest next to XGBoost
----------------------------
* Bagging instead of boosting: trees are grown independently on bootstrap samples and averaged, which lowers variance
  but cannot correct the bias of earlier trees (Breiman, 2001).  It is the bagging counterpart of the boosted model.
* Native multi-output regression: every split is chosen to reduce the *summed* squared error of all four horizon
  targets, so the horizons share one tree structure.  XGBoost fits one model per horizon.

Design choices (documented because they differ from the XGBoost setting)
-----------------------------------------------------------------------
* Targets are residuals from the last observed speed (the anchor), as for XGBoost and persistence.
* Only rows whose targets are observed at **all** horizons are used for fitting (a forest cannot learn from a partly
  missing target vector).  Scoring still uses every observed target.
* The split criterion is squared error (scikit-learn only supports missing feature values with that family of
  criteria).  The forest therefore optimises MSE while XGBoost optimises MAE; this is a real difference between the two.
* Missing feature values are routed natively by the trees (scikit-learn >= 1.4), not imputed.
* ``sensor_id`` is left out (a nominal id would be treated as an ordered number); the sensor is identified through its
  training mean speed and its coordinates.
* Calendar features describe the *origin* time because one tree structure serves all horizons.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from . import config


def fit_forest(x: np.ndarray, y_residual: np.ndarray, params: dict | None = None) -> RandomForestRegressor:
    """Fit the forest on ``x [rows, F]`` (NaN allowed) and residual targets ``y_residual [rows, H]`` (no NaN)."""
    if not np.isfinite(y_residual).all():
        raise ValueError("targets must be fully observed; filter the rows first")
    p = dict(config.RF_PARAMS if params is None else params)
    return RandomForestRegressor(**p).fit(x.astype(np.float32, copy=False), y_residual)


def predict_speed(model: RandomForestRegressor, x: np.ndarray, anchor: np.ndarray, chunk: int = 400_000) -> np.ndarray:
    """Predicted speeds in mph ``[rows, H]`` = anchor + predicted residual (predicted in chunks to bound memory)."""
    out = np.empty((len(x), model.n_outputs_), dtype=np.float32)
    for a in range(0, len(x), chunk):
        out[a:a + chunk] = model.predict(x[a:a + chunk].astype(np.float32, copy=False))
    return out + anchor[:, None].astype(np.float32)


def model_path(prefix: str = "rf") -> Path:
    return config.MODELS_DIR / f"{config.tagged(prefix)}.joblib"


def save_model(model: RandomForestRegressor, prefix: str = "rf") -> Path:
    """Save with joblib (a pickle format: only load files that this project produced)."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(prefix)
    joblib.dump(model, path, compress=3)
    return path


def load_model(prefix: str = "rf") -> RandomForestRegressor:
    path = model_path(prefix)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found - run notebook 05c_random_forest first.")
    return joblib.load(path)
