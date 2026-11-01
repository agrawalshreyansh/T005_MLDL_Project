"""Hyper-parameter search utilities for the XGBoost forecaster (chronological validation only).

Protocol (notebook 05d)
-----------------------
1. **Random search** over a small, explicit grid.  Each candidate is fitted on a thinned training set and scored on the
   validation period (early stopping and ranking use validation only; the test period is never touched).
2. **Rolling-origin cross-validation** inside the *training* period for the best candidates and for the default
   configuration: expanding training window, the next block as validation, and a gap so that no training target lies
   after the first validation origin.  This guards against choosing a configuration that merely suits the validation weeks.
"""
from __future__ import annotations

import time

import numpy as np

from . import config
from .xgboost_model import fit_horizon_model

SEARCH_SPACE: dict[str, list] = dict(
    max_depth=[4, 6, 8, 10],
    min_child_weight=[5, 20, 50, 100],
    learning_rate=[0.05, 0.08, 0.12],
    subsample=[0.6, 0.8, 1.0],
    colsample_bytree=[0.5, 0.8, 1.0],
    reg_lambda=[1.0, 5.0, 20.0],
)


def default_candidate() -> dict:
    """The configuration of notebook 05, restricted to the searched parameters."""
    return {k: config.XGB_PARAMS[k] for k in SEARCH_SPACE}


def sample_configs(n: int, seed: int = config.SEED) -> list[dict]:
    """``n`` distinct random configurations from :data:`SEARCH_SPACE`, never equal to the default."""
    rng = np.random.default_rng(seed)
    keys, seen, out = list(SEARCH_SPACE), {tuple(default_candidate().values())}, []
    total = int(np.prod([len(v) for v in SEARCH_SPACE.values()]))
    if n > total - 1:
        raise ValueError(f"only {total - 1} distinct non-default configurations exist")
    while len(out) < n:
        cfg = {k: SEARCH_SPACE[k][int(rng.integers(len(SEARCH_SPACE[k])))] for k in keys}
        sig = tuple(cfg.values())
        if sig not in seen:
            seen.add(sig); out.append(cfg)
    return out


def rows_for_origins(origin_idx: np.ndarray, n_sensors: int) -> np.ndarray:
    """Row indices of the (origin, sensor) design matrix that belong to the origins at positions ``origin_idx``."""
    idx = np.asarray(origin_idx, dtype=np.int64)
    return (idx[:, None] * n_sensors + np.arange(n_sensors)[None, :]).reshape(-1)


def rolling_origin_folds(
    origins: np.ndarray, n_folds: int, max_horizon: int = config.MAX_HORIZON,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds over sorted ``origins``: ``[(train positions, validation positions), ...]``.

    The origins are cut into ``n_folds + 1`` consecutive blocks.  Fold ``i`` trains on blocks ``0..i-1`` and validates on
    block ``i``.  Training origins whose furthest target ``o + max_horizon`` reaches the first validation origin are
    dropped (the gap), so no training target overlaps the validation block.
    """
    origins = np.asarray(origins)
    if n_folds < 1 or len(origins) < 2 * (n_folds + 1):
        raise ValueError("not enough origins for the requested number of folds")
    if not (np.diff(origins) > 0).all():
        raise ValueError("origins must be strictly increasing")
    edges = np.linspace(0, len(origins), n_folds + 2).astype(int)
    folds = []
    for i in range(1, n_folds + 1):
        val_pos = np.arange(edges[i], edges[i + 1])
        train_pos = np.arange(0, edges[i])
        train_pos = train_pos[origins[train_pos] + max_horizon < origins[val_pos[0]]]
        folds.append((train_pos, val_pos))
    return folds


def tuning_params(overrides: dict) -> dict:
    """Full XGBoost parameter dict for one search candidate (reduced round budget for the search)."""
    p = dict(config.XGB_PARAMS)
    p.update(overrides)
    p.update(n_estimators=config.TUNE_MAX_ROUNDS, early_stopping_rounds=config.TUNE_EARLY_STOPPING)
    return p


def fit_and_score(x_train, y_train, x_val, y_val, overrides: dict) -> dict:
    """Fit one candidate with early stopping on ``(x_val, y_val)``; the score is the validation MAE of the residual (mph)."""
    t0 = time.time()
    model = fit_horizon_model(x_train, y_train, x_val, y_val, tuning_params(overrides))
    curve = model.evals_result()["validation_0"]["mae"]
    return {"val_mae": float(min(curve)), "best_iteration": int(model.best_iteration) + 1, "seconds": round(time.time() - t0, 1)}
