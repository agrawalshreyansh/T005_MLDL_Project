"""Masked forecasting metrics.

Policy (identical for every model)
----------------------------------
* A point is **valid** when the true value is observed (``mask`` True) *and* the
  prediction is finite.  Invalid points never enter any metric and are never counted.
* **MAE / RMSE** use every valid point.
* **MAPE** additionally requires ``true >= mape_min_speed`` (default 1 mph) so that a
  near-zero denominator cannot dominate; the number of points used is reported separately.
* Aggregation is **pooled**: sums over all valid points, divided once at the end
  (RMSE is the square root of the pooled mean squared error, *not* an average of
  per-group RMSEs).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from . import config


def valid_points(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Boolean array of points usable for evaluation."""
    pred, true = np.asarray(pred), np.asarray(true)
    if pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} and true {true.shape} must have the same shape")
    ok = np.isfinite(pred) & np.isfinite(true)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != true.shape:
            raise ValueError(f"mask {mask.shape} must match true {true.shape}")
        ok &= mask
    return ok


def masked_mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, int]:
    """Mean absolute error and number of valid points (NaN when none)."""
    ok = valid_points(pred, true, mask)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.abs(pred[ok].astype(np.float64) - true[ok]).mean()), n


def masked_rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, int]:
    """Root mean squared error and number of valid points (NaN when none)."""
    ok = valid_points(pred, true, mask)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.sqrt(np.square(pred[ok].astype(np.float64) - true[ok]).mean())), n


def masked_mape(
    pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None,
    min_true: float = config.MAPE_MIN_SPEED,
) -> tuple[float, int]:
    """Mean absolute percentage error in percent and number of points used.

    Points with ``true < min_true`` are excluded (documented near-zero policy).
    """
    ok = valid_points(pred, true, mask) & (np.asarray(true) >= min_true)
    n = int(ok.sum())
    if n == 0:
        return float("nan"), 0
    t = true[ok].astype(np.float64)
    return float(100.0 * (np.abs(pred[ok].astype(np.float64) - t) / t).mean()), n


def compute_metrics(
    pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None,
    mape_min_speed: float = config.MAPE_MIN_SPEED,
) -> dict[str, float | int]:
    """MAE, RMSE, MAPE plus the valid-point counts used by each."""
    mae, n = masked_mae(pred, true, mask)
    rmse, _ = masked_rmse(pred, true, mask)
    mape, n_mape = masked_mape(pred, true, mask, mape_min_speed)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "n_valid": n, "n_valid_mape": n_mape}


def pooled_from_groups(groups: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Combine per-group MAE/RMSE into the pooled value (weights = ``n_valid``).

    Demonstrates (and tests) that pooled RMSE is ``sqrt(sum(n_g * rmse_g^2) / sum(n_g))``.
    """
    total = sum(g["n_valid"] for g in groups.values())
    if total == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "n_valid": 0}
    mae = sum(g["MAE"] * g["n_valid"] for g in groups.values()) / total
    mse = sum((g["RMSE"] ** 2) * g["n_valid"] for g in groups.values()) / total
    return {"MAE": float(mae), "RMSE": float(np.sqrt(mse)), "n_valid": int(total)}


def improvement_pct(reference_error: float, new_error: float) -> float:
    """Relative error reduction of ``new`` w.r.t. ``reference`` in percent.

    ``100 * (reference - new) / reference``.  **Positive = ``new`` has lower error (better).**
    """
    if not np.isfinite(reference_error) or reference_error == 0:
        return float("nan")
    return float(100.0 * (reference_error - new_error) / reference_error)
