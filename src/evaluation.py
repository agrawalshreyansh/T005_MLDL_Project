"""Model-agnostic evaluation on saved predictions.

Every model saves ``[S, H, N]`` predictions (mph) for the same ordered origins; this module
aligns them with the observed targets and applies **identical** rules to all of them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .metrics import compute_metrics, improvement_pct
from .preprocessing import gather_targets


# --------------------------------------------------------------------------- #
# Prediction artifacts
# --------------------------------------------------------------------------- #
def prediction_path(model: str, split: str) -> Path:
    return config.PREDICTIONS_DIR / f"{config.tagged(model)}__{split}.npz"


def save_predictions(
    model: str, split: str, pred: np.ndarray, origins: np.ndarray,
    horizons: tuple[int, ...], meta: dict | None = None,
) -> Path:
    """Persist ``pred [S, H, N]`` with the origins/horizons that define its axes."""
    if pred.ndim != 3 or pred.shape[0] != len(origins) or pred.shape[1] != len(horizons):
        raise ValueError(f"pred {pred.shape} inconsistent with {len(origins)} origins x {len(horizons)} horizons")
    path = prediction_path(model, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, pred=pred.astype(np.float32), origins=np.asarray(origins, dtype=np.int64),
        horizons=np.asarray(horizons, dtype=np.int64),
        meta=json.dumps(meta or {}, default=str),
    )
    return path


def clip_predictions(pred: np.ndarray, bounds: tuple[float, float] = config.SPEED_BOUNDS_MPH) -> np.ndarray:
    """Clip predictions to the physically possible speed range (NaN stays NaN)."""
    return np.clip(pred, bounds[0], bounds[1]).astype(np.float32)


def load_predictions(model: str, split: str, clip: bool = True) -> dict:
    """Load saved predictions; by default they are clipped to ``config.SPEED_BOUNDS_MPH``.

    The files on disk hold the raw model output; clipping is applied identically to every model
    at load time so that no model is scored on impossible speeds (negative or above the 70 mph cap).
    """
    path = prediction_path(model, split)
    if not path.is_file():
        raise FileNotFoundError(f"No predictions for {model!r}/{split} at {path}")
    with np.load(path, allow_pickle=False) as z:
        pred = z["pred"]
        return {
            "pred": clip_predictions(pred) if clip else pred, "origins": z["origins"],
            "horizons": tuple(int(h) for h in z["horizons"]), "meta": json.loads(str(z["meta"])),
        }


# --------------------------------------------------------------------------- #
# Peak / off-peak
# --------------------------------------------------------------------------- #
def is_peak(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """True for weekday hours inside ``config.PEAK_WINDOWS`` (local time of the target)."""
    hour = np.asarray(timestamps.hour)
    flag = np.zeros(len(timestamps), dtype=bool)
    for start, end in config.PEAK_WINDOWS:
        flag |= (hour >= start) & (hour < end)
    if config.PEAK_WEEKDAYS_ONLY:
        flag &= np.asarray(timestamps.dayofweek) < 5
    return flag


def target_timestamps(timestamps: pd.DatetimeIndex, origins: np.ndarray, horizons: tuple[int, ...]) -> np.ndarray:
    """``[S, H]`` array of the timestamps being predicted (origin + h steps)."""
    idx = np.asarray(origins)[:, None] + np.asarray(horizons)[None, :]
    return timestamps.values[idx]


def peak_flags_sh(timestamps: pd.DatetimeIndex, origins: np.ndarray, horizons: tuple[int, ...]) -> np.ndarray:
    """``[S, H]`` boolean peak flag of the forecast target time."""
    idx = np.asarray(origins)[:, None] + np.asarray(horizons)[None, :]
    return is_peak(timestamps[idx.ravel()]).reshape(idx.shape)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def common_valid_mask(
    preds: dict[str, np.ndarray], ymask: np.ndarray
) -> np.ndarray:
    """Targets observed AND finite in every model's prediction."""
    ok = ymask.copy()
    for name, p in preds.items():
        if p.shape != ymask.shape:
            raise ValueError(f"{name}: prediction {p.shape} != target {ymask.shape}")
        ok &= np.isfinite(p)
    return ok


def load_aligned(
    models: list[str], split: str, speed: np.ndarray, mask: np.ndarray, skip_missing: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], list[str]]:
    """Load several models' predictions and the matching targets.

    Returns ``(preds, y, observed_mask, origins, horizons, missing_models)``.  All models
    must share identical origins and horizons, otherwise an error is raised.  Models whose
    prediction file is absent are reported in ``missing_models`` (never invented).
    """
    preds: dict[str, np.ndarray] = {}
    origins = horizons = None
    missing: list[str] = []
    for m in models:
        try:
            d = load_predictions(m, split)
        except FileNotFoundError:
            if not skip_missing:
                raise
            missing.append(m)
            continue
        if origins is None:
            origins, horizons = d["origins"], d["horizons"]
        elif not (np.array_equal(origins, d["origins"]) and horizons == d["horizons"]):
            raise ValueError(f"{m}: origins/horizons differ from the other models")
        preds[m] = d["pred"]
    if origins is None:
        raise FileNotFoundError(f"No prediction files found for {models} / {split}")
    y, ymask = gather_targets(speed, mask, origins, horizons)
    return preds, y, ymask, origins, horizons, missing


# --------------------------------------------------------------------------- #
# Metric tables
# --------------------------------------------------------------------------- #
def metrics_by_horizon(
    preds: dict[str, np.ndarray], y: np.ndarray, valid: np.ndarray, horizons: tuple[int, ...],
) -> pd.DataFrame:
    """Long table ``model, horizon_steps, horizon_min, MAE, RMSE, MAPE, n_valid, n_valid_mape``."""
    rows = []
    for name, p in preds.items():
        for j, h in enumerate(horizons):
            m = compute_metrics(p[:, j], y[:, j], valid[:, j])
            rows.append({"model": name, "horizon_steps": h, "horizon_min": h * config.STEP_MINUTES, **m})
    return pd.DataFrame(rows)


def metrics_by_group(
    preds: dict[str, np.ndarray], y: np.ndarray, valid: np.ndarray, horizons: tuple[int, ...],
    groups: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Metrics for named boolean ``[S, H]`` groups (e.g. peak / off-peak) by model and horizon."""
    rows = []
    for gname, gflag in groups.items():
        gmask = valid & gflag[:, :, None]
        for name, p in preds.items():
            for j, h in enumerate(horizons):
                m = compute_metrics(p[:, j], y[:, j], gmask[:, j])
                rows.append({"group": gname, "model": name, "horizon_steps": h,
                             "horizon_min": h * config.STEP_MINUTES, **m})
    return pd.DataFrame(rows)


def per_sensor_metrics(
    pred: np.ndarray, y: np.ndarray, valid: np.ndarray, horizons: tuple[int, ...], sensor_ids: list[str],
) -> pd.DataFrame:
    """MAE / RMSE per sensor and horizon for one model."""
    rows = []
    for n, sid in enumerate(sensor_ids):
        for j, h in enumerate(horizons):
            m = compute_metrics(pred[:, j, n], y[:, j, n], valid[:, j, n])
            rows.append({"node": n, "sensor_id": sid, "horizon_steps": h, **m})
    return pd.DataFrame(rows)


def improvement_table(metrics: pd.DataFrame, reference: str, candidate: str, columns=("MAE", "RMSE", "MAPE")) -> pd.DataFrame:
    """Percent improvement of ``candidate`` over ``reference`` per horizon (positive = lower error)."""
    ref = metrics[metrics.model == reference].set_index("horizon_steps")
    cand = metrics[metrics.model == candidate].set_index("horizon_steps")
    out = pd.DataFrame(index=ref.index)
    out.index.name = "horizon_steps"
    out["horizon_min"] = ref["horizon_min"]
    for c in columns:
        out[f"{c}_{reference}"] = ref[c]
        out[f"{c}_{candidate}"] = cand[c]
        out[f"{c}_improvement_pct"] = [improvement_pct(r, k) for r, k in zip(ref[c], cand[c])]
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Uncertainty of an improvement estimate: bootstrap over whole days
# --------------------------------------------------------------------------- #
def day_block_bootstrap_improvement(
    pred_ref: np.ndarray, pred_new: np.ndarray, y: np.ndarray, valid: np.ndarray,
    day_of_target: np.ndarray, n_boot: int = 2000, seed: int = config.SEED, metric: str = "MAE",
) -> dict[str, float]:
    """Percent improvement of ``pred_new`` over ``pred_ref`` with a day-block bootstrap CI.

    ``pred_*``, ``y``, ``valid`` are ``[S, N]`` for ONE horizon; ``day_of_target`` is the integer
    day index ``[S]`` of each sample's target.  Whole days are resampled to respect the strong
    temporal autocorrelation of traffic errors (resampling individual points would give
    over-confident intervals).  ``valid`` must already be the common valid mask.
    """
    days = np.unique(day_of_target)
    stat_ref, stat_new, cnt = [], [], []
    for d in days:
        sel = (day_of_target == d)[:, None] & valid
        e_ref = (pred_ref - y)[sel].astype(np.float64)
        e_new = (pred_new - y)[sel].astype(np.float64)
        if metric == "MAE":
            stat_ref.append(np.abs(e_ref).sum()); stat_new.append(np.abs(e_new).sum())
        else:  # RMSE: bootstrap on squared error sums
            stat_ref.append((e_ref ** 2).sum()); stat_new.append((e_new ** 2).sum())
        cnt.append(sel.sum())
    stat_ref, stat_new, cnt = map(np.asarray, (stat_ref, stat_new, cnt))

    def value(sr: np.ndarray, sn: np.ndarray, c: np.ndarray) -> np.ndarray:
        if metric == "MAE":
            ref_v, new_v = sr.sum(-1) / c.sum(-1), sn.sum(-1) / c.sum(-1)
        else:
            ref_v, new_v = np.sqrt(sr.sum(-1) / c.sum(-1)), np.sqrt(sn.sum(-1) / c.sum(-1))
        return 100.0 * (ref_v - new_v) / ref_v

    point = float(value(stat_ref, stat_new, cnt))
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(days), size=(n_boot, len(days)))
    boots = value(stat_ref[draw], stat_new[draw], cnt[draw])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"improvement_pct": point, "ci_low": float(lo), "ci_high": float(hi),
            "share_boot_positive": float((boots > 0).mean()), "n_days": int(len(days))}


# --------------------------------------------------------------------------- #
# Sensor selection for plots (uses training rows only, fixed rule)
# --------------------------------------------------------------------------- #
def select_representative_sensors(
    speed: np.ndarray, mask: np.ndarray, train_range: tuple[int, int], sensor_ids: list[str]
) -> dict[str, int]:
    """Four sensors chosen by an explicit rule on **training** statistics only.

    ``slowest`` (lowest mean speed), ``median`` (median-mean sensor), ``fastest`` (highest mean) and
    ``most_variable`` (highest standard deviation).  Returns ``{label: node_index}``; because the
    selection never looks at validation/test data or at any model result it cannot be cherry-picked.
    """
    a, b = train_range
    block, obs = speed[a:b], mask[a:b]
    count = np.maximum(obs.sum(0), 1)
    mean = np.where(obs, block, 0.0).sum(0) / count
    var = np.where(obs, (block - mean) ** 2, 0.0).sum(0) / count
    order = np.argsort(mean)
    chosen = {
        "slowest": int(order[0]), "median": int(order[len(order) // 2]),
        "fastest": int(order[-1]), "most_variable": int(np.argmax(var)),
    }
    return chosen


# --------------------------------------------------------------------------- #
# Interpretation guard
# --------------------------------------------------------------------------- #
def report_claims(claims: dict[str, bool], column: str = "claim") -> pd.DataFrame:
    """Table of interpretation claims and whether they hold in this run.

    The prose in the notebooks was written for the delivered run.  If a claim does not hold (for example after
    retraining with another seed) a loud warning is printed so the text is re-read; with ``TFC_STRICT_CLAIMS=1`` it
    raises instead.  Smoke runs are never judged.
    """
    table = pd.DataFrame({column: list(claims), "holds in this run": list(claims.values())})
    failed = [k for k, v in claims.items() if not v]
    if config.IS_SMOKE:
        print("(smoke run on a tiny subset: claims are not judged)")
    elif failed:
        msg = "claims that do NOT hold in this run (the interpretation text was written for the delivered run, re-read it): " + "; ".join(failed)
        if config.STRICT_CLAIMS:
            raise AssertionError(msg)
        print("*** WARNING: " + msg)
    return table
