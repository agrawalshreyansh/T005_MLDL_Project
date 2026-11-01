"""Central configuration for the traffic-flow forecasting project.

Everything that a notebook or module needs to agree on (paths, split rule,
window lengths, forecast horizons, metric policy, peak definition and model
hyper-parameters) lives here so that no notebook silently diverges.

Run modes
---------
``TFC_RUN_MODE=full`` (default) produces the experimental results.
``TFC_RUN_MODE=smoke`` shrinks every expensive step so the whole pipeline can
be debugged in minutes.  Smoke outputs are written with a ``_smoke`` suffix
and must never be reported as final results.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Run mode
# --------------------------------------------------------------------------- #
RUN_MODE: str = os.environ.get("TFC_RUN_MODE", "full").strip().lower()
if RUN_MODE not in {"full", "smoke"}:
    raise ValueError(f"TFC_RUN_MODE must be 'full' or 'smoke', got {RUN_MODE!r}")
IS_SMOKE: bool = RUN_MODE == "smoke"
RUN_TAG: str = "_smoke" if IS_SMOKE else ""
# Interpretation text in the notebooks was written for the delivered run. After retraining (different seeds,
# hardware) some claims may stop holding: by default that prints a warning, TFC_STRICT_CLAIMS=1 makes it an error.
STRICT_CLAIMS: bool = os.environ.get("TFC_STRICT_CLAIMS", "0") == "1"


def tagged(name: str) -> str:
    """Append the run-mode tag to a file stem (``gat`` -> ``gat_smoke`` in smoke mode)."""
    return f"{name}{RUN_TAG}"


# --------------------------------------------------------------------------- #
# Paths (all relative to the project root, never machine specific)
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
SENSOR_GRAPH_DIR: Path = DATA_RAW / "sensor_graph"
ARTIFACTS: Path = PROJECT_ROOT / "artifacts"
MODELS_DIR: Path = ARTIFACTS / "models"
PREDICTIONS_DIR: Path = ARTIFACTS / "predictions"
FIGURES_DIR: Path = ARTIFACTS / "figures"
METRICS_DIR: Path = ARTIFACTS / "metrics"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

METR_LA_H5: Path = DATA_RAW / "metr-la.h5"
H5_KEY: str = "df"
SENSOR_IDS_TXT: Path = SENSOR_GRAPH_DIR / "graph_sensor_ids.txt"
SENSOR_LOCATIONS_CSV: Path = SENSOR_GRAPH_DIR / "graph_sensor_locations.csv"
DISTANCES_CSV: Path = SENSOR_GRAPH_DIR / "distances_la_2012.csv"

# Processed artifacts produced by notebooks 02 and 03
TRAFFIC_CLEAN_NPZ: Path = DATA_PROCESSED / "traffic_clean.npz"
SPLITS_JSON: Path = DATA_PROCESSED / "splits.json"
SCALER_JSON: Path = DATA_PROCESSED / "scaler.json"
ADJACENCY_NPY: Path = DATA_PROCESSED / "adjacency.npy"
DISTANCE_MATRIX_NPY: Path = DATA_PROCESSED / "road_distance_matrix.npy"
SENSOR_ORDER_CSV: Path = DATA_PROCESSED / "sensor_order.csv"
GRAPH_META_JSON: Path = DATA_PROCESSED / "graph_meta.json"

REQUIRED_RAW_FILES: dict[Path, str] = {
    METR_LA_H5: "METR-LA traffic speeds (pandas HDFStore, key 'df'); see data/README.md",
    SENSOR_IDS_TXT: "DCRNN data/sensor_graph/graph_sensor_ids.txt",
    SENSOR_LOCATIONS_CSV: "DCRNN data/sensor_graph/graph_sensor_locations.csv",
    DISTANCES_CSV: "DCRNN data/sensor_graph/distances_la_2012.csv",
}


def ensure_dirs() -> None:
    """Create every output directory (idempotent)."""
    for directory in (
        DATA_RAW, SENSOR_GRAPH_DIR, DATA_PROCESSED, MODELS_DIR, PREDICTIONS_DIR,
        FIGURES_DIR, METRICS_DIR, REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42

# --------------------------------------------------------------------------- #
# Data semantics
# --------------------------------------------------------------------------- #
STEP_MINUTES: int = 5
# METR-LA encodes a missing reading as exactly 0.0 mph (verified in notebook 01:
# no NaNs exist and the smallest positive speed is 0.33 mph).
MISSING_SENTINEL: float = 0.0
# Timestamps in the file are naive; the sensors are in Los Angeles, and the
# weekday congestion dips (about 08h and 17h) agree with Pacific local time.
DATA_TIMEZONE: str = "America/Los_Angeles"

# --------------------------------------------------------------------------- #
# Chronological split (day aligned, exact timestamps are saved in splits.json)
# --------------------------------------------------------------------------- #
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.10          # test fraction is the remainder (0.20)
SPLIT_ALIGN_TO_DAY: bool = True

# --------------------------------------------------------------------------- #
# Forecasting windows
# --------------------------------------------------------------------------- #
INPUT_LEN: int = 12                          # 12 x 5 min = 60 minutes of history
HORIZONS: tuple[int, ...] = (1, 3, 6, 12)    # 5, 15, 30, 60 minutes ahead
MAX_HORIZON: int = max(HORIZONS)
HORIZON_MINUTES: dict[int, int] = {h: h * STEP_MINUTES for h in HORIZONS}

# --------------------------------------------------------------------------- #
# Metric policy
# --------------------------------------------------------------------------- #
# A target is *valid* if it is observed (not the missing sentinel).  MAPE
# additionally requires the true speed to be at least this many mph so that
# near-zero denominators cannot dominate the percentage error.
MAPE_MIN_SPEED: float = 1.0

# Predictions are clipped to this physically meaningful range before scoring, for EVERY model
# (notebook 01/02: observed speeds are positive and capped at 70 mph).  Clipping can only
# reduce the error of a prediction that lies outside the range of possible true values.
SPEED_BOUNDS_MPH: tuple[float, float] = (0.0, 70.0)

# --------------------------------------------------------------------------- #
# Peak / off-peak definition (fixed BEFORE any model was trained)
# --------------------------------------------------------------------------- #
# Based on the weekday speed profile in notebook 01.  Peak = weekday hours in
# [start, end) local time of the *forecast target*.  Everything else
# (weekday off-hours and all weekend hours) is off-peak.
PEAK_WINDOWS: tuple[tuple[int, int], ...] = ((7, 10), (15, 19))
PEAK_WEEKDAYS_ONLY: bool = True

# --------------------------------------------------------------------------- #
# Graph construction (DCRNN gen_adj_mx logic)
# --------------------------------------------------------------------------- #
GRAPH_NORMALIZED_K: float = 0.1   # thresholded Gaussian kernel cut-off


# Smoke mode evaluates on an evenly spaced subset of this many origins per split.
SMOKE_ORIGIN_CAP: int = 300


# --------------------------------------------------------------------------- #
# Model hyper-parameters (recorded next to the artifacts they produce)
# --------------------------------------------------------------------------- #
def _pick(full: Any, smoke: Any) -> Any:
    return smoke if IS_SMOKE else full


XGB_PARAMS: dict[str, Any] = dict(
    objective="reg:absoluteerror",      # same MAE objective as the graph model's masked-MAE loss
    n_estimators=_pick(1500, 40),       # upper bound; early stopping on validation
    learning_rate=0.08,
    max_depth=8,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    tree_method="hist",
    max_cat_to_onehot=1,
    enable_categorical=True,
    early_stopping_rounds=_pick(40, 10),
    random_state=SEED,
    n_jobs=-1,
)
# Training rows: every ``XGB_TRAIN_STRIDE``-th origin (adjacent windows are
# ~92% identical, so thinning loses little and keeps RAM manageable).
XGB_TRAIN_STRIDE: int = _pick(2, 12)
XGB_VAL_STRIDE: int = _pick(3, 12)

# Multi-output Random Forest baseline (notebook 05c): one forest predicts the four horizons jointly.
# Every tree sees ``max_samples`` of the training rows, which keeps the fit inside a laptop's RAM and time budget.
RF_PARAMS: dict[str, Any] = dict(
    n_estimators=_pick(150, 10),
    max_features=0.4,
    min_samples_leaf=20,
    max_samples=_pick(0.1, 0.5),
    criterion="squared_error",          # the criterion that supports missing feature values in scikit-learn
    n_jobs=-1,
    random_state=SEED,
)
RF_TRAIN_STRIDE: int = XGB_TRAIN_STRIDE

# XGBoost hyper-parameter study (notebook 05d): random search on a thinned training set at one horizon, then
# rolling-origin cross-validation inside the training period for the best candidates.
TUNE_HORIZON: int = 6                      # 30 minutes: the middle of the four horizons
TUNE_TRAIN_THIN: int = _pick(3, 1)         # extra thinning of the already strided training origins
TUNE_N_CONFIGS: int = _pick(12, 3)
TUNE_TOP_K: int = _pick(3, 1)
TUNE_CV_FOLDS: int = 3
TUNE_MAX_ROUNDS: int = _pick(600, 30)
TUNE_EARLY_STOPPING: int = _pick(30, 5)

GAT_PARAMS: dict[str, Any] = dict(
    d_model=64,
    n_heads=4,
    n_gat_layers=2,
    node_emb_dim=16,
    dropout=0.1,
    attn_dropout=0.1,
    batch_size=64,
    lr=2e-3,
    weight_decay=1e-5,
    max_epochs=_pick(30, 2),
    patience=_pick(6, 2),
    grad_clip=5.0,
    train_origin_limit=_pick(None, 384),   # smoke mode trains on a subset of windows
    eval_origin_limit=_pick(None, 256),
    seed=SEED,
)

# Optional DCRNN-style comparison (notebook 06b): smaller training budget than the paper
DCRNN_PARAMS: dict[str, Any] = dict(
    hidden=64,
    n_layers=2,
    k_hops=2,
    batch_size=16,                        # the unrolled seq2seq graph is memory hungry (64 used ~4.7 GB in a smoke test)
    lr=2e-3,
    weight_decay=1e-5,
    max_epochs=_pick(30, 2),             # same epoch budget as the ST-GAT
    patience=_pick(6, 2),
    grad_clip=5.0,
    train_stride=_pick(3, 1),             # use every 3rd training origin (recurrent seq2seq is slow)
    train_origin_limit=_pick(None, 192),
    seed=SEED,
)
