"""Tabular assembly, Random Forest wrapper and tuning utilities (synthetic data, no raw files needed)."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import config, preprocessing as pp, rf_model, tabular, tuning


def _toy_ctx(toy_series):
    ts, speed, mask = toy_series
    locf_value, locf_age = pp.locf_with_age(speed, mask)
    origins = {"train": np.arange(20, 500), "val": np.arange(520, 700), "test": np.arange(720, 900)}
    scaler = SimpleNamespace(sensor_mean=np.nanmean(speed[:500], axis=0).astype(np.float32))
    return SimpleNamespace(speed=speed, mask=mask, timestamps=ts, sensor_ids=list("abcd"), locf_value=locf_value,
                           locf_age=locf_age, scaler=scaler, origins=origins)


def test_targets_are_future_residuals_aligned_with_rows(toy_series):
    ctx = _toy_ctx(toy_series)
    lat = lon = np.zeros(4, dtype=np.float32)
    td = tabular.build_tabular(ctx, lat, lon, train_stride=5, val_stride=3, splits=("train", "val"))
    res, ok, anchor = td.targets("train", 3)
    o = td.origin_sets["train"]
    s, n = 7, 2
    assert ok[s * 4 + n] == ctx.mask[o[s] + 3, n]
    if ok[s * 4 + n]:
        assert np.isclose(res[s * 4 + n] + anchor[s * 4 + n], ctx.speed[o[s] + 3, n])


def test_multi_targets_require_all_horizons_observed(toy_series):
    ctx = _toy_ctx(toy_series)
    td = tabular.build_tabular(ctx, np.zeros(4), np.zeros(4), train_stride=1, splits=("train",))
    hz = (1, 3, 6, 12)
    res, ok_all, _ = td.multi_targets("train", hz)
    assert res.shape == (len(td.origin_sets["train"]) * 4, 4)
    per_h = np.stack([td.targets("train", h)[1] for h in hz], axis=1)
    assert np.array_equal(ok_all, per_h.all(axis=1)) and ok_all.sum() < len(ok_all)


def test_flat_matrix_has_no_sensor_id_and_origin_calendar(toy_series):
    ctx = _toy_ctx(toy_series)
    td = tabular.build_tabular(ctx, np.zeros(4), np.zeros(4), train_stride=10, splits=("train",))
    x, names = td.flat_matrix("train")
    assert x.shape[1] == len(names) and "sensor_id" not in names and set(tabular.ORIGIN_CALENDAR) <= set(names)
    # the calendar columns describe the origin time (same value for the 4 sensors of one origin)
    c = x[:, names.index("origin_tod_sin")].reshape(-1, 4)
    assert (c == c[:, :1]).all()


def test_forest_handles_nan_features_and_adds_the_anchor():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5)).astype(np.float32)
    y = np.stack([x[:, 0], 2 * x[:, 1], x[:, 0] + x[:, 2]], axis=1)
    x_nan = x.copy(); x_nan[::7, 3] = np.nan
    params = dict(n_estimators=20, min_samples_leaf=3, max_features=1.0, n_jobs=1, random_state=0)
    model = rf_model.fit_forest(x_nan, y, params)
    anchor = np.full(len(x), 10.0, dtype=np.float32)
    pred = rf_model.predict_speed(model, x_nan, anchor, chunk=64)
    assert pred.shape == y.shape and np.isfinite(pred).all()
    assert np.abs((pred - 10.0) - y).mean() < 0.3 * np.abs(y).mean() + 0.1
    with pytest.raises(ValueError):
        rf_model.fit_forest(x, np.where(y > 1, np.nan, y), params)


def test_sampled_configs_are_distinct_valid_and_exclude_the_default():
    cfgs = tuning.sample_configs(12, seed=1)
    sigs = {tuple(c.values()) for c in cfgs}
    assert len(sigs) == 12 and tuple(tuning.default_candidate().values()) not in sigs
    for c in cfgs:
        assert all(v in tuning.SEARCH_SPACE[k] for k, v in c.items())
    assert cfgs == tuning.sample_configs(12, seed=1)          # reproducible
    with pytest.raises(ValueError):
        tuning.sample_configs(10 ** 6)


def test_rolling_origin_folds_never_leak_future_targets():
    origins = np.arange(100, 100 + 3 * 400, 3)                # strictly increasing, uneven spacing in time
    folds = tuning.rolling_origin_folds(origins, n_folds=3, max_horizon=12)
    assert len(folds) == 3
    prev_val_end = -1
    for train, val in folds:
        assert train.max() < val.min()
        assert origins[train].max() + 12 < origins[val].min()           # no training target reaches the validation block
        assert val.min() > prev_val_end                                 # validation blocks move forward in time
        prev_val_end = val.max()
    assert len(folds[0][0]) < len(folds[-1][0])                          # expanding window
    with pytest.raises(ValueError):
        tuning.rolling_origin_folds(origins[::-1], 3)


def test_rows_for_origins_match_origin_major_layout():
    rows = tuning.rows_for_origins(np.array([2, 5]), 4)
    assert rows.tolist() == [8, 9, 10, 11, 20, 21, 22, 23]


def test_tuning_params_use_the_reduced_round_budget():
    before = dict(config.XGB_PARAMS)
    p = tuning.tuning_params({"max_depth": 4})
    assert p["max_depth"] == 4 and p["n_estimators"] == config.TUNE_MAX_ROUNDS
    assert p["early_stopping_rounds"] == config.TUNE_EARLY_STOPPING
    assert config.XGB_PARAMS == before                                   # the global config is not mutated
    assert tuning.tuning_params({})["max_depth"] == config.XGB_PARAMS["max_depth"]
