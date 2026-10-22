"""Chronological splits, training-only scaling, window generation and leakage checks."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src import preprocessing as pp
from src.baselines import SeasonalAverage
from src.feature_engineering import build_base_features


def test_split_boundaries_are_contiguous_ordered_and_day_aligned(toy_series):
    ts, _, _ = toy_series
    sp = pp.make_chronological_splits(ts, 0.5, 0.2, align_to_day=True)
    assert sp.train[0] == 0 and sp.train[1] == sp.val[0] and sp.val[1] == sp.test[0] and sp.test[1] == len(ts)
    for name in pp.SPLIT_NAMES:
        a, b = sp.range(name)
        assert b > a
    for end in (sp.train[1], sp.val[1]):
        assert ts[end].hour == 0 and ts[end].minute == 0       # boundaries fall on midnight
    assert sp.first_timestamp["val"] > sp.last_timestamp["train"]
    assert sp.first_timestamp["test"] > sp.last_timestamp["val"]


def test_split_rejects_unsorted_or_degenerate_input(toy_series):
    ts, _, _ = toy_series
    with pytest.raises(ValueError):
        pp.make_chronological_splits(ts[::-1], 0.7, 0.1)
    with pytest.raises(ValueError):
        pp.make_chronological_splits(ts, 0.9, 0.2)


def test_origins_have_all_targets_inside_their_split(toy_series):
    ts, _, _ = toy_series
    n = len(ts)
    sp = pp.make_chronological_splits(ts, 0.5, 0.2)
    L, H = 12, 12
    for name in pp.SPLIT_NAMES:
        origins = pp.valid_origins(sp.range(name), L, H, n, name)
        assert len(origins) > 0
        pp.assert_no_leakage(origins, sp.range(name), L, H, n)
        _, y_idx = pp.window_index_arrays(origins, L, H)
        a, b = sp.range(name)
        assert y_idx.min() >= a and y_idx.max() < b


def test_train_windows_never_touch_validation_or_test_rows(toy_series):
    ts, _, _ = toy_series
    sp = pp.make_chronological_splits(ts, 0.5, 0.2)
    origins = pp.valid_origins(sp.train, 12, 12, len(ts), "train")
    x_idx, y_idx = pp.window_index_arrays(origins, 12, 12)
    assert x_idx.max() < sp.train[1] and y_idx.max() < sp.train[1]


def test_val_and_test_may_use_earlier_rows_only_as_input_context(toy_series):
    ts, _, _ = toy_series
    sp = pp.make_chronological_splits(ts, 0.5, 0.2)
    origins = pp.valid_origins(sp.test, 12, 12, len(ts), "test")
    x_idx, y_idx = pp.window_index_arrays(origins, 12, 12)
    assert x_idx.min() < sp.test[0]            # legitimate past context from validation
    assert y_idx.min() >= sp.test[0]           # but no target outside the test period
    assert (x_idx.max(axis=1) == origins).all()


def test_assert_no_leakage_detects_a_violation(toy_series):
    ts, _, _ = toy_series
    sp = pp.make_chronological_splits(ts, 0.5, 0.2)
    bad = np.array([sp.train[1] - 3])          # targets spill into validation
    with pytest.raises(AssertionError):
        pp.assert_no_leakage(bad, sp.train, 12, 12, len(ts))


def test_window_generation_values_and_shapes():
    speed = np.arange(50, dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
    origins = np.array([15, 20])
    x_idx, y_idx = pp.window_index_arrays(origins, 12, 12)
    assert x_idx.shape == (2, 12) and y_idx.shape == (2, 12)
    np.testing.assert_array_equal(speed[x_idx[0], 0], np.arange(4, 16))        # inputs t-11..t
    np.testing.assert_array_equal(speed[y_idx[0], 0], np.arange(16, 28))       # targets t+1..t+12
    y, ok = pp.gather_targets(speed, np.ones_like(speed, dtype=bool), origins, (1, 3, 6, 12))
    assert y.shape == (2, 4, 3) and ok.all()
    np.testing.assert_array_equal(y[0, :, 0], [16, 18, 21, 27])


def test_scaler_is_fitted_on_training_rows_only(toy_series):
    ts, speed, mask = toy_series
    sp = pp.make_chronological_splits(ts, 0.5, 0.2)
    scaler = pp.SpeedScaler.fit(speed, mask, sp.train)
    tampered = speed.copy()
    tampered[sp.val[0]:] = 1e6                         # wreck validation + test
    scaler2 = pp.SpeedScaler.fit(tampered, mask, sp.train)
    assert scaler.mean == scaler2.mean and scaler.std == scaler2.std
    a, b = sp.train
    manual = speed[a:b][mask[a:b]].astype(np.float64)
    assert scaler.mean == pytest.approx(manual.mean()) and scaler.std == pytest.approx(manual.std())
    z = scaler.transform(manual.astype(np.float32))
    np.testing.assert_allclose(scaler.inverse_transform(z), manual, rtol=1e-5)


def test_missing_values_are_never_treated_as_zero_speed(toy_series):
    ts, speed, mask = toy_series
    sp = pp.make_chronological_splits(ts, 0.5, 0.2)
    scaler = pp.SpeedScaler.fit(speed, mask, sp.train)
    assert scaler.mean > 10                                       # zeros would drag the mean down
    filled = pp.fill_for_model(speed, mask, scaler)
    assert np.all(filled[~mask] == 0.0)                           # 0 in *scaled* space == training mean
    assert np.isfinite(filled).all()


def test_locf_uses_only_past_and_reports_age(toy_series):
    _, speed, mask = toy_series
    value, age = pp.locf_with_age(speed, mask)
    assert np.isnan(value[5:9, 1]).sum() == 0                    # carried forward from t=4
    np.testing.assert_allclose(value[5:9, 1], speed[4, 1])
    np.testing.assert_array_equal(age[4:10, 1], [0, 1, 2, 3, 4, 0])
    # changing a future reading must not change earlier LOCF values
    s2 = speed.copy(); s2[300:, 0] = 999
    v2, _ = pp.locf_with_age(s2, mask)
    np.testing.assert_array_equal(value[:300], v2[:300])


def test_window_anchor_falls_back_when_the_whole_window_is_missing(toy_series):
    _, speed, mask = toy_series
    value, age = pp.locf_with_age(speed, mask)
    fallback = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    origins = np.array([20, 730])                                 # sensor 3 missing t=700..759
    anchor, used = pp.window_anchor(value, age, origins, 12, fallback)
    assert not used[0].any()
    assert used[1, 3] and anchor[1, 3] == 4.0                     # fallback, not zero, not stale data
    assert not used[1, :3].any()


def test_tabular_features_do_not_leak_the_future(toy_series):
    ts, speed, mask = toy_series
    value, age = pp.locf_with_age(speed, mask)
    sensor_mean = np.nanmean(speed, axis=0).astype(np.float32)
    lat = np.zeros(4, np.float32); lon = np.zeros(4, np.float32)
    origins = np.array([400, 410, 420])
    base, names = build_base_features(speed, value, age, sensor_mean, origins, lat, lon)

    future = speed.copy(); future[421:] = 777.0                  # everything after the last origin
    v2, a2 = pp.locf_with_age(future, np.where(np.isnan(future), False, True))
    base2, _ = build_base_features(future, v2, a2, sensor_mean, origins, lat, lon)
    np.testing.assert_array_equal(np.nan_to_num(base, nan=-1), np.nan_to_num(base2, nan=-1))

    # features of origin t depend on readings up to t only: change t+1 for a single origin
    lag0 = names.index("lag_0")
    one = speed.copy(); one[401] = -5.0
    v3, a3 = pp.locf_with_age(one, ~np.isnan(one))
    b3, _ = build_base_features(one, v3, a3, sensor_mean, np.array([400]), lat, lon)
    np.testing.assert_array_equal(np.nan_to_num(b3[:, :], nan=-1), np.nan_to_num(base[:4], nan=-1))
    assert names[lag0] == "lag_0"


def test_seasonal_average_ignores_validation_and_test_data(toy_series):
    ts, speed, mask = toy_series
    sp = pp.make_chronological_splits(ts, 0.5, 0.2)
    a = SeasonalAverage.fit(speed, mask, ts, sp.train, min_count=1)
    tampered = speed.copy(); tampered[sp.val[0]:] = 5e5
    b = SeasonalAverage.fit(tampered, mask, ts, sp.train, min_count=1)
    np.testing.assert_array_equal(np.nan_to_num(a.table_dow), np.nan_to_num(b.table_dow))
    np.testing.assert_array_equal(a.sensor_mean, b.sensor_mean)


def test_seasonal_prediction_is_finite_and_never_zero_for_unseen_cells(toy_series):
    ts, speed, mask = toy_series
    sp = pp.make_chronological_splits(ts, 0.5, 0.2)
    seas = SeasonalAverage.fit(speed, mask, ts, sp.train, min_count=3)
    pred, level = seas.predict_at(ts[sp.test[0]:sp.test[0] + 288], return_level=True)
    assert np.isfinite(pred).all() and (pred > 0).all()
    assert set(np.unique(level)) <= {1, 2, 3}


def test_missingness_summary_counts(toy_series):
    ts, _, mask = toy_series
    sp = pp.make_chronological_splits(ts, 0.5, 0.2)
    summary = pp.missingness_summary(mask, sp)
    assert summary["missing_cells"].sum() == int((~mask).sum())
    assert summary["timesteps"].sum() == len(ts)


def test_time_features_are_cyclic_and_bounded(toy_series):
    ts, _, _ = toy_series
    f = pp.time_feature_matrix(ts)
    assert f.shape == (len(ts), 4) and np.abs(f).max() <= 1.0 + 1e-6
    np.testing.assert_allclose(f[0, :2], [0.0, 1.0], atol=1e-6)            # midnight
    np.testing.assert_allclose(f[0, :2], f[288, :2], atol=1e-6)            # same time of day, next day
    assert not np.allclose(f[0, 2:], f[288, 2:])                           # but a different weekday


def test_neighbour_features_aggregate_the_right_sensors_without_future_leakage():
    from src.feature_engineering import build_neighbour_features
    t, n = 30, 4
    speed = np.tile(np.array([10.0, 20.0, 40.0, 60.0], dtype=np.float32), (t, 1))
    speed[:, 0] += np.arange(t)                            # sensor 0 changes over time
    adj = np.eye(n, dtype=np.float32)
    adj[0, 1] = 1.0; adj[2, 1] = 0.5                        # edges 0 -> 1 (weight 1) and 2 -> 1 (weight 0.5); 3 isolated
    origins = np.array([10, 20])
    feats, names = build_neighbour_features(speed, adj, origins)
    f = feats.reshape(len(origins), n, -1)
    up_mean, up_min, up_n = names.index("up_mean_lag0"), names.index("up_min_lag0"), names.index("up_n_observed")
    # sensor 1 at origin 10: upstream = {0 (speed 20, w 1), 2 (speed 40, w .5)} -> weighted mean 26.67, min 20
    assert f[0, 1, up_mean] == pytest.approx((20 * 1 + 40 * 0.5) / 1.5)
    assert f[0, 1, up_min] == 20 and f[0, 1, up_n] == 2
    assert f[0, 0, names.index("down_mean_lag0")] == pytest.approx(20.0)      # sensor 0's downstream neighbour is 1
    assert f[0, 1, names.index("up_mean_lag3")] == pytest.approx((17 * 1 + 40 * 0.5) / 1.5)  # 15 min earlier
    assert np.isnan(f[0, 3, up_mean]) and f[0, 3, up_n] == 0                  # isolated sensor: no neighbours
    future = speed.copy(); future[21:] = 999.0                                # nothing after the last origin may matter
    f2, _ = build_neighbour_features(future, adj, origins)
    np.testing.assert_array_equal(np.nan_to_num(feats, nan=-1), np.nan_to_num(f2, nan=-1))
    miss = speed.copy(); miss[10, 2] = np.nan                                # a missing neighbour is ignored, not zero
    f3, _ = build_neighbour_features(miss, adj, origins)
    assert f3.reshape(2, n, -1)[0, 1, up_mean] == pytest.approx(20.0) and f3.reshape(2, n, -1)[0, 1, up_n] == 1
