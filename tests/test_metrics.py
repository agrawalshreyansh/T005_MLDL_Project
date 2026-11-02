"""MAE / RMSE / MAPE, missing-target handling, aggregation across sensors and horizons."""
import numpy as np
import pandas as pd
import pytest

from src import metrics as M
from src import evaluation as ev
from src import config


def test_basic_metric_values():
    true = np.array([10.0, 20.0, 40.0]); pred = np.array([12.0, 18.0, 30.0])
    mae, n = M.masked_mae(pred, true)
    rmse, _ = M.masked_rmse(pred, true)
    mape, n_mape = M.masked_mape(pred, true)
    assert n == 3 and n_mape == 3
    assert mae == pytest.approx((2 + 2 + 10) / 3)
    assert rmse == pytest.approx(np.sqrt((4 + 4 + 100) / 3))
    assert mape == pytest.approx(100 * (0.2 + 0.1 + 0.25) / 3)


def test_missing_targets_are_excluded_and_not_counted():
    true = np.array([10.0, np.nan, 40.0, 25.0]); pred = np.array([11.0, 99.0, 38.0, 25.0])
    obs = np.array([True, False, True, False])               # last one flagged missing by the mask
    mae, n = M.masked_mae(pred, true, obs)
    assert n == 2 and mae == pytest.approx((1 + 2) / 2)


def test_nonfinite_predictions_are_excluded():
    true = np.array([10.0, 20.0]); pred = np.array([np.nan, 22.0])
    mae, n = M.masked_mae(pred, true)
    assert n == 1 and mae == 2.0


def test_no_valid_points_gives_nan_not_zero():
    mae, n = M.masked_mae(np.array([1.0]), np.array([1.0]), np.array([False]))
    assert n == 0 and np.isnan(mae)


def test_mape_excludes_near_zero_actuals_but_mae_keeps_them():
    true = np.array([0.5, 50.0]); pred = np.array([30.0, 55.0])
    mape, n_mape = M.masked_mape(pred, true, min_true=1.0)
    mae, n = M.masked_mae(pred, true)
    assert n == 2 and n_mape == 1
    assert mape == pytest.approx(10.0)
    assert mae == pytest.approx((29.5 + 5) / 2)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.masked_mae(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        M.masked_mae(np.zeros(3), np.zeros(3), np.ones(4, dtype=bool))


def test_perfect_and_constant_offset_predictions():
    true = np.linspace(20, 60, 100)
    assert M.compute_metrics(true, true)["MAE"] == 0
    m = M.compute_metrics(true + 3.0, true)
    assert m["MAE"] == pytest.approx(3.0) and m["RMSE"] == pytest.approx(3.0)


def test_pooled_rmse_is_not_the_mean_of_group_rmses():
    rng = np.random.default_rng(0)
    t1, t2 = rng.uniform(20, 60, 100), rng.uniform(20, 60, 300)
    p1, p2 = t1 + rng.normal(0, 2, 100), t2 + rng.normal(0, 8, 300)
    g1, g2 = M.compute_metrics(p1, t1), M.compute_metrics(p2, t2)
    pooled_direct = M.compute_metrics(np.r_[p1, p2], np.r_[t1, t2])
    pooled_from = M.pooled_from_groups({"a": g1, "b": g2})
    assert pooled_from["RMSE"] == pytest.approx(pooled_direct["RMSE"])
    assert pooled_from["MAE"] == pytest.approx(pooled_direct["MAE"])
    assert pooled_from["RMSE"] != pytest.approx((g1["RMSE"] + g2["RMSE"]) / 2)


def test_improvement_sign_convention():
    assert M.improvement_pct(10.0, 8.0) == pytest.approx(20.0)      # lower error -> positive
    assert M.improvement_pct(10.0, 12.0) == pytest.approx(-20.0)    # higher error -> negative
    assert np.isnan(M.improvement_pct(0.0, 1.0))


def _toy_predictions():
    rng = np.random.default_rng(1)
    S, H, N = 20, 4, 5
    y = rng.uniform(20, 60, (S, H, N)).astype(np.float32)
    ymask = rng.random((S, H, N)) > 0.2
    return y, ymask, y + rng.normal(0, 3, y.shape).astype(np.float32), y + rng.normal(0, 6, y.shape).astype(np.float32)


def test_metrics_by_horizon_matches_manual_pooling_and_shapes():
    y, ymask, p_good, p_bad = _toy_predictions()
    horizons = (1, 3, 6, 12)
    valid = ev.common_valid_mask({"good": p_good, "bad": p_bad}, ymask)
    table = ev.metrics_by_horizon({"good": p_good, "bad": p_bad}, y, valid, horizons)
    assert len(table) == 2 * 4 and set(table.columns) >= {"MAE", "RMSE", "MAPE", "n_valid", "n_valid_mape"}
    row = table[(table.model == "good") & (table.horizon_steps == 3)].iloc[0]
    sel = valid[:, 1]
    assert row.n_valid == sel.sum()
    assert row.MAE == pytest.approx(np.abs(p_good[:, 1][sel] - y[:, 1][sel]).mean(), rel=1e-5)
    assert (table[table.model == "good"].MAE.values < table[table.model == "bad"].MAE.values).all()


def test_aggregation_over_sensors_equals_pooled_metric():
    y, ymask, p, _ = _toy_predictions()
    horizons = (1, 3, 6, 12)
    ids = [f"s{i}" for i in range(y.shape[2])]
    per = ev.per_sensor_metrics(p, y, ymask, horizons, ids)
    h1 = per[per.horizon_steps == 1]
    pooled = M.pooled_from_groups({r.sensor_id: {"MAE": r.MAE, "RMSE": r.RMSE, "n_valid": r.n_valid} for r in h1.itertuples()})
    direct = M.compute_metrics(p[:, 0], y[:, 0], ymask[:, 0])
    assert pooled["MAE"] == pytest.approx(direct["MAE"]) and pooled["RMSE"] == pytest.approx(direct["RMSE"])


def test_common_valid_mask_drops_points_any_model_cannot_predict():
    y, ymask, p1, p2 = _toy_predictions()
    p2 = p2.copy(); p2[0, 0, 0] = np.nan
    v = ev.common_valid_mask({"a": p1, "b": p2}, ymask)
    assert not v[0, 0, 0] and v.sum() <= ymask.sum()


def test_prediction_roundtrip_and_shape_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path)
    pred = np.zeros((7, 4, 5), np.float32); origins = np.arange(7) + 20
    ev.save_predictions("dummy", "test", pred, origins, (1, 3, 6, 12), {"note": "x"})
    d = ev.load_predictions("dummy", "test")
    assert d["pred"].shape == (7, 4, 5) and d["horizons"] == (1, 3, 6, 12)
    np.testing.assert_array_equal(d["origins"], origins)
    with pytest.raises(ValueError):
        ev.save_predictions("bad", "test", pred[:, :3], origins, (1, 3, 6, 12))


def test_peak_definition_uses_weekday_local_hours():
    ts = pd.DatetimeIndex(["2012-06-04 08:00", "2012-06-04 12:00", "2012-06-04 16:30",
                           "2012-06-04 19:00", "2012-06-09 08:00", "2012-06-04 06:55", "2012-06-04 09:55"])
    np.testing.assert_array_equal(ev.is_peak(ts), [True, False, True, False, False, False, True])


def test_day_block_bootstrap_detects_a_clear_improvement():
    rng = np.random.default_rng(3)
    S, N = 240, 6
    y = rng.uniform(20, 60, (S, N)); valid = np.ones((S, N), bool)
    ref = y + rng.normal(0, 6, y.shape); new = y + rng.normal(0, 3, y.shape)
    out = ev.day_block_bootstrap_improvement(ref, new, y, valid, np.arange(S) // 24, n_boot=300)
    assert out["improvement_pct"] > 20 and out["ci_low"] > 0 and out["n_days"] == 10


def test_clip_predictions_bounds_and_keeps_nan(tmp_path, monkeypatch):
    raw = np.array([[[-3.0, 35.0, 72.5, np.nan]]], dtype=np.float32)
    np.testing.assert_array_equal(np.nan_to_num(ev.clip_predictions(raw), nan=-1), [[[0.0, 35.0, 70.0, -1.0]]])
    monkeypatch.setattr(config, "PREDICTIONS_DIR", tmp_path)
    ev.save_predictions("m", "val", raw, np.array([5]), (1,))
    assert np.nanmax(ev.load_predictions("m", "val")["pred"]) == 70.0
    assert ev.load_predictions("m", "val", clip=False)["pred"][0, 0, 0] == -3.0      # raw output stays on disk


def test_report_claims_warns_but_does_not_raise_unless_strict(capsys, monkeypatch):
    table = ev.report_claims({"a": True, "b": False})
    assert list(table["holds in this run"]) == [True, False]
    assert "WARNING" in capsys.readouterr().out
    monkeypatch.setattr(config, "STRICT_CLAIMS", True)
    with pytest.raises(AssertionError):
        ev.report_claims({"b": False})
    monkeypatch.setattr(config, "IS_SMOKE", True)
    ev.report_claims({"b": False})                       # smoke runs are not judged


def test_improvement_pct_handles_nonfinite_reference():
    assert np.isnan(M.improvement_pct(float("nan"), 1.0))
    assert np.isnan(M.improvement_pct(float("inf"), 1.0))
    assert M.improvement_pct(5.0, 5.0) == pytest.approx(0.0)


def test_pooled_from_groups_with_no_valid_points_is_nan():
    out = M.pooled_from_groups({"a": {"MAE": 1.0, "RMSE": 1.0, "n_valid": 0}})
    assert out["n_valid"] == 0 and np.isnan(out["MAE"]) and np.isnan(out["RMSE"])


def test_compute_metrics_respects_custom_mape_threshold():
    true = np.array([0.5, 2.0, 10.0]); pred = np.array([1.0, 2.0, 11.0])
    default = M.compute_metrics(pred, true)
    strict = M.compute_metrics(pred, true, mape_min_speed=5.0)
    assert default["n_valid"] == strict["n_valid"] == 3
    assert default["n_valid_mape"] == 2        # 0.5 mph is below the 1 mph default
    assert strict["n_valid_mape"] == 1         # only the 10 mph point survives
    assert strict["MAPE"] == pytest.approx(10.0)
