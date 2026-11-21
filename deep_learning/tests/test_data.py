"""The DL batches must equal Phase 1's window functions and must never look into the future."""
import numpy as np
import pandas as pd
import torch

from deep_learning.dl.data import WindowData
from src import config
from src.preprocessing import (SpeedScaler, gather_model_inputs, gather_targets, locf_with_age,
                               time_feature_matrix, fill_for_model, window_anchor)

T, N = 600, 5
L, H = config.INPUT_LEN, config.MAX_HORIZON


def synthetic():
    rng = np.random.default_rng(0)
    speed = rng.uniform(5, 70, (T, N)).astype(np.float32)
    mask = rng.random((T, N)) > 0.15
    mask[100:130] = False                       # a network-wide outage
    speed[~mask] = np.nan
    ts = pd.date_range("2012-03-01", periods=T, freq="5min")
    scaler = SpeedScaler.fit(speed, mask, (0, 400))
    origins = {"train": np.arange(L - 1, 400 - H), "test": np.arange(399, T - H)}
    return speed, mask, ts, scaler, origins


def test_batch_matches_phase1_window_functions():
    speed, mask, ts, scaler, origins = synthetic()
    data = WindowData.from_arrays(speed, mask, ts, scaler, origins)
    o = np.array([11, 57, 105, 120, 333, 450, 587])
    b = data.batch(torch.as_tensor(o))

    x_ref = gather_model_inputs(fill_for_model(speed, mask, scaler), mask, time_feature_matrix(ts), o, L)
    np.testing.assert_allclose(b["x"].numpy(), x_ref, rtol=0, atol=1e-6)

    lv, la = locf_with_age(speed, mask)
    anchor_ref, _ = window_anchor(lv, la, o, L, scaler.sensor_mean)
    np.testing.assert_allclose(b["anchor"].numpy(), scaler.transform(anchor_ref), atol=1e-5)

    y_raw, y_obs = gather_targets(speed, mask, o, tuple(range(1, H + 1)))
    np.testing.assert_array_equal(b["valid"].numpy(), y_obs.astype(np.float32))
    np.testing.assert_allclose(b["y"].numpy()[y_obs], scaler.transform(y_raw[y_obs]), atol=1e-5)
    assert b["future_time"].shape == (len(o), H, N, 4)


def test_inputs_ignore_the_future_and_targets_ignore_the_past():
    speed, mask, ts, scaler, origins = synthetic()
    o = torch.tensor([200])
    before = WindowData.from_arrays(speed, mask, ts, scaler, origins).batch(o)
    wrecked = speed.copy(); wrecked[201:] = 1.0              # change everything after the origin
    after = WindowData.from_arrays(wrecked, mask, ts, scaler, origins).batch(o)
    assert torch.equal(before["x"], after["x"]) and torch.equal(before["anchor"], after["anchor"])
    assert not torch.equal(before["y"], after["y"])          # ...which only the targets see


def test_train_windows_stay_inside_training_rows():
    speed, mask, ts, scaler, origins = synthetic()
    o = origins["train"]
    assert o.min() - L + 1 >= 0 and o.max() + H < 400
