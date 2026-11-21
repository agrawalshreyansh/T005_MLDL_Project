"""Turn the cleaned speed matrix into model-ready tensors.

Shapes used everywhere (B = batch, L = 12 input steps, H = 12 output steps, N = 207 sensors):

    x            [B, L, N, 6]   scaled speed (0 where missing), observed flag, tod sin/cos, dow sin/cos
    anchor       [B, N]         last observed speed in the window (scaled): the persistence forecast
    y, valid     [B, H, N]      scaled targets and 1/0 observed flags (missing targets carry no loss)
    future_time  [B, H, N, 4]   calendar features of the target times (known in advance)

Instead of materialising every window (24k x 12 x 207 x 6 floats = 1.4 GB for training alone),
the whole series lives on the device as ``[T, N]`` tensors and a batch is gathered by index.
The values are identical to Phase 1's ``src.preprocessing.gather_model_inputs`` (tested).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src import config
from src.preprocessing import (ExperimentContext, SpeedScaler, fill_for_model, locf_with_age,
                               time_feature_matrix, window_anchor)

FEATURES = ("speed_z", "observed", "tod_sin", "tod_cos", "dow_sin", "dow_cos")


@dataclass
class WindowData:
    values: torch.Tensor          # [T, N] scaled speed, 0 (= training mean) where missing
    mask: torch.Tensor            # [T, N] 1.0 observed / 0.0 missing
    time: torch.Tensor            # [T, 4] calendar features
    anchor: torch.Tensor          # [T, N] scaled anchor for a forecast made at origin t
    origins: dict[str, torch.Tensor]
    scaler: SpeedScaler
    input_len: int = config.INPUT_LEN
    n_steps: int = config.MAX_HORIZON

    @classmethod
    def from_arrays(cls, speed: np.ndarray, mask: np.ndarray, timestamps: pd.DatetimeIndex,
                    scaler: SpeedScaler, origins: dict[str, np.ndarray], device="cpu") -> "WindowData":
        locf_value, locf_age = locf_with_age(speed, mask)
        anchor_mph, _ = window_anchor(locf_value, locf_age, np.arange(len(speed)), config.INPUT_LEN,
                                      scaler.sensor_mean)
        t = lambda a, dtype=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)  # noqa: E731
        return cls(
            values=t(fill_for_model(speed, mask, scaler)),
            mask=t(mask),
            time=t(time_feature_matrix(timestamps)),
            anchor=t(scaler.transform(anchor_mph)),
            origins={k: t(v, torch.long) for k, v in origins.items()},
            scaler=scaler,
        )

    @classmethod
    def from_context(cls, ctx: ExperimentContext, device="cpu") -> "WindowData":
        return cls.from_arrays(ctx.speed, ctx.mask, ctx.timestamps, ctx.scaler, ctx.origins, device)

    @property
    def n_nodes(self) -> int:
        return self.values.shape[1]

    def batch(self, origins: torch.Tensor) -> dict[str, torch.Tensor]:
        dev = self.values.device
        origins = origins.to(dev)
        x_idx = origins[:, None] + torch.arange(-self.input_len + 1, 1, device=dev)   # [B, L]
        y_idx = origins[:, None] + torch.arange(1, self.n_steps + 1, device=dev)      # [B, H]
        n = self.n_nodes
        cal_in = self.time[x_idx][:, :, None, :].expand(-1, -1, n, -1)                # [B, L, N, 4]
        x = torch.cat([self.values[x_idx][..., None], self.mask[x_idx][..., None], cal_in], dim=-1)
        return {
            "x": x,
            "anchor": self.anchor[origins],
            "y": self.values[y_idx],
            "valid": self.mask[y_idx],
            "future_time": self.time[y_idx][:, :, None, :].expand(-1, -1, n, -1),
        }

    def batches(self, split: str, batch_size: int, shuffle: bool = False, generator=None):
        """Yield batches over a split; shuffling happens only *inside* the split."""
        o = self.origins[split]
        if shuffle:
            o = o[torch.randperm(len(o), generator=generator).to(o.device)]
        for i in range(0, len(o), batch_size):
            yield self.batch(o[i:i + batch_size])

    def to_mph(self, z: torch.Tensor | np.ndarray) -> np.ndarray:
        z = z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z
        return self.scaler.inverse_transform(z)
