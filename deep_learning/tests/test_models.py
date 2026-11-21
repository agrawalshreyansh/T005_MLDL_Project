"""Shape and "can it memorise a tiny batch" checks, run for every registered model."""
import numpy as np
import pytest
import torch

from deep_learning.dl import engine
from deep_learning.dl.data import WindowData
from deep_learning.dl.models import MODELS, build
from deep_learning.tests.test_data import synthetic

N = 5
ADJ = np.eye(N, dtype=np.float32)
ADJ[0, 1] = 0.8; ADJ[1, 2] = 0.5; ADJ[3, 2] = 0.9          # directed chain 0 -> 1 -> 2 <- 3, sensor 4 isolated


def tiny_data():
    speed, mask, ts, scaler, origins = synthetic()
    return WindowData.from_arrays(speed, mask, ts, scaler, {"train": origins["train"][:8], "val": origins["train"][:8]})


@pytest.mark.parametrize("name", list(MODELS))
def test_output_shape(name):
    data = tiny_data()
    model = build(name, ADJ, **({"hidden": 8} if name == "dcrnn" else {"d_model": 8}))
    out = model(data.batch(data.origins["train"][:3]))
    assert out.shape == (3, 12, N) and torch.isfinite(out).all()


@pytest.mark.parametrize("name", list(MODELS))
def test_overfits_one_small_batch(name):
    """If a model cannot memorise 8 windows, the bug is in the code, not in the data."""
    torch.manual_seed(0)
    data = tiny_data()
    model = build(name, ADJ, **({"hidden": 16} if name == "dcrnn" else {"d_model": 16, "dropout": 0.0}))
    b = data.batch(data.origins["train"])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(150):
        loss = engine.masked_mae(model(b), b["y"], b["valid"])
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.35 * first, (name, first, loss.item())
