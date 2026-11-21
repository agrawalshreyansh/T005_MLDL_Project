"""The training loop must learn, respect masks, keep the best epoch and write reloadable files."""
import torch
from torch import nn

from deep_learning.dl import engine
from deep_learning.dl.data import WindowData
from deep_learning.tests.test_data import synthetic


class LinearResidual(nn.Module):
    """Tiny model: anchor + linear map of the last input step."""
    def __init__(self, n_steps=12):
        super().__init__()
        self.lin = nn.Linear(6, n_steps)

    def forward(self, b, teacher_forcing=0.0):
        return b["anchor"][:, None, :] + self.lin(b["x"][:, -1]).permute(0, 2, 1)


def test_masked_mae_ignores_missing_targets():
    pred, y = torch.zeros(2, 3), torch.tensor([[1.0, 100.0, 1.0], [1.0, 1.0, 1.0]])
    valid = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    assert engine.masked_mae(pred, y, valid).item() == 1.0


def test_teacher_forcing_decays_to_zero():
    r = [engine.teacher_forcing_ratio(e, 10, 0.5) for e in range(1, 11)]
    assert r[0] == 1.0 and r[-1] == 0.0 and all(a >= b for a, b in zip(r, r[1:]))


def test_train_improves_and_saves(tmp_path):
    speed, mask, ts, scaler, origins = synthetic()
    origins["val"] = origins.pop("test")
    data = WindowData.from_arrays(speed, mask, ts, scaler, origins)
    model = LinearResidual()
    before = engine.val_mae_mph(model, data)
    out = engine.train(model, data, dict(max_epochs=4, batch_size=32, lr=1e-2), out_dir=tmp_path, name="lin", log=lambda s: None)
    assert out["best_val_mae_mph"] <= before
    assert (tmp_path / "metrics" / "lin_history.csv").is_file()
    fresh = LinearResidual()
    engine.load_weights(fresh, tmp_path / "models" / "lin.pt")
    assert engine.val_mae_mph(fresh, data) == engine.val_mae_mph(model, data)
