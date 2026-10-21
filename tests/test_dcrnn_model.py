"""Optional DCRNN re-implementation: supports, shapes, graph locality."""
import numpy as np
import pytest
import torch

from src import dcrnn_model as dm


def chain(n=5):
    a = np.eye(n, dtype=np.float32)
    for i in range(n - 2):
        a[i, i + 1] = 0.8                          # 0 -> 1 -> 2 -> 3 ; node 4 isolated
    return a


def test_random_walk_supports_are_row_stochastic_where_defined_and_directed():
    fwd, bwd = dm.random_walk_supports(chain())
    np.testing.assert_allclose(fwd.sum(1)[[0, 1, 2]], 1.0, rtol=1e-6)      # nodes with an out-edge
    assert fwd[3].sum() == 0 and fwd[4].sum() == 0                          # no out-edge -> zero row, not NaN
    assert fwd[0, 1] == pytest.approx(1.0) and bwd[1, 0] == pytest.approx(1.0)
    assert np.isfinite(fwd).all() and np.isfinite(bwd).all() and fwd.diagonal().sum() == 0


def test_forward_shapes_and_finiteness():
    torch.manual_seed(0)
    m = dm.DCRNN(chain(), in_features=6, hidden=8, n_layers=2, k_hops=2).eval()
    out = m(torch.randn(2, 12, 5, 6), torch.zeros(2, 5), torch.randn(2, 12, 5, 4))
    assert out.shape == (2, 12, 5) and torch.isfinite(out).all()


def test_receptive_field_follows_the_graph():
    torch.manual_seed(0)
    m = dm.DCRNN(chain(), in_features=6, hidden=8, n_layers=1, k_hops=1).eval()

    def influences(src, dst):
        x = torch.randn(1, 12, 5, 6); a = torch.zeros(1, 5); d = torch.randn(1, 12, 5, 4)
        with torch.no_grad():
            base = m(x, a, d)[0, :, dst]
            x2 = x.clone(); x2[:, :, src] += 3.0
            return not torch.allclose(base, m(x2, a, d)[0, :, dst], atol=1e-6)

    assert influences(0, 1) and influences(1, 0)       # directed diffusion in both directions (forward + backward walks)
    assert not influences(4, 0) and not influences(0, 4)    # isolated node does not interact
