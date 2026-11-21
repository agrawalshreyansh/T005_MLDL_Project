"""DCRNN: diffusion matrices must be proper random walks in the right direction."""
import numpy as np

from deep_learning.dl.models.dcrnn import random_walk_supports
from deep_learning.tests.test_models import ADJ


def test_random_walk_rows_sum_to_one_and_follow_direction():
    s = random_walk_supports(ADJ).numpy()
    np.testing.assert_allclose(s[0].sum(axis=0), 1.0, atol=1e-6)      # columns of (D^-1 A)^T sum to 1
    assert s[0][1, 0] > 0 and s[0][0, 1] == 0                          # forward: 1 gathers from upstream 0
    assert s[1][0, 1] > 0 and s[1][1, 0] == 0                          # backward: 0 gathers from downstream 1
