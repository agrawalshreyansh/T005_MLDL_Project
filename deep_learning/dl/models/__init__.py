"""Neural forecasting models and the registry used by ``deep_learning.train``."""
from __future__ import annotations

import numpy as np
from torch import nn

from .dcrnn import DCRNN
from .st_gat import STGAT
from .temporal import TemporalGRU

# name -> (builder, training overrides). Every model shares the same engine defaults otherwise.
MODELS = {
    "gru": (lambda adj, **kw: TemporalGRU(adj.shape[0], **kw), {}),
    "st_gat": (lambda adj, **kw: STGAT(adj, use_graph=True, **kw), {}),
    "st_gat_noedge": (lambda adj, **kw: STGAT(adj, use_graph=False, **kw), {}),
    "dcrnn": (lambda adj, **kw: DCRNN(adj, **kw), {"batch_size": 32}),
}


def build(name: str, adj: np.ndarray, **kwargs) -> nn.Module:
    return MODELS[name][0](adj, **kwargs)
