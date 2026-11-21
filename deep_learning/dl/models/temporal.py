"""Rung 1: a GRU that only looks at time.

Every sensor's 12-step history goes through the *same* GRU (weights shared by all 207 sensors),
so the model learns one general notion of "how speed evolves". A small learned embedding per
sensor lets it specialise a little (like the sensor-id feature in XGBoost). No sensor ever sees
another sensor: this is the deep-learning version of "independent segment models".
"""
from __future__ import annotations

import torch
from torch import nn

from src import config


class TemporalEncoder(nn.Module):
    """``x [B, L, N, F]`` -> one vector per sensor ``[B, N, d]`` summarising its last hour."""

    def __init__(self, n_nodes: int, in_features: int = 6, d_model: int = 64, node_emb_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.node_emb = nn.Embedding(n_nodes, node_emb_dim)
        self.fuse = nn.Linear(d_model + node_emb_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, n, _ = x.shape
        z = self.proj(x).permute(0, 2, 1, 3).reshape(b * n, l, -1)     # one sequence per (sample, sensor)
        _, h = self.gru(z)                                             # h: [1, B*N, d] = state after the last step
        h = h[-1].view(b, n, -1)
        emb = self.node_emb.weight.unsqueeze(0).expand(b, -1, -1)
        return torch.relu(self.fuse(torch.cat([h, emb], dim=-1)))


def horizon_head(d_in: int, n_steps: int, dropout: float) -> nn.Module:
    """Maps a sensor's vector to 12 residuals (one per future step) in a single shot."""
    return nn.Sequential(nn.Linear(d_in, 2 * d_in), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_in, n_steps))


class TemporalGRU(nn.Module):
    def __init__(self, n_nodes: int, d_model: int = 64, node_emb_dim: int = 16, dropout: float = 0.1,
                 n_steps: int = config.MAX_HORIZON):
        super().__init__()
        self.encoder = TemporalEncoder(n_nodes, 6, d_model, node_emb_dim)
        self.head = horizon_head(d_model, n_steps, dropout)

    def forward(self, b: dict, teacher_forcing: float = 0.0) -> torch.Tensor:
        h = self.encoder(b["x"])                                       # [B, N, d]
        residual = self.head(h).permute(0, 2, 1)                       # [B, H, N]
        return b["anchor"].unsqueeze(1) + residual                     # forecast = persistence + learned change
