"""Rung 2: ST-GAT = the temporal GRU + graph attention over the road network.

After the GRU has summarised each sensor's hour, two graph-attention layers let every sensor
read the summaries of its road neighbours, weighting each neighbour by a learned score
(Velickovic et al. 2018, PyTorch Geometric ``GATConv``).

The road graph is directed (only ~27% of edges run both ways), so each layer has two
attention convolutions: one over *upstream* sensors (edges j -> i) and one over *downstream*
sensors (edges i -> j). The adjacency weight (closeness on the road) is an edge feature.

``use_graph=False`` keeps every parameter but removes all edges, so each sensor only attends
to itself. Comparing the two isolates what the road graph adds (hypothesis H1).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATConv

from src import config
from .temporal import TemporalEncoder, horizon_head


def directed_edges(adj: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """``edge_index [2, E]`` (row 0 = source, row 1 = target) for ``adj[i, j] > 0, i != j``, plus weights ``[E, 1]``."""
    a = adj.copy()
    np.fill_diagonal(a, 0.0)                     # GATConv adds its own self-loops
    src, dst = np.nonzero(a > 0)
    return torch.as_tensor(np.stack([src, dst]), dtype=torch.long), torch.as_tensor(a[src, dst, None], dtype=torch.float32)


class GraphAttentionBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, attn_dropout: float):
        super().__init__()
        kw = dict(heads=heads, dropout=attn_dropout, edge_dim=1, add_self_loops=True, fill_value=1.0)
        self.upstream = GATConv(d_model, d_model // heads, **kw)
        self.downstream = GATConv(d_model, d_model // heads, **kw)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, g, ei_up, w_up, ei_down, w_down, return_attention=False):
        up = self.upstream(g, ei_up, w_up, return_attention_weights=return_attention)
        down = self.downstream(g, ei_down, w_down, return_attention_weights=return_attention)
        if return_attention:
            (up, att_up), (down, att_down) = up, down
        out = self.norm(g + self.drop(nn.functional.elu(up + down)))    # residual connection + layer norm
        return (out, {"upstream": att_up, "downstream": att_down}) if return_attention else out


class STGAT(nn.Module):
    def __init__(self, adj: np.ndarray, d_model: int = 64, heads: int = 4, n_layers: int = 2,
                 node_emb_dim: int = 16, dropout: float = 0.1, attn_dropout: float = 0.1,
                 use_graph: bool = True, n_steps: int = config.MAX_HORIZON):
        super().__init__()
        self.n_nodes = adj.shape[0]
        ei, w = directed_edges(adj) if use_graph else (torch.empty(2, 0, dtype=torch.long), torch.empty(0, 1))
        # upstream conv: messages travel along the road direction (j -> i); downstream: reversed
        self.register_buffer("ei_up", ei, persistent=False)
        self.register_buffer("ei_down", ei.flip(0), persistent=False)
        self.register_buffer("w", w, persistent=False)
        self.encoder = TemporalEncoder(self.n_nodes, 6, d_model, node_emb_dim)
        self.blocks = nn.ModuleList(GraphAttentionBlock(d_model, heads, dropout, attn_dropout) for _ in range(n_layers))
        self.head = horizon_head(2 * d_model, n_steps, dropout)

    def _batched(self, ei: torch.Tensor, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack B copies of the graph into one big disconnected graph (PyG's batching trick)."""
        offsets = torch.arange(b, device=ei.device).repeat_interleave(ei.shape[1]) * self.n_nodes
        return ei.repeat(1, b) + offsets, self.w.repeat(b, 1)

    def forward(self, b: dict, teacher_forcing: float = 0.0, return_attention: bool = False):
        h0 = self.encoder(b["x"])                                      # [B, N, d]
        bsz = h0.shape[0]
        ei_up, w = self._batched(self.ei_up, bsz)
        ei_down, _ = self._batched(self.ei_down, bsz)
        g, attention = h0.reshape(bsz * self.n_nodes, -1), []
        for block in self.blocks:
            g = block(g, ei_up, w, ei_down, w, return_attention)
            if return_attention:
                g, att = g
                attention.append(att)
        g = g.view(bsz, self.n_nodes, -1)
        residual = self.head(torch.cat([g, h0], dim=-1)).permute(0, 2, 1)   # skip: head also sees the pre-graph state
        pred = b["anchor"].unsqueeze(1) + residual
        return (pred, attention) if return_attention else pred
