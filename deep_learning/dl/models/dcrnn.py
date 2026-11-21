"""Rung 3: DCRNN, diffusion-convolutional recurrent network (Li et al. 2018), re-implemented.

Instead of attention, information spreads over the graph like a random walk: one diffusion
step mixes each sensor with its neighbours in proportion to road closeness, two steps reach
neighbours of neighbours. Forward walks follow the traffic direction (upstream influence),
backward walks go against it (downstream influence, e.g. a queue backing up).

This graph convolution replaces the matrix multiplications *inside* every GRU gate, and the
model is an encoder-decoder: the encoder reads 12 steps, the decoder writes 12 steps one at a
time, feeding each prediction back in. During training the decoder is sometimes given the true
previous value instead (teacher forcing), less and less often as training goes on.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src import config


def random_walk_supports(adj: np.ndarray) -> torch.Tensor:
    """``[2, N, N]``: row i of each matrix says how sensor i mixes in the others after one step.

    forward  = (D_out^-1 A)^T  -> sensor i gathers from sensors j with an edge j -> i (upstream)
    backward = (D_in^-1 A^T)^T -> sensor i gathers from sensors j with an edge i -> j (downstream)
    Self-loops (adjacency diagonal = 1) keep part of each sensor's own state.
    """
    def rw(a):
        d = a.sum(axis=1, keepdims=True)
        return np.divide(a, d, out=np.zeros_like(a), where=d > 0)
    return torch.as_tensor(np.stack([rw(adj).T, rw(adj.T).T]), dtype=torch.float32)


def diffusion_powers(supports: torch.Tensor, k_hops: int) -> torch.Tensor:
    """``[2K, N, N]`` = S_f, S_f^2, .., S_b, S_b^2, ..: every diffusion step precomputed once.

    Applying all of them in one batched multiply is the same maths as stepping S @ (S @ x),
    with far fewer GPU kernel launches (the model is launch-bound: 24 steps x 2 layers x 2 gates).
    """
    mats = []
    for s in supports:
        p = s
        for _ in range(k_hops):
            mats.append(p)
            p = s @ p
    return torch.stack(mats)


class DiffusionConv(nn.Module):
    """``[B, N, C] -> [B, N, out]``: concatenate x and S^k x (both directions, k = 1..K), then a linear layer."""

    def __init__(self, in_dim: int, out_dim: int, k_hops: int):
        super().__init__()
        self.lin = nn.Linear(in_dim * (1 + 2 * k_hops), out_dim)

    def forward(self, x: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        k = powers.shape[0]
        flat = x.permute(1, 0, 2).reshape(n, b * c)                              # [N, B*C]
        diffused = (powers.reshape(k * n, n) @ flat).view(k, n, b, c)             # one matrix multiply
        diffused = diffused.permute(2, 1, 0, 3).reshape(b, n, k * c)              # [B, N, 2K*C]
        return self.lin(torch.cat([x, diffused], dim=-1))


class DCGRUCell(nn.Module):
    """A GRU cell whose gates are diffusion convolutions over ``[input, hidden]``."""

    def __init__(self, in_dim: int, hidden: int, k_hops: int):
        super().__init__()
        self.hidden = hidden
        self.gates = DiffusionConv(in_dim + hidden, 2 * hidden, k_hops)
        self.cand = DiffusionConv(in_dim + hidden, hidden, k_hops)

    def forward(self, x, h, supports):
        r, u = torch.sigmoid(self.gates(torch.cat([x, h], -1), supports)).chunk(2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], -1), supports))
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    uses_teacher_forcing = True

    def __init__(self, adj: np.ndarray, hidden: int = 64, n_layers: int = 2, k_hops: int = 2,
                 n_steps: int = config.MAX_HORIZON):
        super().__init__()
        self.register_buffer("supports", diffusion_powers(random_walk_supports(adj), k_hops), persistent=False)
        self.n_steps = n_steps
        self.encoder = nn.ModuleList(DCGRUCell(6 if i == 0 else hidden, hidden, k_hops) for i in range(n_layers))
        self.decoder = nn.ModuleList(DCGRUCell(5 if i == 0 else hidden, hidden, k_hops) for i in range(n_layers))
        self.out = nn.Linear(hidden, 1)

    def _step(self, cells, inp, hs):
        for i, cell in enumerate(cells):
            hs[i] = cell(inp, hs[i], self.supports)
            inp = hs[i]
        return inp

    def forward(self, b: dict, teacher_forcing: float = 0.0) -> torch.Tensor:
        x, anchor = b["x"], b["anchor"]
        bsz, l, n, _ = x.shape
        hs = [x.new_zeros(bsz, n, cell.hidden) for cell in self.encoder]
        for t in range(l):
            self._step(self.encoder, x[:, t], hs)
        prev, outs = anchor, []                  # the decoder starts from the last observed speed
        for t in range(self.n_steps):
            inp = torch.cat([prev.unsqueeze(-1), b["future_time"][:, t]], dim=-1)    # [B, N, 5]
            pred = anchor + self.out(self._step(self.decoder, inp, hs)).squeeze(-1)
            outs.append(pred)
            if self.training and teacher_forcing > 0 and torch.rand(()) < teacher_forcing:
                prev = torch.where(b["valid"][:, t] > 0, b["y"][:, t], pred.detach())
            else:
                prev = pred
        return torch.stack(outs, dim=1)
