"""Optional DCRNN-style baseline: diffusion-convolution GRU encoder-decoder (PyTorch re-implementation).

Follows the idea of Li et al. (ICLR 2018): the recurrent cell replaces the matrix products of a GRU by
**diffusion convolutions** over the directed road graph using the forward and backward random-walk
matrices ``D_O^-1 W`` and ``D_I^-1 W^T``, inside a sequence-to-sequence model.

This is a clean re-implementation for comparison, **not** a port of the original TensorFlow code, and it
has not been checked for numerical equivalence with it.  Documented differences:

* diffusion uses plain powers of the random-walk matrices (as in the paper's formula);
* the decoder starts from the *anchor* (last observed speed) and predicts residuals w.r.t. it, exactly like
  the other learned models in this project, so that all models are compared on the same formulation;
* the decoder additionally receives the calendar features of the target time (known in advance);
* far fewer epochs than the original paper's training schedule (compute budget), with linear
  teacher-forcing decay per epoch instead of the paper's inverse-sigmoid decay per iteration.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from . import config
from .gat_model import GraphData, count_parameters, masked_mae_loss, save_checkpoint, set_seed  # noqa: F401
from .preprocessing import window_index_arrays


def random_walk_supports(adj: np.ndarray) -> list[np.ndarray]:
    """Forward ``D_O^-1 W`` and backward ``D_I^-1 W^T`` transition matrices (self-loops removed, zero rows kept zero)."""
    w = np.asarray(adj, dtype=np.float64).copy()
    np.fill_diagonal(w, 0.0)

    def normalise(m: np.ndarray) -> np.ndarray:
        deg = m.sum(axis=1, keepdims=True)
        return np.divide(m, deg, out=np.zeros_like(m), where=deg > 0)

    return [normalise(w).astype(np.float32), normalise(w.T).astype(np.float32)]


class DiffusionConv(nn.Module):
    """``sum_{support s} sum_{k=0..K} theta_{s,k} (S^k X)`` with k = 0 shared by both supports."""

    def __init__(self, in_dim: int, out_dim: int, supports: list[np.ndarray], k_hops: int = 2) -> None:
        super().__init__()
        self.k_hops, self.n_supports = k_hops, len(supports)
        for i, s in enumerate(supports):
            self.register_buffer(f"support_{i}", torch.from_numpy(s), persistent=False)
        self.n_mats = 1 + self.n_supports * k_hops
        self.lin = nn.Linear(in_dim * self.n_mats, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:               # x [B, N, C]
        feats = [x]
        for i in range(self.n_supports):
            s = getattr(self, f"support_{i}")
            cur = x
            for _ in range(self.k_hops):
                cur = torch.einsum("ij,bjc->bic", s, cur)
                feats.append(cur)
        return self.lin(torch.cat(feats, dim=-1))


class DCGRUCell(nn.Module):
    """GRU cell whose linear maps are diffusion convolutions."""

    def __init__(self, in_dim: int, hidden: int, supports: list[np.ndarray], k_hops: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.gates = DiffusionConv(in_dim + hidden, 2 * hidden, supports, k_hops)
        self.cand = DiffusionConv(in_dim + hidden, hidden, supports, k_hops)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ru = torch.sigmoid(self.gates(torch.cat([x, h], dim=-1)))
        r, u = ru.chunk(2, dim=-1)
        c = torch.tanh(self.cand(torch.cat([x, r * h], dim=-1)))
        return u * h + (1 - u) * c


class DCRNN(nn.Module):
    """Seq2seq with stacked DCGRU layers; predicts ``n_steps`` residuals w.r.t. the anchor."""

    def __init__(self, adj: np.ndarray, in_features: int = 6, hidden: int = 64, n_layers: int = 2, k_hops: int = 2,
                 n_steps: int = config.MAX_HORIZON, dec_extra: int = 4) -> None:
        super().__init__()
        supports = random_walk_supports(adj)
        self.n_steps, self.n_layers = n_steps, n_layers
        self.encoder = nn.ModuleList([DCGRUCell(in_features if i == 0 else hidden, hidden, supports, k_hops) for i in range(n_layers)])
        self.decoder = nn.ModuleList([DCGRUCell((1 + dec_extra) if i == 0 else hidden, hidden, supports, k_hops) for i in range(n_layers)])
        self.proj = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, anchor: torch.Tensor, dec_feats: torch.Tensor,
                y_true: torch.Tensor | None = None, y_valid: torch.Tensor | None = None, teacher_forcing: float = 0.0) -> torch.Tensor:
        """``x [B,T,N,F]``, ``anchor [B,N]``, ``dec_feats [B,H,N,4]`` -> prediction ``[B,H,N]`` (scaled)."""
        b, t, n, _ = x.shape
        hid = [x.new_zeros(b, n, self.encoder[0].hidden) for _ in range(self.n_layers)]
        for step in range(t):
            inp = x[:, step]
            for i, cell in enumerate(self.encoder):
                hid[i] = cell(inp, hid[i]); inp = hid[i]
        prev, outs = anchor, []
        for step in range(self.n_steps):
            inp = torch.cat([prev.unsqueeze(-1), dec_feats[:, step]], dim=-1)
            for i, cell in enumerate(self.decoder):
                hid[i] = cell(inp, hid[i]); inp = hid[i]
            pred = anchor + self.proj(inp).squeeze(-1)
            outs.append(pred)
            if self.training and y_true is not None and teacher_forcing > 0 and np.random.random() < teacher_forcing:
                prev = torch.where(y_valid[:, step] > 0, y_true[:, step], pred.detach())
            else:
                prev = pred.detach() if self.training else pred
        return torch.stack(outs, dim=1)


def decoder_features(data: GraphData, origins: np.ndarray, device: torch.device) -> torch.Tensor:
    """Calendar features of the target times ``origin+1..origin+H`` broadcast to all sensors: ``[B, H, N, 4]``."""
    _, y_idx = window_index_arrays(origins, data.input_len, data.n_steps)
    tf = data.time_feats[y_idx]                                           # [B, H, 4]
    n = data.speed.shape[1]
    tf = np.broadcast_to(tf[:, :, None, :], (*tf.shape[:2], n, tf.shape[2]))
    return torch.from_numpy(np.ascontiguousarray(tf, dtype=np.float32)).to(device)


@torch.no_grad()
def predict_scaled(model: DCRNN, data: GraphData, origins: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(origins), batch_size):
        o = origins[i:i + batch_size]
        x, anchor, _, _ = data.batch(o, device)
        out.append(model(x, anchor, decoder_features(data, o, device)).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def predict_mph(model: DCRNN, data: GraphData, origins: np.ndarray, device: torch.device,
                steps: tuple[int, ...] = config.HORIZONS) -> np.ndarray:
    z = predict_scaled(model, data, origins, device)
    return data.scaler.inverse_transform(z[:, [s - 1 for s in steps], :])


@torch.no_grad()
def evaluate_mae_mph(model: DCRNN, data: GraphData, origins: np.ndarray, device: torch.device, batch_size: int = 128) -> float:
    model.eval()
    total = count = 0.0
    for i in range(0, len(origins), batch_size):
        o = origins[i:i + batch_size]
        x, anchor, y, valid = data.batch(o, device)
        pred = model(x, anchor, decoder_features(data, o, device))
        total += float((torch.abs(pred - y) * valid).sum()); count += float(valid.sum())
    return data.scaler.std * total / max(count, 1.0)


def train_dcrnn(model: DCRNN, data: GraphData, train_origins: np.ndarray, val_origins: np.ndarray, device: torch.device,
                params: dict, checkpoint_path: Path | None = None, log: Callable[[str], None] = print) -> dict:
    """AdamW + masked MAE, validation early stopping, linear teacher-forcing decay (0.5 -> 0 over 60% of the epochs)."""
    set_seed(params["seed"]); model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(params["patience"] // 3, 1))
    rng = np.random.default_rng(params["seed"])
    hist = {"epoch": [], "train_loss_scaled": [], "val_mae_mph": [], "lr": [], "seconds": [], "teacher_forcing": []}
    best, best_state, best_epoch, bad = float("inf"), None, -1, 0
    for epoch in range(1, params["max_epochs"] + 1):
        t0 = time.time(); model.train()
        tf = max(0.0, 0.5 * (1 - (epoch - 1) / max(0.6 * params["max_epochs"], 1)))
        order = rng.permutation(len(train_origins)); running = 0.0; nb = 0
        for i in range(0, len(order), params["batch_size"]):
            o = train_origins[order[i:i + params["batch_size"]]]
            x, anchor, y, valid = data.batch(o, device)
            opt.zero_grad(set_to_none=True)
            loss = masked_mae_loss(model(x, anchor, decoder_features(data, o, device), y, valid, tf), y, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), params["grad_clip"]); opt.step()
            running += loss.item(); nb += 1
        val = evaluate_mae_mph(model, data, val_origins, device); sched.step(val)
        hist["epoch"].append(epoch); hist["train_loss_scaled"].append(running / max(nb, 1)); hist["val_mae_mph"].append(val)
        hist["lr"].append(opt.param_groups[0]["lr"]); hist["seconds"].append(time.time() - t0); hist["teacher_forcing"].append(tf)
        improved = val < best - 1e-4
        log(f"epoch {epoch:3d} | train MAE(scaled) {hist['train_loss_scaled'][-1]:.4f} | val MAE {val:.4f} mph | tf {tf:.2f} | {hist['seconds'][-1]:.0f}s" + ("  *" if improved else ""))
        if improved:
            best, best_epoch, bad = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model, {"epoch": epoch, "val_mae_mph": val, "params": {k: str(v) for k, v in params.items()}})
        else:
            bad += 1
            if bad >= params["patience"]:
                log(f"early stopping at epoch {epoch} (best epoch {best_epoch})"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"], hist["best_val_mae_mph"] = best_epoch, best
    return hist
