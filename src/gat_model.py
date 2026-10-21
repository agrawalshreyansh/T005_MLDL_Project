"""Spatio-temporal Graph Attention Network (ST-GAT) for multi-step speed forecasting.

Information flow (B = batch, T = 12 input steps, N = 207 sensors, F = 6 features)::

    x [B, T, N, F] --Linear--> [B, T, N, d] --GRU over T (shared by all sensors)--> h [B, N, d]
    h  ++ learned sensor embedding --Linear--> h0 [B, N, d]
    h0 --(LayerNorm -> GAT_in + GAT_out -> ELU -> dropout) + residual--> x 2 blocks--> g [B, N, d]
    [g, h0] --MLP--> residual forecast [B, N, 12]
    prediction = anchor + residual          (anchor = latest observed speed in the window)

The graph attention is *dense but masked*: node ``i`` may only attend to nodes ``j`` for
which the validated road graph has an edge ``j -> i`` (``GAT_in``) or ``i -> j``
(``GAT_out``), plus itself.  Scores of all other pairs are set to ``-inf`` before the
softmax, so they receive exactly zero attention.  For N = 207 this is as cheap as a sparse
implementation, runs unchanged on CPU / CUDA / Apple MPS (no scatter ops) and is verified
numerically against ``torch_geometric.nn.GATConv`` in ``tests/test_gat_model.py``.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .graph_utils import neighbor_masks
from .preprocessing import (
    SpeedScaler, gather_model_inputs, gather_targets, window_anchor,
)


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #
def select_device(prefer: str | None = None) -> torch.device:
    """CUDA if available, else Apple MPS, else CPU (override with ``prefer``)."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Graph attention layer
# --------------------------------------------------------------------------- #
class DenseGATLayer(nn.Module):
    """Multi-head graph attention (Velickovic et al., 2018) restricted by a boolean mask.

    For target node ``i`` and source node ``j`` allowed by ``mask[i, j]``::

        e_ij   = LeakyReLU(a_dst . W x_i + a_src . W x_j)
        alpha  = softmax_j(e_ij)                 (over allowed j only)
        out_i  = sum_j alpha_ij W x_j + bias     (heads concatenated)
    """

    def __init__(self, in_dim: int, out_per_head: int, heads: int, attn_dropout: float = 0.0,
                 negative_slope: float = 0.2) -> None:
        super().__init__()
        self.heads, self.out_per_head = heads, out_per_head
        self.lin = nn.Linear(in_dim, heads * out_per_head, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_per_head))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_per_head))
        self.bias = nn.Parameter(torch.zeros(heads * out_per_head))
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.negative_slope = negative_slope
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attention: bool = False):
        """``x [B, N, D]``, ``mask [N, N]`` (True = i may attend to j). Returns ``[B, N, H*C]``."""
        b, n, _ = x.shape
        h = self.lin(x).view(b, n, self.heads, self.out_per_head)            # [B, N, H, C]
        a_src = (h * self.att_src).sum(-1)                                   # [B, N, H] source j
        a_dst = (h * self.att_dst).sum(-1)                                   # [B, N, H] target i
        e = F.leaky_relu(a_dst[:, :, None, :] + a_src[:, None, :, :], self.negative_slope)  # [B, i, j, H]
        e = e.masked_fill(~mask[None, :, :, None], float("-inf"))
        alpha = torch.softmax(e, dim=2)                                      # normalise over sources j
        alpha_d = self.attn_dropout(alpha)
        out = torch.einsum("bijh,bjhc->bihc", alpha_d, h).reshape(b, n, -1) + self.bias
        return (out, alpha) if return_attention else out


class GATBlock(nn.Module):
    """Pre-norm residual block combining attention over several neighbourhood masks."""

    def __init__(self, d_model: int, heads: int, masks: list[torch.Tensor], dropout: float,
                 attn_dropout: float) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by the number of heads")
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList(
            [DenseGATLayer(d_model, d_model // heads, heads, attn_dropout) for _ in masks]
        )
        for i, m in enumerate(masks):
            self.register_buffer(f"mask_{i}", m, persistent=False)
        self.n_masks = len(masks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        z = self.norm(x)
        total, attentions = 0.0, []
        for i, layer in enumerate(self.layers):
            mask = getattr(self, f"mask_{i}")
            if return_attention:
                out, alpha = layer(z, mask, return_attention=True)
                attentions.append(alpha)
            else:
                out = layer(z, mask)
            total = total + out
        y = x + self.dropout(F.elu(total))
        return (y, attentions) if return_attention else y


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class STGAT(nn.Module):
    """GRU temporal encoder -> stacked graph-attention blocks -> multi-horizon head."""

    def __init__(self, n_nodes: int, in_features: int, adj: np.ndarray, d_model: int = 64,
                 n_heads: int = 4, n_gat_layers: int = 2, node_emb_dim: int = 16,
                 dropout: float = 0.1, attn_dropout: float = 0.1, n_steps: int = config.MAX_HORIZON,
                 use_graph: bool = True) -> None:
        super().__init__()
        self.n_nodes, self.n_steps, self.use_graph = n_nodes, n_steps, use_graph
        masks_np = neighbor_masks(adj)
        # use_graph=False keeps the parameter count identical but removes every edge
        # (each node only "attends" to itself), isolating the contribution of the graph.
        names = ("in", "out") if use_graph else ("self", "self")
        masks = [torch.from_numpy(masks_np[k]) for k in names]
        assert all(bool(m.diagonal().all()) for m in masks), "every node needs a self-loop"

        self.input_proj = nn.Linear(in_features, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.node_emb = nn.Embedding(n_nodes, node_emb_dim)
        self.fuse = nn.Linear(d_model + node_emb_dim, d_model)
        self.blocks = nn.ModuleList(
            [GATBlock(d_model, n_heads, masks, dropout, attn_dropout) for _ in range(n_gat_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, n_steps),
        )

    def forward(self, x: torch.Tensor, anchor: torch.Tensor, return_attention: bool = False):
        """``x [B, T, N, F]``, ``anchor [B, N]`` (scaled) -> prediction ``[B, n_steps, N]`` (scaled)."""
        b, t, n, _ = x.shape
        z = self.input_proj(x)                                         # [B, T, N, d]
        z = z.permute(0, 2, 1, 3).reshape(b * n, t, -1)                # one sequence per (batch, sensor)
        _, hidden = self.gru(z)                                        # [1, B*N, d]
        h = hidden[-1].reshape(b, n, -1)                               # [B, N, d]
        emb = self.node_emb.weight.unsqueeze(0).expand(b, -1, -1)      # [B, N, e]
        h0 = self.fuse(torch.cat([h, emb], dim=-1))                    # [B, N, d]
        g, attentions = h0, []
        for block in self.blocks:
            if return_attention:
                g, att = block(g, return_attention=True)
                attentions.append(att)
            else:
                g = block(g)
        residual = self.head(torch.cat([g, h0], dim=-1))               # [B, N, n_steps]
        pred = anchor.unsqueeze(1) + residual.permute(0, 2, 1)         # [B, n_steps, N]
        return (pred, attentions) if return_attention else pred


def build_model(adj: np.ndarray, params: dict | None = None, use_graph: bool = True,
                in_features: int = 6) -> STGAT:
    p = dict(config.GAT_PARAMS if params is None else params)
    return STGAT(
        n_nodes=adj.shape[0], in_features=in_features, adj=adj, d_model=p["d_model"],
        n_heads=p["n_heads"], n_gat_layers=p["n_gat_layers"], node_emb_dim=p["node_emb_dim"],
        dropout=p["dropout"], attn_dropout=p["attn_dropout"], use_graph=use_graph,
    )


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
@dataclass
class GraphData:
    """Everything needed to build batches (kept on the CPU, moved per batch)."""

    speed: np.ndarray            # [T, N] mph, NaN missing
    mask: np.ndarray             # [T, N] bool
    values_scaled: np.ndarray    # [T, N] scaled speed, 0 where missing
    time_feats: np.ndarray       # [T, 4]
    locf_value: np.ndarray       # [T, N]
    locf_age: np.ndarray         # [T, N]
    scaler: SpeedScaler
    input_len: int = config.INPUT_LEN
    n_steps: int = config.MAX_HORIZON

    def batch(self, origins: np.ndarray, device: torch.device):
        """``x [B,T,N,F]``, ``anchor [B,N]``, ``y [B,H,N]``, ``y_valid [B,H,N]`` (all scaled/float)."""
        x = gather_model_inputs(self.values_scaled, self.mask, self.time_feats, origins, self.input_len)
        anchor, _ = window_anchor(self.locf_value, self.locf_age, origins, self.input_len,
                                  self.scaler.sensor_mean)
        y_raw, y_obs = gather_targets(self.speed, self.mask, origins, tuple(range(1, self.n_steps + 1)))
        y = np.where(y_obs, self.scaler.transform(np.nan_to_num(y_raw, nan=0.0)), 0.0).astype(np.float32)
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
        return to(x), to(self.scaler.transform(anchor)), to(y), to(y_obs.astype(np.float32))


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over valid targets only (invalid ones carry zero weight)."""
    denom = valid.sum().clamp(min=1.0)
    return (torch.abs(pred - target) * valid).sum() / denom


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_scaled(model: STGAT, data: GraphData, origins: np.ndarray, device: torch.device,
                   batch_size: int = 256) -> np.ndarray:
    """All ``n_steps`` forecasts in scaled units: ``[S, n_steps, N]``."""
    model.eval()
    out = []
    for i in range(0, len(origins), batch_size):
        x, anchor, _, _ = data.batch(origins[i:i + batch_size], device)
        out.append(model(x, anchor).float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.empty((0, data.n_steps, data.speed.shape[1]), np.float32)


def predict_mph(model: STGAT, data: GraphData, origins: np.ndarray, device: torch.device,
                steps: tuple[int, ...] = config.HORIZONS, batch_size: int = 256) -> np.ndarray:
    """Forecasts in mph at the requested 1-based ``steps``: ``[S, len(steps), N]``."""
    z = predict_scaled(model, data, origins, device, batch_size)
    sel = z[:, [s - 1 for s in steps], :]
    return data.scaler.inverse_transform(sel)


def evaluate_mae_mph(model: STGAT, data: GraphData, origins: np.ndarray, device: torch.device,
                     batch_size: int = 256) -> float:
    """Masked MAE in mph over all forecast steps (the validation/early-stopping criterion)."""
    model.eval()
    total, count = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(origins), batch_size):
            x, anchor, y, valid = data.batch(origins[i:i + batch_size], device)
            pred = model(x, anchor)
            total += float((torch.abs(pred - y) * valid).sum())
            count += float(valid.sum())
    return data.scaler.std * total / max(count, 1.0)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model: STGAT, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path: Path, model: STGAT, device: torch.device) -> dict:
    """Load weights with ``weights_only=True`` (tensors and plain dicts only)."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Checkpoint {path} not found - train the model first.")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    return ckpt["meta"]


def train_model(
    model: STGAT, data: GraphData, train_origins: np.ndarray, val_origins: np.ndarray,
    device: torch.device, params: dict | None = None, checkpoint_path: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Train with AdamW, masked-MAE loss, LR plateau decay, early stopping on validation MAE.

    Returns a history dict; ``model`` is left holding the best-validation weights.
    """
    p = dict(config.GAT_PARAMS if params is None else params)
    set_seed(p["seed"])
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=max(p["patience"] // 3, 1))
    rng = np.random.default_rng(p["seed"])
    history = {"epoch": [], "train_loss_scaled": [], "val_mae_mph": [], "lr": [], "seconds": []}
    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0

    for epoch in range(1, p["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(len(train_origins))
        running, n_batches = 0.0, 0
        for i in range(0, len(order), p["batch_size"]):
            idx = train_origins[order[i:i + p["batch_size"]]]
            x, anchor, y, valid = data.batch(idx, device)
            optimiser.zero_grad(set_to_none=True)
            loss = masked_mae_loss(model(x, anchor), y, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}; check inputs/masks")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
            optimiser.step()
            running += loss.item(); n_batches += 1
        val_mae = evaluate_mae_mph(model, data, val_origins, device)
        scheduler.step(val_mae)
        history["epoch"].append(epoch)
        history["train_loss_scaled"].append(running / max(n_batches, 1))
        history["val_mae_mph"].append(val_mae)
        history["lr"].append(optimiser.param_groups[0]["lr"])
        history["seconds"].append(time.time() - t0)
        improved = val_mae < best_val - 1e-4
        log(f"epoch {epoch:3d} | train MAE(scaled) {history['train_loss_scaled'][-1]:.4f} | "
            f"val MAE {val_mae:.4f} mph | lr {history['lr'][-1]:.1e} | {history['seconds'][-1]:.0f}s"
            + ("  *" if improved else ""))
        if improved:
            best_val, best_epoch, bad = val_mae, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                save_checkpoint(checkpoint_path, model, {"epoch": epoch, "val_mae_mph": val_mae, "params": p})
        else:
            bad += 1
            if bad >= p["patience"]:
                log(f"early stopping at epoch {epoch} (best epoch {best_epoch}, val MAE {best_val:.4f} mph)")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"], history["best_val_mae_mph"] = best_epoch, best_val
    return history
