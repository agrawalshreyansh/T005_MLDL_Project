"""One training loop for every model.

Contract: ``model(batch, teacher_forcing=0.0)`` returns scaled predictions ``[B, H, N]``.
Models that feed their own outputs back in (seq2seq) set ``model.uses_teacher_forcing = True``.

The loop: masked MAE loss -> AdamW -> gradient clipping -> validation MAE in mph after every
epoch -> halve the learning rate when validation stalls -> keep the best weights -> stop when
validation has not improved for ``patience`` epochs.  The test split is never touched here.
"""
from __future__ import annotations

import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src import config
from .data import WindowData

DEFAULTS = dict(lr=1e-3, weight_decay=1e-5, batch_size=64, max_epochs=60, patience=10,
                grad_clip=5.0, lr_patience=3, eval_batch_size=128, tf_decay_frac=0.4, seed=config.SEED)


def select_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def masked_mae(pred: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean |pred - y| over observed targets only; a missing target contributes nothing."""
    return (torch.abs(pred - y) * valid).sum() / valid.sum().clamp(min=1.0)


@torch.no_grad()
def predict_scaled(model: nn.Module, data: WindowData, split: str, batch_size: int = 128) -> torch.Tensor:
    model.eval()
    return torch.cat([model(b).float().cpu() for b in data.batches(split, batch_size)], dim=0)


@torch.no_grad()
def val_mae_mph(model: nn.Module, data: WindowData, split: str = "val", batch_size: int = 128) -> float:
    """Masked MAE in mph over all 12 steps (the early-stopping criterion)."""
    model.eval()
    err = cnt = 0.0
    for b in data.batches(split, batch_size):
        err += float((torch.abs(model(b) - b["y"]) * b["valid"]).sum())
        cnt += float(b["valid"].sum())
    return data.scaler.std * err / max(cnt, 1.0)


def teacher_forcing_ratio(epoch: int, max_epochs: int, decay_frac: float) -> float:
    """Linear 1 -> 0 over the first ``decay_frac`` of the epochs (curriculum for seq2seq decoders)."""
    return max(0.0, 1.0 - (epoch - 1) / max(decay_frac * max_epochs, 1.0))


def train(model: nn.Module, data: WindowData, params: dict | None = None,
          out_dir: Path | None = None, name: str = "model", log=print) -> dict:
    """Train ``model`` in place (it ends holding the best-validation weights); returns the history."""
    p = {**DEFAULTS, **(params or {})}
    set_seed(p["seed"])
    device = data.values.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=p["lr_patience"])
    gen = torch.Generator().manual_seed(p["seed"])
    uses_tf = getattr(model, "uses_teacher_forcing", False)
    hist: list[dict] = []
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    log(f"{name}: {count_parameters(model):,} parameters on {device}")

    for epoch in range(1, p["max_epochs"] + 1):
        t0 = time.time()
        tf = teacher_forcing_ratio(epoch, p["max_epochs"], p["tf_decay_frac"]) if uses_tf else 0.0
        model.train()
        total = n = 0.0
        for b in data.batches("train", p["batch_size"], shuffle=True, generator=gen):
            loss = masked_mae(model(b, teacher_forcing=tf), b["y"], b["valid"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in epoch {epoch}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
            opt.step()
            total += loss.item(); n += 1
        val = val_mae_mph(model, data, "val", p["eval_batch_size"])
        settled = tf == 0.0                    # while teacher forcing is on, validation is noisy
        if settled:
            sched.step(val)
        row = dict(epoch=epoch, train_mae_mph=data.scaler.std * total / max(n, 1), val_mae_mph=val,
                   lr=opt.param_groups[0]["lr"], teacher_forcing=tf, seconds=time.time() - t0)
        hist.append(row)
        improved = val < best - 1e-4
        log(f"{name} | epoch {epoch:3d} | train {row['train_mae_mph']:.3f} | val {val:.3f} mph | "
            f"lr {row['lr']:.1e}{f' | tf {tf:.2f}' if uses_tf else ''} | {row['seconds']:.0f}s" + (" *" if improved else ""))
        if improved:
            best, best_epoch, bad = val, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        elif settled:
            bad += 1
            if bad >= p["patience"]:
                log(f"{name}: early stop at epoch {epoch}, best epoch {best_epoch} ({best:.3f} mph)")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    summary = dict(name=name, best_epoch=best_epoch, best_val_mae_mph=best, epochs_run=len(hist),
                   parameters=count_parameters(model), params=p, device=str(device))
    if out_dir is not None:
        save_run(out_dir, name, model, hist, summary)
    return {"history": hist, **summary}


def save_run(out_dir: Path, name: str, model: nn.Module, hist: list[dict], summary: dict) -> None:
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "summary": summary}, out_dir / "models" / f"{name}.pt")
    with open(out_dir / "metrics" / f"{name}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader(); w.writerows(hist)
    (out_dir / "metrics" / f"{name}_summary.json").write_text(json.dumps(summary, indent=2, default=str))


def load_weights(model: nn.Module, path: Path, device="cpu") -> dict:
    """Load a checkpoint saved by :func:`save_run` (tensors and plain types only)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return ckpt["summary"]
