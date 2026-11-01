"""Plotting helpers shared by the notebooks.

Colour conventions (documented once, used everywhere)
-----------------------------------------------------
* Every model keeps one fixed colour/marker in every figure (``MODEL_STYLE``).
* **Speed maps** use one sequential ramp, dark navy = slow / congested, pale = free flow,
  on a fixed 0-70 mph scale so that maps of different timestamps, horizons and models
  are directly comparable.  *Missing* readings are drawn as grey crosses, never as a speed.
* **Error maps** use a diverging blue-grey-red scale centred on 0
  (red = over-predicted speed, blue = under-predicted speed).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (registers mpl.ticker)
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from . import config

SURFACE = "#fcfcfb"
INK, INK_2, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

SPEED_VMIN, SPEED_VMAX = 0.0, 70.0
SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed_seq", ["#0d366b", "#1c5cab", "#3987e5", "#9ec5f4", "#e6f0fc"]
)
ERROR_CMAP = LinearSegmentedColormap.from_list(
    "error_div", ["#184f95", "#5598e7", "#f0efec", "#ec835a", "#b02a2a"]
)
MISSING_COLOR = "#898781"

# fixed colour + marker + linestyle per model (colour follows the entity)
MODEL_STYLE: dict[str, dict] = {
    "persistence":  dict(label="Persistence",          color="#898781", marker="s", ls="--"),
    "rolling_mean": dict(label="Rolling mean (60 min)", color="#c98500", marker="v", ls="--"),
    "seasonal_avg": dict(label="Seasonal average",      color="#d55181", marker="^", ls="--"),
    "xgboost":      dict(label="XGBoost",               color="#2a78d6", marker="o", ls="-"),
    "xgboost_spatial": dict(label="XGBoost + neighbour features", color="#6250d6", marker="X", ls="-."),
    "random_forest": dict(label="Random Forest (multi-output)", color="#7a8f1f", marker="h", ls="-"),
    "xgboost_tuned": dict(label="XGBoost (tuned)",      color="#0b4f9c", marker="*", ls="-"),
    "gat_nograph":  dict(label="ST-GAT without edges (ablation)", color="#1baf7a", marker="D", ls="-"),
    "gat":          dict(label="ST-GAT",                color="#eb6834", marker="o", ls="-"),
    "dcrnn":        dict(label="DCRNN (optional)",      color="#e34948", marker="P", ls="-"),
}
HORIZON_LABEL = {h: f"{h * config.STEP_MINUTES} min" for h in range(1, 13)}


def style(model: str) -> dict:
    return MODEL_STYLE.get(model, dict(label=model, color=INK_2, marker="o", ls="-"))


def apply_style() -> None:
    """Quiet, readable defaults: thin marks, recessive grid, no top/right spines."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 1.8, "lines.markersize": 6, "legend.frameon": False,
        "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10,
    })


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save under ``artifacts/figures`` (smoke runs get a ``_smoke`` suffix)."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{config.tagged(name)}.png"
    fig.savefig(path)
    return path


# --------------------------------------------------------------------------- #
# Maps (longitude/latitude scatter - historical sensor readings, not live traffic)
# --------------------------------------------------------------------------- #
def _lonlat(loc: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return loc["longitude"].to_numpy(), loc["latitude"].to_numpy()


def _equalise(ax: plt.Axes, loc: pd.DataFrame) -> None:
    ax.set_aspect(1.0 / np.cos(np.deg2rad(loc["latitude"].mean())))
    ax.set_xlabel("Longitude (deg)"); ax.set_ylabel("Latitude (deg)")


def plot_sensor_locations(ax: plt.Axes, loc: pd.DataFrame, title: str = "METR-LA sensor locations") -> None:
    lon, lat = _lonlat(loc)
    ax.scatter(lon, lat, s=22, c="#2a78d6", edgecolor=SURFACE, linewidth=0.6, zorder=3)
    ax.set_title(title); _equalise(ax, loc)


def plot_graph_on_map(
    ax: plt.Axes, loc: pd.DataFrame, adj: np.ndarray, max_edges: int | None = None,
    seed: int = config.SEED, title: str = "Road graph edges (subset)",
) -> int:
    """Draw directed-graph edges as straight segments between sensor coordinates.

    The straight segment is only a drawing aid: it is NOT the road geometry.  ``max_edges``
    keeps the strongest edges when the full graph would be unreadable.
    """
    lon, lat = _lonlat(loc)
    work = np.asarray(adj).copy()
    np.fill_diagonal(work, 0)
    src, dst = np.nonzero(work > 0)
    if max_edges is not None and len(src) > max_edges:
        keep = np.argsort(-work[src, dst])[:max_edges]
        src, dst = src[keep], dst[keep]
    segments = np.stack([np.stack([lon[src], lat[src]], 1), np.stack([lon[dst], lat[dst]], 1)], 1)
    ax.add_collection(LineCollection(segments, colors="#6da7ec", linewidths=0.7, alpha=0.55, zorder=2))
    ax.scatter(lon, lat, s=14, c=INK_2, edgecolor=SURFACE, linewidth=0.4, zorder=3)
    ax.set_xlim(lon.min() - 0.01, lon.max() + 0.01); ax.set_ylim(lat.min() - 0.01, lat.max() + 0.01)
    ax.set_title(f"{title}\n{len(src)} directed edges"); _equalise(ax, loc)
    return int(len(src))


def plot_speed_map(
    ax: plt.Axes, loc: pd.DataFrame, speed: np.ndarray, title: str, observed: np.ndarray | None = None,
    cbar: bool = True, s: float = 34,
) -> mpl.collections.PathCollection:
    """Sensors coloured by speed on the fixed scale; missing sensors are grey crosses."""
    lon, lat = _lonlat(loc)
    ok = np.isfinite(speed) if observed is None else (observed & np.isfinite(speed))
    sc = ax.scatter(lon[ok], lat[ok], c=speed[ok], cmap=SPEED_CMAP, vmin=SPEED_VMIN, vmax=SPEED_VMAX,
                    s=s, edgecolor=INK_2, linewidth=0.4, zorder=3)
    if (~ok).any():
        ax.scatter(lon[~ok], lat[~ok], marker="x", c=MISSING_COLOR, s=s * 0.8, linewidth=1.0,
                   zorder=3, label=f"missing ({int((~ok).sum())})")
        ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title); _equalise(ax, loc)
    if cbar:
        plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, label="speed (mph)")
    return sc


def plot_error_map(
    ax: plt.Axes, loc: pd.DataFrame, error: np.ndarray, title: str, limit: float = 15.0,
    cbar: bool = True, s: float = 34,
) -> None:
    """Signed error (predicted - actual, mph) on a diverging scale; NaN (no truth) = grey cross."""
    lon, lat = _lonlat(loc)
    ok = np.isfinite(error)
    sc = ax.scatter(lon[ok], lat[ok], c=error[ok], cmap=ERROR_CMAP, vmin=-limit, vmax=limit, s=s,
                    edgecolor=INK_2, linewidth=0.4, zorder=3)
    if (~ok).any():
        ax.scatter(lon[~ok], lat[~ok], marker="x", c=MISSING_COLOR, s=s * 0.8, linewidth=1.0, zorder=3,
                   label=f"no ground truth ({int((~ok).sum())})")
        ax.legend(loc="lower left", fontsize=8)
    ax.set_title(title); _equalise(ax, loc)
    if cbar:
        plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02, label="predicted - actual (mph)")


# --------------------------------------------------------------------------- #
# Graph statistics plots
# --------------------------------------------------------------------------- #
def plot_degree_distribution(ax: plt.Axes, in_deg: np.ndarray, out_deg: np.ndarray) -> None:
    top = int(max(in_deg.max(), out_deg.max())) + 2
    bins = np.arange(0, top + 1) - 0.5
    ax.hist([in_deg, out_deg], bins=bins, label=["in-degree", "out-degree"],
            color=["#2a78d6", "#eb6834"], edgecolor=SURFACE, linewidth=1.0)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("degree (neighbours, excluding self)"); ax.set_ylabel("number of sensors")
    ax.set_title("Degree distribution of the road graph"); ax.legend()


def plot_adjacency_heatmap(ax: plt.Axes, adj: np.ndarray, title: str = "Weighted adjacency matrix") -> None:
    im = ax.imshow(adj, cmap=LinearSegmentedColormap.from_list("w", ["#fcfcfb", "#9ec5f4", "#1c5cab", "#0d366b"]),
                   vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    ax.grid(False)
    ax.set_xlabel("target sensor index j"); ax.set_ylabel("source sensor index i")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8, label="edge weight A[i, j]")


# --------------------------------------------------------------------------- #
# Metric plots
# --------------------------------------------------------------------------- #
def plot_metric_vs_horizon(
    ax: plt.Axes, table: pd.DataFrame, metric: str, models: Sequence[str] | None = None,
    ylabel: str | None = None, title: str | None = None,
) -> None:
    """Line per model: ``metric`` against forecast horizon (minutes)."""
    models = list(models) if models is not None else list(table["model"].unique())
    for m in models:
        sub = table[table.model == m].sort_values("horizon_min")
        st = style(m)
        ax.plot(sub["horizon_min"], sub[metric], color=st["color"], marker=st["marker"], ls=st["ls"],
                label=st["label"])
    ax.set_xlabel("forecast horizon (minutes)")
    ax.set_ylabel(ylabel or ("MAPE (%)" if metric == "MAPE" else f"{metric} (mph)"))
    ax.set_xticks(sorted(table["horizon_min"].unique()))
    if title:
        ax.set_title(title)
    ax.legend()


def plot_actual_vs_predicted(
    ax: plt.Axes, times: Sequence, actual: np.ndarray, preds: Mapping[str, np.ndarray], title: str,
) -> None:
    """Observed speed (black) against model predictions; gaps are real missing observations."""
    ax.plot(times, actual, color=INK, lw=1.6, label="Observed", zorder=5)
    for m, p in preds.items():
        st = style(m)
        ax.plot(times, p, color=st["color"], ls=st["ls"], lw=1.4, label=st["label"], alpha=0.95)
    ax.set_ylabel("speed (mph)"); ax.set_title(title)
    ax.legend(ncol=3, loc="lower left", frameon=True, framealpha=0.85, facecolor=SURFACE, edgecolor="none")
