"""Generate every table (report/tables/*.tex) and figure (report/figures/*.pdf) of the IEEE report.

    python report/make_report_assets.py

Reads only saved artifacts (prediction files, training histories, attention and per-sensor gain),
so the numbers in the report always match notebooks 12-15 and reports/dl_results_summary.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deep_learning.dl import results  # noqa: E402
from src import config, evaluation as ev, preprocessing as pp  # noqa: E402
from src.graph_utils import load_graph_artifacts  # noqa: E402

OUT = Path(__file__).parent
TAB, FIG = OUT / "tables", OUT / "figures"
M = config.METRICS_DIR
SEEDS = (42, 1, 2)
HMIN = (5, 15, 30, 60)
SHORT = {"persistence": "Persistence", "seasonal_avg": "Seasonal avg.", "xgboost": "XGBoost",
         "xgboost_spatial": "XGBoost + neigh.", "dl_gru_s42": "GRU (time only)",
         "dl_st_gat_noedge_s42": "ST-GAT, no edges", "dl_st_gat_s42": "ST-GAT", "dl_dcrnn_s42": "DCRNN"}
COL = {"persistence": "#8c8c8c", "seasonal_avg": "#bdbdbd", "xgboost": "#f0b429", "xgboost_spatial": "#d9822b",
       "dl_gru_s42": "#5b8fd6", "dl_st_gat_noedge_s42": "#9b8ccf", "dl_st_gat_s42": "#1f9d63", "dl_dcrnn_s42": "#c0392b"}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COLW, FULLW = 3.5, 7.16       # IEEE column and page widths in inches


def bold_min(values: list[float], fmt: str) -> list[str]:
    best = min(values)
    return [f"\\textbf{{{fmt.format(v)}}}" if abs(v - best) < 1e-9 else fmt.format(v) for v in values]


def write(name: str, body: str) -> None:
    (TAB / f"{name}.tex").write_text(body.strip() + "\n")


def main() -> None:
    TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    ctx = pp.load_context()
    graph = load_graph_artifacts(expected_sensor_ids=ctx.sensor_ids)
    lat, lon = graph.locations["latitude"].to_numpy(), graph.locations["longitude"].to_numpy()
    main_keys = list(results.MODELS)
    seed_keys = [f"dl_{m}_s{s}" for m in ("gru", "st_gat_noedge", "st_gat") for s in (1, 2)]
    preds, y, ymask, origins, horizons, missing = ev.load_aligned(main_keys + seed_keys, "test", ctx.speed, ctx.mask)
    assert not missing, missing
    valid = ev.common_valid_mask(preds, ymask)
    t = ev.metrics_by_horizon(preds, y, valid, horizons)
    piv = {m: t.pivot(index="model", columns="horizon_min", values=m) for m in ("MAE", "RMSE", "MAPE")}

    # ------------------------------------------------------------------ Table: main results
    rows = []
    for metric, fmt in (("MAE", "{:.2f}"), ("RMSE", "{:.2f}"), ("MAPE", "{:.1f}")):
        rows.append(f"\\midrule\n\\multicolumn{{5}}{{l}}{{\\textit{{{metric}{' (\\%)' if metric == 'MAPE' else ' (mph)'}}}}}\\\\")
        cols = {h: bold_min([piv[metric].loc[k, h] for k in main_keys], fmt) for h in HMIN}
        for i, k in enumerate(main_keys):
            rows.append(f"{SHORT[k]} & " + " & ".join(cols[h][i] for h in HMIN) + r" \\")
    write("tab_main", r"""
\begin{table}[t]
\centering
\caption{Test error by forecast horizon (24 test days, 6{,}901 windows, common mask; deep-learning rows: seed 42). Best per column in bold.}
\label{tab:main}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
Model & 5 min & 15 min & 30 min & 60 min \\
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}""")

    # ------------------------------------------------------------------ Table: seeds
    srows = []
    for m in ("gru", "st_gat_noedge", "st_gat"):
        sub = piv["MAE"].loc[[f"dl_{m}_s{s}" for s in SEEDS]]
        srows.append(f"{SHORT[f'dl_{m}_s42']} & " + " & ".join(f"{sub[h].mean():.3f} $\\pm$ {sub[h].std():.3f}" for h in HMIN) + r" \\")
    write("tab_seeds", r"""
\begin{table*}[t]
\centering
\caption{Seed stability: test MAE (mph), mean $\pm$ standard deviation over seeds 42, 1 and 2.}
\label{tab:seeds}
\begin{tabular}{@{}lcccc@{}}
\toprule
Model & 5 min & 15 min & 30 min & 60 min \\
\midrule
""" + "\n".join(srows) + r"""
\bottomrule
\end{tabular}
\end{table*}""")

    # ------------------------------------------------------------------ Bootstrap
    tt = ev.target_timestamps(ctx.timestamps, origins, horizons)
    day = pd.factorize(pd.to_datetime(tt.ravel()).normalize())[0].reshape(tt.shape)
    peak = ev.peak_flags_sh(ctx.timestamps, origins, horizons)

    def boot(ref, new, j, extra=None):
        v = valid[:, j] if extra is None else valid[:, j] & extra[:, j, None]
        return ev.day_block_bootstrap_improvement(preds[ref][:, j], preds[new][:, j], y[:, j], v, day[:, j])

    pairs = [("dl_st_gat_noedge_s42", "dl_st_gat_s42", "ST-GAT vs.\\ no-edge twin"),
             ("xgboost_spatial", "dl_st_gat_s42", "ST-GAT vs.\\ XGBoost + neigh."),
             ("xgboost_spatial", "dl_dcrnn_s42", "DCRNN vs.\\ XGBoost + neigh."),
             ("dl_gru_s42", "dl_st_gat_noedge_s42", "No-edge twin vs.\\ GRU"),
             ("dl_dcrnn_s42", "dl_st_gat_s42", "ST-GAT vs.\\ DCRNN")]
    B = {(r, n, h): boot(r, n, j) for r, n, _ in pairs for j, h in enumerate(HMIN)}
    brows = []
    for r, n, lab in pairs:
        cells = []
        for h in HMIN:
            b = B[(r, n, h)]
            cells.append(f"{b['improvement_pct']:+.1f} [{b['ci_low']:+.1f}, {b['ci_high']:+.1f}]")
        brows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    write("tab_boot", r"""
\begin{table*}[t]
\centering
\caption{Reduction of test MAE (\%) of the first model relative to the second, with 95\% day-block bootstrap intervals (2{,}000 resamples of the 24 test days). Positive means the first model is better; an interval containing 0 is not a reliable difference.}
\label{tab:boot}
\begin{tabular}{@{}lcccc@{}}
\toprule
Comparison & 5 min & 15 min & 30 min & 60 min \\
\midrule
""" + "\n".join(brows) + r"""
\bottomrule
\end{tabular}
\end{table*}""")

    # ------------------------------------------------------------------ Peak
    g = ev.metrics_by_group({k: preds[k] for k in main_keys}, y, valid, horizons, {"peak": peak, "off-peak": ~peak})
    gp = g.pivot_table(index="model", columns=["group", "horizon_min"], values="MAE")
    prow = []
    for k in ("persistence", "xgboost", "xgboost_spatial", "dl_gru_s42", "dl_st_gat_noedge_s42", "dl_st_gat_s42", "dl_dcrnn_s42"):
        prow.append(f"{SHORT[k]} & " + " & ".join(f"{gp.loc[k, (grp, h)]:.2f}" for grp in ("off-peak", "peak") for h in (15, 60)) + r" \\")
    share = (valid & peak[:, :, None]).sum() / valid.sum()
    write("tab_peak", r"""
\begin{table}[t]
\centering
\caption{Test MAE (mph) off-peak and in the peak (weekdays 07--10 and 15--19 at the target time; """ + f"{share:.0%}".replace("%", r"\%") + r""" of scored targets).}
\label{tab:peak}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{Off-peak} & \multicolumn{2}{c}{Peak} \\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Model & 15 min & 60 min & 15 min & 60 min \\
\midrule
""" + "\n".join(prow) + r"""
\bottomrule
\end{tabular}
\end{table}""")

    # ------------------------------------------------------------------ Search grid
    def summ(run):
        return json.loads((M / f"{run}_summary.json").read_text())
    lrs = ("0.0003", "0.001", "0.003")
    grows = []
    for m, lab in (("st_gat", "ST-GAT"), ("st_gat_noedge", "No-edge twin")):
        for w in (64, 128):
            vals = [summ(f"dl_{m}_s42_lr{lr}_w{w}")["best_val_mae_mph"] for lr in lrs]
            grows.append(f"{lab} & {w} & " + " & ".join(f"{v:.3f}" for v in vals) + r" \\")
    write("tab_search", r"""
\begin{table}[t]
\centering
\caption{Hyper-parameter search on the validation period (MAE in mph, mean over 12 steps; 25 epochs, seed 42).}
\label{tab:search}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
 & & \multicolumn{3}{c}{Learning rate} \\
\cmidrule(l){3-5}
Model & Width & $3\cdot10^{-4}$ & $10^{-3}$ & $3\cdot10^{-3}$ \\
\midrule
""" + "\n".join(grows) + r"""
\bottomrule
\end{tabular}
\end{table}""")

    # ------------------------------------------------------------------ Model / training table
    mrows = []
    for m, lab in (("gru", "GRU (time only)"), ("st_gat_noedge", "ST-GAT, no edges"), ("st_gat", "ST-GAT"), ("dcrnn", "DCRNN")):
        runs = [f"dl_{m}_s{s}" for s in (SEEDS if m != "dcrnn" else (42,))]
        sm = [summ(r) for r in runs]
        hist = [pd.read_csv(M / f"{r}_history.csv") for r in runs]
        val = np.array([s["best_val_mae_mph"] for s in sm])
        mrows.append(f"{lab} & {sm[0]['parameters'] / 1e3:.0f}k & {np.mean([h.seconds.mean() for h in hist]):.0f} & "
                     f"{np.mean([s['best_epoch'] for s in sm]):.0f} & {val.mean():.3f} \\\\")
    write("tab_models", r"""
\begin{table}[t]
\centering
\caption{Model size and training on one NVIDIA T4 (Kaggle). Epoch time and best epoch averaged over seeds; validation MAE is the mean over 12 steps.}
\label{tab:models}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
Model & Params & s/epoch & Best epoch & Val.\ MAE \\
\midrule
""" + "\n".join(mrows) + r"""
\bottomrule
\end{tabular}
\end{table}""")

    # ------------------------------------------------------------------ Literature
    lit = [("Historical average", 4.16, 4.16, 4.16), ("FC-LSTM~\\cite{sutskever2014}", 3.44, 3.77, 4.37),
           ("STGCN~\\cite{yu2018stgcn}", 2.88, 3.47, 4.59), ("DCRNN~\\cite{li2018dcrnn}", 2.77, 3.15, 3.60),
           ("Graph WaveNet~\\cite{wu2019graphwavenet}", 2.69, 3.07, 3.53), ("GMAN~\\cite{zheng2020gman}", 2.80, 3.12, 3.44)]
    lrows = [f"{n} & {a:.2f} & {b:.2f} & {c:.2f} \\\\" for n, a, b, c in lit]
    lrows.append(r"\midrule")
    for k in ("seasonal_avg", "xgboost_spatial", "dl_dcrnn_s42", "dl_st_gat_s42"):
        lrows.append(f"Ours: {SHORT[k]} & " + " & ".join(f"{piv['MAE'].loc[k, h]:.2f}" for h in (15, 30, 60)) + r" \\")
    write("tab_lit", r"""
\begin{table}[t]
\centering
\caption{Published METR-LA test MAE (mph) as collected in~\cite{wu2019graphwavenet,zheng2020gman}, and ours. Splits, masks and test days differ, so this is a sanity check, not a ranking.}
\label{tab:lit}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lccc@{}}
\toprule
Model & 15 min & 30 min & 60 min \\
\midrule
""" + "\n".join(lrows) + r"""
\bottomrule
\end{tabular}
\end{table}""")

    # ================================================================== Figures
    # 1. MAE by horizon
    fig, ax = plt.subplots(figsize=(COLW, 2.3))
    for k in ("persistence", "xgboost", "xgboost_spatial", "dl_gru_s42", "dl_st_gat_noedge_s42", "dl_dcrnn_s42", "dl_st_gat_s42"):
        ls = "--" if k in ("persistence", "xgboost", "dl_st_gat_noedge_s42") else "-"
        ax.plot(HMIN, [piv["MAE"].loc[k, h] for h in HMIN], ls, marker="o", ms=3, lw=1.6 if k == "dl_st_gat_s42" else 1.1,
                c=COL[k], label=SHORT[k])
    ax.set(xlabel="Forecast horizon (min)", ylabel="Test MAE (mph)", xticks=HMIN)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    fig.savefig(FIG / "fig_horizon.pdf"); plt.close(fig)

    # 2. Ablation + vs XGBoost bars with CIs
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 1.9), sharey=False)
    x = np.arange(4)
    for ax, (r, n, lab), col in zip(axes, pairs[:2], ("#1f9d63", "#d9822b")):
        b = [B[(r, n, h)] for h in HMIN]
        v = [q["improvement_pct"] for q in b]
        err = [[q["improvement_pct"] - q["ci_low"] for q in b], [q["ci_high"] - q["improvement_pct"] for q in b]]
        ax.bar(x, v, color=col, width=.6, yerr=err, capsize=3, error_kw=dict(lw=.8))
        for xi, vi, q in zip(x, v, b):
            ax.text(xi + .33, vi, f"{vi:.1f}", ha="left", va="center", fontsize=6.5)
        ax.axhline(0, c="k", lw=.6)
        ax.set(xticks=x, xticklabels=[f"{h} min" for h in HMIN], ylabel="MAE reduction (%)", title=lab.replace("\\ ", " "))
    fig.savefig(FIG / "fig_ablation.pdf"); plt.close(fig)

    # 3. Validation curves
    fig, ax = plt.subplots(figsize=(COLW, 2.1))
    for m, c in (("gru", COL["dl_gru_s42"]), ("st_gat_noedge", COL["dl_st_gat_noedge_s42"]), ("st_gat", COL["dl_st_gat_s42"]), ("dcrnn", COL["dl_dcrnn_s42"])):
        for s in (SEEDS if m != "dcrnn" else (42,)):
            h = pd.read_csv(M / f"dl_{m}_s{s}_history.csv")
            ax.plot(h.epoch, h.val_mae_mph, c=c, lw=.9, alpha=.85, label=SHORT[f"dl_{m}_s42"] if s == 42 else None)
    ax.set(xlabel="Epoch", ylabel="Validation MAE (mph)", ylim=(2.7, 3.3))
    ax.legend(frameon=False, ncol=2)
    fig.savefig(FIG / "fig_curves.pdf"); plt.close(fig)

    # 4. Per-sensor graph gain map
    gain = pd.read_csv(M / "dl_sensor_gain_60min.csv", index_col="node").reindex(range(len(lat)))
    off = (graph.adjacency > 0) & ~np.eye(len(lat), dtype=bool)
    src, dst = np.nonzero(off)
    seg = np.stack([np.c_[lon[src], lat[src]], np.c_[lon[dst], lat[dst]]], axis=1)
    fig, ax = plt.subplots(figsize=(COLW, 2.4))
    ax.add_collection(LineCollection(seg, colors="#cfcfcf", linewidths=.4, zorder=1))
    gv = gain.graph_gain.to_numpy()
    lim = np.nanpercentile(np.abs(gv), 95)
    sc = ax.scatter(lon, lat, c=gv, cmap="RdBu", vmin=-lim, vmax=lim, s=9, edgecolors="k", linewidths=.15, zorder=2)
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean()))); ax.set_xticks([]); ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.colorbar(sc, ax=ax, shrink=.8, label="MAE gain at 60 min (mph)")
    fig.savefig(FIG / "fig_gain_map.pdf"); plt.close(fig)

    # 5. Snapshot: slowest weekday-afternoon moment of the test period (rule on observed speeds only)
    speed = np.where(ctx.mask, ctx.speed, np.nan)
    rows = np.arange(origins.min() + 12, origins.max() + 13)
    fr = pd.DataFrame({"t": ctx.timestamps[rows], "m": pd.DataFrame(speed[rows]).mean(axis=1).to_numpy(),
                       "obs": ctx.mask[rows].mean(1)}, index=rows)
    pm = fr[(fr.t.dt.dayofweek < 5) & fr.t.dt.hour.between(15, 18) & (fr.obs > .9)]
    worst = pm.groupby(pm.t.dt.date).m.mean().idxmin()
    target = int(pm[pm.t.dt.date == worst].m.idxmin())
    s = int(np.flatnonzero(origins == target - 12)[0])
    j60 = list(horizons).index(12)
    panels = [("Observed " + f"{ctx.timestamps[target]:%H:%M}", np.where(ctx.mask[target], ctx.speed[target], np.nan)),
              ("Persistence", preds["persistence"][s, j60]), ("XGBoost + neigh.", preds["xgboost_spatial"][s, j60]),
              ("ST-GAT", preds["dl_st_gat_s42"][s, j60])]
    fig, axes = plt.subplots(1, 4, figsize=(FULLW, 1.75))
    for ax, (lab, v) in zip(axes, panels):
        ax.add_collection(LineCollection(seg, colors="#d5d5d5", linewidths=.35, zorder=1))
        ok = np.isfinite(v)
        ax.scatter(lon[~ok], lat[~ok], s=4, c="#bbbbbb", zorder=2)
        sc = ax.scatter(lon[ok], lat[ok], c=v[ok], cmap="RdYlGn", vmin=0, vmax=70, s=7, edgecolors="k", linewidths=.1, zorder=3)
        ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean()))); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
    fig.colorbar(sc, ax=axes, shrink=.8, label="Speed (mph)")
    fig.savefig(FIG / "fig_snapshot.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ numbers quoted in the text
    gs = pd.read_csv(M / "dl_sensor_gain_60min.csv")
    facts = {
        "snapshot": f"{pd.Timestamp(worst):%A %d %B %Y}, {ctx.timestamps[target]:%H:%M}",
        "graph_helps_pct": float((gs.graph_gain > 0).mean()), "beats_xgb_pct": float((gs.vs_xgb_spatial > 0).mean()),
        "gain_indeg_corr": float(gs[["graph_gain", "in_degree"]].corr().iloc[0, 1]),
        "peak_share": float(share), "scored_share": float(valid.mean()),
        "boot60": {lab: B[(r, n, 60)] for r, n, lab in pairs},
        "peak60": {grp: boot("xgboost_spatial", "dl_st_gat_s42", 3, peak if grp == "peak" else ~peak)["improvement_pct"] for grp in ("peak", "off-peak")},
    }
    (OUT / "report_facts.json").write_text(json.dumps(facts, indent=1, default=float))
    print("wrote", sorted(p.name for p in TAB.iterdir()), sorted(p.name for p in FIG.iterdir()))
    print(json.dumps(facts, indent=1, default=float))


if __name__ == "__main__":
    main()
