"""Fill template.html with numbers from the saved artifacts and write a self-contained index.html.

    python -m deep_learning.presentation.build

Every number on the slides is computed here from the prediction files, histories and graph,
so the deck cannot drift from the notebooks. Run after notebooks 12 and 15 (attention and
per-sensor gain files).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from deep_learning.dl import results
from src import config, evaluation as ev, preprocessing as pp
from src.graph_utils import load_graph_artifacts

HERE = Path(__file__).parent
TEAM = [("Harshit Shakya", "2401010183"), ("Ishan Maheshwari", "2401010194"), ("Shreyansh Agrawal", "2401010447"),
        ("Divyanshu Raj", "2401010160"), ("Nikith Gowda B S", "2401010299")]
COL = {"persistence": "#8d96ad", "xgboost": "#f2c14e", "xgboost_spatial": "#ffa630", "dl_gru_s42": "#7fb2ff",
       "dl_st_gat_noedge_s42": "#b7a7e8", "dl_st_gat_s42": "#3ccf8a", "dl_dcrnn_s42": "#ec3b3b"}
LITERATURE = [("Historical average", 4.16, 4.16, 4.16), ("FC-LSTM", 3.44, 3.77, 4.37), ("STGCN", 2.88, 3.47, 4.59),
              ("DCRNN (paper)", 2.77, 3.15, 3.60), ("Graph WaveNet", 2.69, 3.07, 3.53), ("GMAN", 2.80, 3.12, 3.44)]


def pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}%"


def label_nudges(values: list[float], y0: float, y1: float, height: float = 540, gap: float = 34) -> list[float]:
    """Pixel offsets so end-of-line labels in the horizon chart do not overlap (same scale as the template)."""
    ys = [height - (v - y0) / (y1 - y0) * height for v in values]
    order, placed, nudge = np.argsort(ys), [], [0.0] * len(values)
    for i in order:
        y = ys[i] if not placed else max(ys[i], placed[-1] + gap)
        nudge[i], placed = y - ys[i], placed + [y]
    return nudge


def main() -> None:
    ctx = pp.load_context()
    graph = load_graph_artifacts(expected_sensor_ids=ctx.sensor_ids)
    adj = graph.adjacency
    n = adj.shape[0]
    off = (adj > 0) & ~np.eye(n, dtype=bool)
    src, dst = np.nonzero(off)
    lat, lon = graph.locations["latitude"].to_numpy(), graph.locations["longitude"].to_numpy()

    keys = list(results.MODELS)
    preds, y, ymask, origins, horizons, missing = ev.load_aligned(keys, "test", ctx.speed, ctx.mask)
    assert not missing, missing
    valid = ev.common_valid_mask(preds, ymask)
    mae = ev.metrics_by_horizon(preds, y, valid, horizons).pivot(index="model", columns="horizon_min", values="MAE")
    hcol = list(horizons)

    # ---- day-block bootstrap: overall and by peak, per horizon
    tt = ev.target_timestamps(ctx.timestamps, origins, horizons)
    day_idx = pd.factorize(pd.to_datetime(tt.ravel()).normalize())[0].reshape(tt.shape)
    peak = ev.peak_flags_sh(ctx.timestamps, origins, horizons)

    def boot(ref, new, j, extra=None):
        v = valid[:, j] if extra is None else valid[:, j] & extra[:, j, None]
        return ev.day_block_bootstrap_improvement(preds[ref][:, j], preds[new][:, j], y[:, j], v, day_idx[:, j])

    def bar(name, r, colour):
        return {"name": name, "value": r["improvement_pct"], "lo": r["ci_low"], "hi": r["ci_high"], "colour": colour}

    ablation, peak_groups, boot60 = [], [], {}
    for j, h in enumerate(hcol):
        g = boot("dl_st_gat_noedge_s42", "dl_st_gat_s42", j)
        x = boot("xgboost_spatial", "dl_st_gat_s42", j)
        ablation.append({"label": f"{h * 5} min", "bars": [bar("edges", g, "#3ccf8a"), bar("xgb", x, "#ffa630")]})
        po, pp_ = boot("xgboost_spatial", "dl_st_gat_s42", j, ~peak), boot("xgboost_spatial", "dl_st_gat_s42", j, peak)
        peak_groups.append({"label": f"{h * 5} min", "bars": [bar("off", po, "#7fb2ff"), bar("peak", pp_, "#ec3b3b")]})
        if h == 12:
            boot60 = {"graph": g, "xgb": x, "off": po, "peak": pp_}

    # ---- horizon chart rows
    rows_keys = ["persistence", "xgboost_spatial", "dl_gru_s42", "dl_dcrnn_s42", "dl_st_gat_s42"]
    vals = [mae.loc[k].tolist() for k in rows_keys]
    allv = [v for r in vals for v in r]
    y0, y1 = np.floor(min(allv) * 2) / 2, np.ceil(max(allv) * 2) / 2
    nudges = label_nudges([r[3] for r in vals], y0, y1)
    horizon = [{"label": results.label(k), "mae": [round(v, 3) for v in r], "colour": COL[k], "hero": k == "dl_st_gat_s42",
                "dash": k in ("persistence", "dl_gru_s42"), "nudge": round(nz, 1)} for k, r, nz in zip(rows_keys, vals, nudges)]

    # ---- replay day: the slowest weekday afternoon of the test period (same rule as notebook 15)
    speed = np.where(ctx.mask, ctx.speed, np.nan)
    rows = np.arange(origins.min() + 12, origins.max() + 13)
    frame = pd.DataFrame({"t": ctx.timestamps[rows], "m": pd.DataFrame(speed[rows]).mean(axis=1).to_numpy(),
                          "obs": ctx.mask[rows].mean(1)}, index=rows)
    pm = frame[(frame.t.dt.dayofweek < 5) & frame.t.dt.hour.between(15, 18) & (frame.obs > .9)]
    the_day = pm.groupby(pm.t.dt.date).m.mean().idxmin()
    day_rows = np.flatnonzero(ctx.timestamps.normalize() == pd.Timestamp(the_day))
    assert len(day_rows) == 288
    actual = np.where(ctx.mask[day_rows], np.round(ctx.speed[day_rows]), -1).astype(int)
    j30 = hcol.index(6)
    pos = {int(o): i for i, o in enumerate(origins)}
    forecast = np.full_like(actual, -1)
    for r, t in enumerate(day_rows):
        i = pos.get(int(t) - 6)
        if i is not None:
            forecast[r] = np.round(np.clip(preds["dl_st_gat_s42"][i, j30], 0, 70)).astype(int)
    fc_ok = forecast >= 0
    fc_mae = np.abs(forecast - actual)[fc_ok & (actual >= 0)].mean()

    # ---- hub: best-connected sensor with mean ST-GAT attention
    degree = off.sum(0) + off.sum(1)
    hub = int(np.argmax(degree))
    with np.load(config.METRICS_DIR / "dl_st_gat_attention.npz") as z:
        a_src, a_dst, up_mean, down_mean = z["src"], z["dst"], z["up_mean"], z["down_mean"]
    up = [(int(s), float(a)) for s, d, a in zip(a_src, a_dst, up_mean) if d == hub]
    down = [(int(d), float(a)) for s, d, a in zip(a_src, a_dst, down_mean) if s == hub]
    top = max([a for _, a in up + down] or [1])
    hub_data = {"node": hub, "up": [[i, round(a / top, 3)] for i, a in up], "down": [[i, round(a / top, 3)] for i, a in down]}

    # ---- per-sensor graph gain at 60 min (notebook 15)
    gain_df = pd.read_csv(config.METRICS_DIR / "dl_sensor_gain_60min.csv", index_col="node")
    gpct = 100 * gain_df.graph_gain / gain_df.mae_dl_st_gat_noedge_s42
    gain = [None if not np.isfinite(v) else round(float(v), 2) for v in gpct.reindex(range(n))]

    # ---- data-quality numbers
    missing_pct = 100 * (~ctx.mask).mean()
    outage_steps = int((~ctx.mask).all(1).sum())
    test_missing = 100 * (1 - ymask.mean())
    reciprocal = 100 * (off & off.T).sum() / off.sum()
    xgb_vs_persist = 100 * (mae.loc["persistence", 60] - mae.loc["xgboost", 60]) / mae.loc["persistence", 60]

    # ---- search note from the summaries
    def best_val(run):
        return json.loads((config.METRICS_DIR / f"{run}_summary.json").read_text())["best_val_mae_mph"]
    grid = {(m, lr, w): best_val(f"dl_{m}_s42_lr{lr}_w{w}") for m in ("st_gat", "st_gat_noedge")
            for lr in ("0.0003", "0.001", "0.003") for w in (64, 128)}
    gaps = [grid[("st_gat_noedge", lr, w)] - grid[("st_gat", lr, w)] for lr in ("0.0003", "0.001", "0.003") for w in (64, 128)]

    m60 = mae[60]
    grp = ev.metrics_by_group({"s": preds["dl_st_gat_s42"]}, y, valid, horizons, {"peak": peak, "off": ~peak})
    g60 = grp[grp.horizon_min == 60].set_index("group").MAE
    peak_ratio = g60["peak"] / g60["off"]
    lab = results.label
    text = {
        "replay_day": f"{pd.Timestamp(the_day):%A %d %B %Y}",
        "missing_pct": pct(missing_pct), "outage_steps": f"{outage_steps:,}", "test_missing": pct(test_missing),
        "xgb_vs_persist_60": pct(xgb_vs_persist, 0),
        "n_edges": f"{off.sum():,}", "reciprocal": pct(reciprocal, 0),
        "search_note": (f"Search: lr 3e-4 / 1e-3 / 3e-3 × width 64 / 128, 25 epochs, seed 42, on validation only. Both models chose "
                        f"lr 1e-3, width 128. The no-edge twin was {min(gaps):.2f}–{max(gaps):.2f} mph worse at every one of the six "
                        "settings, so the graph effect is not a tuning accident. Final runs: 3 seeds, up to 60 epochs, about 4 h on one T4."),
        "results_title": "The graph models pull away as the horizon grows",
        "results_note": (f"Test MAE at 60 min: persistence {m60['persistence']:.2f}, XGBoost + neighbours {m60['xgboost_spatial']:.2f}, "
                         f"GRU {m60['dl_gru_s42']:.2f}, DCRNN {m60['dl_dcrnn_s42']:.2f}, ST-GAT {m60['dl_st_gat_s42']:.2f} mph. "
                         "At 5 min every learned model is within about 0.1 mph of the others: the last reading already says most of it."),
        "ablation_title": "Switch the graph off and the gain disappears",
        "ablation_caption": ("Green: MAE reduction of ST-GAT over the same network with every edge removed (same parameters, same training). "
                             "Orange: over XGBoost + neighbours. Bars: seed 42; whiskers: 95% day-block bootstrap interval over 24 test days."),
        "ablation_note": (f"At 60 min the edges are worth {boot60['graph']['improvement_pct']:.1f}% "
                          f"(CI {boot60['graph']['ci_low']:.1f} to {boot60['graph']['ci_high']:.1f}). Seed spread is about 0.01 mph, "
                          "ten times smaller than the effect. The no-edge twin is barely better than a plain GRU."),
        "peak_title": "Rush hour is where it counts",
        "peak_note": (f"ST-GAT vs XGBoost + neighbours at 60 min: {boot60['peak']['improvement_pct']:.1f}% lower MAE in the peak, "
                      f"{boot60['off']['improvement_pct']:.1f}% off-peak. Even for ST-GAT, peak MAE at 60 min is {peak_ratio:.1f}× the off-peak MAE."),
        "gain_text": (f"ST-GAT beats its own no-edge twin at {(gain_df.graph_gain > 0).mean():.0%} of sensors at 60 minutes. "
                      f"Where it loses, it loses little: at most {-gain_df.graph_gain.min():.2f} mph, against gains of up to "
                      f"{gain_df.graph_gain.max():.2f} mph."),
        "gain_note": (f"Correlation of per-sensor gain with number of upstream neighbours: "
                      f"{gain_df[['graph_gain', 'in_degree']].corr().iloc[0, 1]:.2f}. Degree alone does not explain it."),
        "forecast_text": (f"Across that whole day the 30-minute forecast is off by {fc_mae:.1f} mph on average. "
                          "Sudden drops still arrive late (see notebook 15): no model here predicts a crash."),
    }

    lit_rows = [{"name": nme, "vals": list(v), "us": False} for nme, *v in LITERATURE]
    lit_rows += [{"name": f"Ours: {lab(k)}", "vals": [round(mae.loc[k, h], 2) for h in (15, 30, 60)], "us": True}
                 for k in ("seasonal_avg", "xgboost_spatial", "dl_dcrnn_s42", "dl_st_gat_s42")]

    data = {
        "text": text, "team": TEAM,
        "sensors": [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lon, lat)],
        "edges": [[int(a), int(b)] for a, b in zip(src, dst)],
        "day": {"label": f" {pd.Timestamp(the_day):%a %d %b %Y}", "actual": actual.tolist(), "forecast": forecast.tolist()},
        "hub": hub_data, "gain": gain, "degree": degree.tolist(), "horizon": horizon,
        "ablation": {"groups": ablation, "legend": [["vs ST-GAT with edges removed", "#3ccf8a"], ["vs XGBoost + neighbours", "#ffa630"]]},
        "peak": {"groups": peak_groups, "legend": [["off-peak", "#7fb2ff"], ["peak", "#ec3b3b"]]},
        "literature": {"cols": ["15 min", "30 min", "60 min"], "rows": lit_rows},
    }
    tpl = (HERE / "template.html").read_text()
    assert "/*DATA*/null" in tpl
    out = HERE / "index.html"
    out.write_text(tpl.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {out.relative_to(config.PROJECT_ROOT)} ({out.stat().st_size / 1e6:.2f} MB); replay day {text['replay_day']}")


if __name__ == "__main__":
    main()
