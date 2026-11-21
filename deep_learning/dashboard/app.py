"""METR-LA forecast dashboard.

    streamlit run deep_learning/dashboard/app.py

Reads only saved files (predictions, training histories, attention summary); nothing is trained here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deep_learning.dl import results  # noqa: E402
from src import config, evaluation as ev, graph_utils as gu, preprocessing as pp  # noqa: E402

st.set_page_config(page_title="METR-LA forecasts", page_icon="🛣️", layout="wide")
HZ_MIN = {h: h * config.STEP_MINUTES for h in config.HORIZONS}


# --------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading data and predictions...")
def load():
    ctx = pp.load_context()
    graph = gu.load_graph_artifacts(expected_sensor_ids=ctx.sensor_ids)
    scored = results.load_scored(ctx)
    return ctx, graph, scored


@st.cache_data
def leaderboard(_scored, keys: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx, _, scored = load()
    return scored.by_horizon(), scored.by_peak(ctx)


def speed_rgb(v: np.ndarray) -> np.ndarray:
    """Red (jam) -> amber -> green (free flow); grey for missing."""
    stops = np.array([[0, 190, 40, 40], [30, 225, 120, 30], [50, 235, 200, 60], [65, 40, 170, 90]], dtype=float)
    out = np.stack([np.interp(v, stops[:, 0], stops[:, i]) for i in (1, 2, 3)], axis=-1)
    out[~np.isfinite(v)] = (120, 120, 120)
    return out.astype(int)


def error_rgb(e: np.ndarray) -> np.ndarray:
    a = np.clip(np.abs(e) / 15.0, 0, 1)[:, None]
    base = np.where(e[:, None] > 0, [[60, 120, 220]], [[220, 70, 60]])      # blue = over-predicted, red = under
    out = (1 - a) * np.array([[200, 200, 200]]) + a * base
    out[~np.isfinite(e)] = (120, 120, 120)
    return out.astype(int)


def choose(widget, label: str, options: dict, default=0, **kw):
    """Widgets get plain string labels; the chosen label is mapped back to its value."""
    names = list(options)
    if getattr(widget, "__name__", "") == "select_slider":
        return options[widget(label, options=names, value=names[default], **kw)]
    return options[widget(label, names, index=default, **kw)]


ctx, graph, scored = load()
keys = list(scored.preds)
loc = graph.locations
lat, lon = loc["latitude"].to_numpy(), loc["longitude"].to_numpy()
test_ts = ctx.timestamps[scored.origins]

st.sidebar.title("METR-LA forecasts")
st.sidebar.caption("207 Los Angeles freeway sensors · test period 4–27 June 2012 · speeds in mph")
page = st.sidebar.radio("View", ["Network map", "Replay", "Sensor explorer", "Leaderboard", "Graph lens", "Training", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Models loaded: " + ", ".join(results.label(k) for k in keys))


# --------------------------------------------------------------------------- pages
def network_map():
    st.header("Network map")
    c1, c2, c3 = st.columns([2, 1, 1])
    key = choose(c1.selectbox, "Model", {results.label(k): k for k in keys},
                 keys.index("dl_st_gat_s42") if "dl_st_gat_s42" in keys else len(keys) - 1)
    h = choose(c2.selectbox, "Horizon", {f"{HZ_MIN[h]} min ahead": h for h in config.HORIZONS}, 3)
    view = c3.radio("Colour by", ["forecast", "actual", "error"], horizontal=True)
    j = config.HORIZONS.index(h)
    target_ts = test_ts + pd.Timedelta(minutes=HZ_MIN[h])
    days = sorted({t.date() for t in target_ts})
    d1, d2 = st.columns([1, 3])
    day = choose(d1.selectbox, "Day", {d.strftime("%a %d %b"): d for d in days}, min(7, len(days) - 1))
    on_day = np.flatnonzero(np.array([t.date() == day for t in target_ts]))
    default = int(on_day[np.argmin(np.abs(np.array([t.hour * 60 + t.minute for t in target_ts[on_day]]) - 17 * 60 - 30))])
    times = {target_ts[i].strftime("%H:%M"): int(i) for i in on_day}
    s = choose(d2.select_slider, "Time of the forecast target", times, list(times.values()).index(default))
    pred, true = scored.preds[key][s, j], np.where(scored.valid[s, j], scored.y[s, j], np.nan)
    colour = {"forecast": speed_rgb(pred), "actual": speed_rgb(true), "error": error_rgb(pred - true)}[view]
    df = pd.DataFrame({"lat": lat, "lon": lon, "sensor": ctx.sensor_ids, "forecast": pred.round(1), "actual": np.round(true, 1),
                       "r": colour[:, 0], "g": colour[:, 1], "b": colour[:, 2]})
    src, dst = np.nonzero((graph.adjacency > 0) & ~np.eye(len(lat), dtype=bool))
    edges = pd.DataFrame({"s": np.c_[lon[src], lat[src]].tolist(), "t": np.c_[lon[dst], lat[dst]].tolist()})
    deck = pdk.Deck(
        map_style=pdk.map_styles.CARTO_DARK,
        initial_view_state=pdk.ViewState(latitude=float(lat.mean()), longitude=float(lon.mean()), zoom=10.3),
        layers=[pdk.Layer("LineLayer", edges, get_source_position="s", get_target_position="t", get_color=[110, 110, 130, 90], get_width=1),
                pdk.Layer("ScatterplotLayer", df, get_position="[lon, lat]", get_fill_color="[r, g, b]", get_radius=260,
                          pickable=True, stroked=True, get_line_color=[20, 20, 20], line_width_min_pixels=1)],
        tooltip={"text": "sensor {sensor}\nforecast {forecast} mph\nactual {actual} mph"})
    st.pydeck_chart(deck, height=560)
    ok = scored.valid[s, j]
    m1, m2, m3 = st.columns(3)
    m1.metric("Network MAE at this moment", f"{np.abs(pred - scored.y[s, j])[ok].mean():.2f} mph")
    m2.metric("Sensors below 35 mph (actual)", int((true[ok] < 35).sum()))
    m3.metric("Forecast made at", (target_ts[s] - pd.Timedelta(minutes=HZ_MIN[h])).strftime("%H:%M"))
    st.caption("Red = slow, green = free flow. In error view: red = forecast too low, blue = too high, grey = sensor offline.")


def sensor_explorer():
    st.header("Sensor explorer")
    c1, c2, c3 = st.columns([1, 1, 2])
    var = np.nanstd(ctx.speed[: ctx.splits.train[1]], axis=0)
    order = [int(i) for i in np.argsort(-var)]
    n = choose(c1.selectbox, "Sensor (most variable first)", {ctx.sensor_ids[i]: i for i in order})
    h = choose(c2.selectbox, "Horizon", {f"{HZ_MIN[h]} min ahead": h for h in config.HORIZONS}, 2)
    labels = {results.label(k): k for k in keys}
    picked = c3.multiselect("Models", list(labels), default=[results.label(k) for k in keys if k in ("xgboost", "dl_st_gat_s42")] or [results.label(keys[-1])])
    chosen = [labels[x] for x in picked]
    j = config.HORIZONS.index(h)
    target_ts = test_ts + pd.Timedelta(minutes=HZ_MIN[h])
    days = sorted({t.date() for t in target_ts})
    day = choose(st.select_slider, "Day", {d.strftime("%a %d %b"): d for d in days}, min(7, len(days) - 1))
    sel = np.array([t.date() == day for t in target_ts])
    fig = go.Figure()
    fig.add_scatter(x=target_ts[sel], y=np.where(scored.valid[sel, j, n], scored.y[sel, j, n], np.nan),
                    name="actual", line=dict(color="black", width=2))
    for k in chosen:
        fig.add_scatter(x=target_ts[sel], y=scored.preds[k][sel, j, n], name=results.label(k), line=dict(width=1.5))
    fig.update_layout(height=460, yaxis_title="speed (mph)", xaxis_title="target time", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")
    rows = [{"model": results.label(k), "MAE this day (mph)": np.abs(scored.preds[k][sel, j, n] - scored.y[sel, j, n])[scored.valid[sel, j, n]].mean()}
            for k in chosen]
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)


def leaderboard_page():
    st.header("Leaderboard (test period)")
    by_h, by_p = leaderboard(scored, tuple(keys))
    metric = st.radio("Metric", ["MAE", "RMSE", "MAPE"], horizontal=True)
    unit = "%" if metric == "MAPE" else "mph"
    table = by_h.pivot(index="model", columns="horizon_min", values=metric).reindex([results.label(k) for k in keys])
    table.columns = [f"{c} min" for c in table.columns]
    st.dataframe(table.style.format("{:.2f}").highlight_min(axis=0, color="#1f6f4a"), width="stretch")
    fig = px.line(by_h, x="horizon_min", y=metric, color="model", markers=True,
                  labels={"horizon_min": "horizon (min)", metric: f"{metric} ({unit})"})
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
    st.subheader("Peak vs off-peak")
    st.caption("Peak = weekdays 07–10 and 15–19 (time of the forecast target), fixed before any model was trained.")
    fig = px.bar(by_p[by_p.horizon_min == st.select_slider("Horizon", [5, 15, 30, 60], 60)], x="model", y=metric,
                 color="group", barmode="group", labels={metric: f"{metric} ({unit})"})
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{int(scored.valid.sum()):,} valid target points; a point missing for any model is excluded for all.")


def graph_lens():
    st.header("Graph lens")
    att_path = config.METRICS_DIR / "dl_st_gat_attention.npz"
    deg = (graph.adjacency > 0).sum(0) + (graph.adjacency > 0).sum(1) - 2
    n = choose(st.selectbox, "Sensor (most connected first)", {ctx.sensor_ids[int(i)]: int(i) for i in np.argsort(-deg)})
    adj = graph.adjacency
    up = [i for i in np.flatnonzero(adj[:, n] > 0) if i != n]       # edges i -> n
    down = [i for i in np.flatnonzero(adj[n, :] > 0) if i != n]     # edges n -> i
    weight = {("up", i): adj[i, n] for i in up} | {("down", i): adj[n, i] for i in down}
    source = "road-closeness weight (adjacency)"
    if att_path.is_file():
        with np.load(att_path) as z:
            src, dst = z["src"], z["dst"]
            for direction in ("up", "down"):
                a = z[f"{direction}_mean"]
                for i in (up if direction == "up" else down):
                    hit = np.flatnonzero((src == i) & (dst == n)) if direction == "up" else np.flatnonzero((src == n) & (dst == i))
                    if len(hit):
                        weight[(direction, i)] = float(a[hit[0]])
        source = "mean ST-GAT attention on the test period (layer 1, averaged over heads)"
    rows = [{"direction": "upstream" if d == "up" else "downstream", "sensor": ctx.sensor_ids[i], "weight": w,
             "lat": lat[i], "lon": lon[i]} for (d, i), w in weight.items()]
    df = pd.DataFrame(rows)
    c1, c2 = st.columns([3, 2])
    with c1:
        df_map = pd.concat([df, pd.DataFrame([{"direction": "selected", "sensor": ctx.sensor_ids[n], "weight": 1.0, "lat": lat[n], "lon": lon[n]}])])
        df_map["colour"] = df_map.direction.map({"upstream": [42, 120, 214], "downstream": [235, 104, 52], "selected": [255, 255, 255]})
        df_map["size"] = 120 + 500 * df_map.weight.clip(0, 1)
        st.pydeck_chart(pdk.Deck(map_style=pdk.map_styles.CARTO_DARK,
                                 initial_view_state=pdk.ViewState(latitude=float(lat[n]), longitude=float(lon[n]), zoom=12.5),
                                 layers=[pdk.Layer("ScatterplotLayer", df_map, get_position="[lon, lat]", get_fill_color="colour",
                                                   get_radius="size", pickable=True)],
                                 tooltip={"text": "{direction} {sensor}\nweight {weight}"}), height=480)
    with c2:
        st.caption(f"Weights shown: {source}. Blue = upstream (feeds into this sensor), orange = downstream.")
        st.dataframe(df[["direction", "sensor", "weight"]].sort_values("weight", ascending=False).round(3), hide_index=True)
        if not len(df):
            st.info("This sensor has no road-graph neighbours, so the graph models cannot help it.")


def training():
    st.header("Training curves")
    files = sorted(config.METRICS_DIR.glob("dl_*_history.csv"))
    if not files:
        st.info("No training histories yet.")
        return
    runs = {f.name.removesuffix("_history.csv"): pd.read_csv(f) for f in files}
    default = [r for r in runs if r.endswith("_s42")] or list(runs)[:4]
    chosen = st.multiselect("Runs", list(runs), default=default)
    long = pd.concat([d.assign(run=r) for r, d in runs.items() if r in chosen]) if chosen else pd.DataFrame()
    if long.empty:
        return
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(long, x="epoch", y="val_mae_mph", color="run", title="validation MAE (mph, all 12 steps)"), width="stretch")
    c2.plotly_chart(px.line(long, x="epoch", y="train_mae_mph", color="run", title="training MAE (mph)"), width="stretch")
    summaries = [json.loads((config.METRICS_DIR / f"{r}_summary.json").read_text()) for r in chosen
                 if (config.METRICS_DIR / f"{r}_summary.json").is_file()]
    st.dataframe(pd.DataFrame([{k: s[k] for k in ("name", "parameters", "best_epoch", "best_val_mae_mph", "epochs_run", "device")}
                               for s in summaries]).round(3), hide_index=True)


def replay():
    st.header("Replay a test day")
    st.caption("Press Play: the left map shows measured speeds, the right map what the model forecast for that same moment, "
               "issued earlier by the chosen horizon. Everything runs in your browser after one load.")
    c1, c2, c3 = st.columns([2, 1, 1])
    key = choose(c1.selectbox, "Model", {results.label(k): k for k in keys},
                 keys.index("dl_st_gat_s42") if "dl_st_gat_s42" in keys else len(keys) - 1, key="rp_model")
    h = choose(c2.selectbox, "Horizon", {f"{HZ_MIN[h]} min ahead": h for h in config.HORIZONS}, 2, key="rp_h")
    step = choose(c3.selectbox, "Frame every", {"5 min": 1, "10 min": 2, "15 min": 3}, 1, key="rp_step")
    j = config.HORIZONS.index(h)
    target_ts = test_ts + pd.Timedelta(minutes=HZ_MIN[h])
    days = sorted({t.date() for t in target_ts})
    day = choose(st.select_slider, "Day", {d.strftime("%a %d %b"): d for d in days}, min(11, len(days) - 1), key="rp_day")
    idx = np.flatnonzero(np.array([t.date() == day and 5 <= t.hour < 22 for t in target_ts]))[::step]   # 05:00-22:00

    missing = -5.0                                     # grey for offline sensors
    scale = [[0, "#6b6b6b"], [5 / 75, "#6b6b6b"], [5 / 75, "#be2828"], [35 / 75, "#e1781e"],
             [55 / 75, "#ebc83c"], [1, "#28aa5a"]]

    def frame_values(s):
        ok = scored.valid[s, j]
        act = np.where(ok, scored.y[s, j], missing)
        fc = scored.preds[key][s, j]
        mae = np.abs(fc - scored.y[s, j])[ok].mean() if ok.any() else np.nan
        return act, fc, mae

    def title(s, mae):
        t = target_ts[s]
        return (f"<b>{t:%H:%M}</b>  ·  {results.label(key)} forecast issued {t - pd.Timedelta(minutes=HZ_MIN[h]):%H:%M}"
                f"  ·  network MAE {mae:.2f} mph")

    def trace(values, subplot, show_scale):
        return go.Scattermap(lat=lat, lon=lon, mode="markers", subplot=subplot, text=ctx.sensor_ids,
                             hovertemplate="sensor %{text}<br>%{marker.color:.1f} mph<extra></extra>",
                             marker=dict(size=9, color=values, colorscale=scale, cmin=missing, cmax=70,
                                         showscale=show_scale, colorbar=dict(title="mph", x=1.0)))

    act0, fc0, mae0 = frame_values(idx[0])
    fig = go.Figure(data=[trace(act0, "map", False), trace(fc0, "map2", True)])
    fig.frames = [go.Frame(name=str(i), data=[go.Scattermap(marker=dict(color=a)), go.Scattermap(marker=dict(color=f))],
                           traces=[0, 1], layout=dict(title_text=title(s, m)))
                  for i, s in enumerate(idx) for a, f, m in [frame_values(s)]]
    view = dict(style="carto-darkmatter", center=dict(lat=float(lat.mean()), lon=float(lon.mean())), zoom=9.4)
    play = dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=350, redraw=True), fromcurrent=True, transition=dict(duration=0))])
    pause = dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=50, b=0), title_text=title(idx[0], mae0),
        showlegend=False, map=dict(domain=dict(x=[0, .49], y=[0, 1]), **view), map2=dict(domain=dict(x=[.51, 1], y=[0, 1]), **view),
        annotations=[dict(text="measured", x=.01, y=.98, xref="paper", yref="paper", showarrow=False, font=dict(color="white", size=15)),
                     dict(text="forecast", x=.52, y=.98, xref="paper", yref="paper", showarrow=False, font=dict(color="white", size=15))],
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-.02, xanchor="left", yanchor="top", buttons=[play, pause], active=-1,
                          bgcolor="#2b6cb0", font=dict(color="white", size=14), bordercolor="#2b6cb0")],
        sliders=[dict(x=.12, len=.88, y=-.02, currentvalue=dict(visible=False),
                      steps=[dict(label=f"{target_ts[s]:%H:%M}", method="animate",
                                  args=[[str(i)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                             for i, s in enumerate(idx)])])
    st.plotly_chart(fig, width="stretch")
    st.caption("Red = slow, green = free flow, grey = sensor offline (not scored). "
               "Watch a rush hour: where the measured map turns red, the forecast map, made earlier, should already be red.")


def about():
    st.header("About")
    st.markdown("""
**Question.** Does letting each road sensor see its upstream and downstream neighbours (a graph neural network)
forecast speed better than models that treat every sensor on its own?

**Data.** METR-LA: 207 loop detectors on Los Angeles freeways, 5-minute average speeds, 1 March – 27 June 2012.
Chronological split 83 / 12 / 24 days. Missing readings (stored as 0 mph) are masked, never treated as a standstill.

**Models.** Persistence and seasonal average; XGBoost with and without neighbour features (Phase 1);
a GRU over time only; ST-GAT (GRU + graph attention over the directed road graph) and the same network with every edge removed;
DCRNN (diffusion convolution inside a seq2seq GRU).

**Scoring.** Every model is scored on the same test points; a point missing for any model is excluded for all.
Forecasts are clipped to 0–70 mph.

**Limits.** One city, 119 days, speed only. Forecasts are not used to control anything: signal control in this project
is simulated only.
""")


{"Network map": network_map, "Replay": replay, "Sensor explorer": sensor_explorer, "Leaderboard": leaderboard_page,
 "Graph lens": graph_lens, "Training": training, "About": about}[page]()
