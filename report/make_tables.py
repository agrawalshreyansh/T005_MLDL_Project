"""Turns the metric CSVs written by notebooks 05c-07a into LaTeX tables for the report (so no number is typed by hand).

Usage (from the project root, after the notebooks have run):  python report/make_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "artifacts" / "metrics"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)

LABEL = {
    "persistence": "Persistence", "rolling_mean": "Rolling mean (60 min)", "seasonal_avg": "Seasonal average",
    "random_forest": "Random forest (multi-output)", "xgboost": "XGBoost", "xgboost_tuned": "XGBoost (tuned)",
    "xgboost_spatial": "XGBoost + neighbours",
}
ORDER = list(LABEL)


def tex(s: str) -> str:
    return s.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("#", r"\#")


def bold_best(col: pd.Series, digits: int) -> list[str]:
    best = col.min()
    return [(rf"\textbf{{{v:.{digits}f}}}" if np.isclose(v, best) else f"{v:.{digits}f}") for v in col]


def metrics_table() -> None:
    t = pd.read_csv(METRICS / "phase1_test_metrics.csv")
    models = [m for m in ORDER if m in set(t.model)]
    hz = sorted(t.horizon_min.unique())
    cols = {}
    for metric in ("MAE", "RMSE", "MAPE"):
        piv = t.pivot(index="model", columns="horizon_min", values=metric).loc[models]
        for h in hz:
            cols[(metric, h)] = bold_best(piv[h], 2)
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Test-period error by forecast horizon (24 held-out days, 207 sensors). MAE and RMSE in mph, MAPE in \%; lower is better, best per column in bold. All models are scored on the same valid points.}",
             r"\label{tab:main}", r"\small",
             r"\begin{tabular}{@{}l" + "r" * (3 * len(hz)) + r"@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{len(hz)}}}{{c}}{{{m}}}" for m in ("MAE (mph)", "RMSE (mph)", "MAPE (\\%)")) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + i * len(hz)}-{1 + (i + 1) * len(hz)}}}" for i in range(3)),
             "Model & " + " & ".join(f"{h} min" for _ in range(3) for h in hz) + r" \\", r"\midrule"]
    for i, m in enumerate(models):
        lines.append(tex(LABEL[m]) + " & " + " & ".join(cols[k][i] for k in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "tab_main.tex").write_text("\n".join(lines) + "\n")


def peak_table() -> None:
    g = pd.read_csv(METRICS / "phase1_test_metrics_peak_offpeak.csv")
    models = [m for m in ("persistence", "xgboost", "random_forest", "xgboost_spatial", "xgboost_tuned") if m in set(g.model)]
    hz = sorted(g.horizon_min.unique())
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Test MAE (mph) in weekday peak hours (07--10, 15--19, by forecast target time) and all other hours.}",
             r"\label{tab:peak}", r"\small", r"\begin{tabular}{@{}ll" + "r" * len(hz) + r"@{}}", r"\toprule",
             "Period & Model & " + " & ".join(f"{h} min" for h in hz) + r" \\", r"\midrule"]
    for grp in ("peak", "off-peak"):
        sub = g[g.group == grp].pivot(index="model", columns="horizon_min", values="MAE").loc[models]
        for i, m in enumerate(models):
            first = rf"\multirow{{{len(models)}}}{{*}}{{{grp}}}" if i == 0 else ""
            lines.append(f"{first} & {tex(LABEL[m])} & " + " & ".join(f"{sub.loc[m, h]:.2f}" for h in hz) + r" \\")
        if grp == "peak":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_peak.tex").write_text("\n".join(lines) + "\n")


def improvement_table() -> None:
    imp = pd.read_csv(METRICS / "phase1_improvements.csv", header=[0], index_col=[0, 1])
    imp.index.names = ["pair", "horizon"]
    imp = imp.reset_index()
    names = {"xgboost_vs_persistence": "XGBoost vs.\\ persistence", "random_forest_vs_xgboost": "Random forest vs.\\ XGBoost",
             "xgboost_spatial_vs_xgboost": "XGB + neighbours vs.\\ XGBoost", "xgboost_spatial_vs_random_forest": "XGB + neighbours vs.\\ random forest",
             "xgboost_tuned_vs_xgboost": "Tuned vs.\\ default XGBoost"}
    hz = sorted(imp.horizon.unique())
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{MAE improvement (\%) of the first model over the second, with 95\% day-block bootstrap interval. Positive means the first model is better; an interval that excludes zero is a consistent difference over the 24 test days.}",
             r"\label{tab:improve}", r"\small", r"\begin{tabular}{@{}l" + "r" * len(hz) + r"@{}}", r"\toprule",
             "Comparison & " + " & ".join(f"{h} min" for h in hz) + r" \\", r"\midrule"]
    for key, label in names.items():
        sub = imp[imp.pair == key].set_index("horizon")
        if sub.empty:
            continue
        cells = [rf"{sub.loc[h, 'improvement_%']:.1f} [{sub.loc[h, '95% CI low']:.1f}, {sub.loc[h, '95% CI high']:.1f}]" for h in hz]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "tab_improve.tex").write_text("\n".join(lines) + "\n")


def ablation_table() -> None:
    a = pd.read_csv(METRICS / "feature_ablation.csv")
    hz = sorted(a.horizon_min.unique())
    stages = list(dict.fromkeys(a.stage))
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Cumulative feature-group ablation (validation MAE in mph; gain relative to the previous stage in \%). Persistence on the same rows is the reference.}",
             r"\label{tab:ablation}", r"\small", r"\begin{tabular}{@{}l" + "rr" * len(hz) + r"@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h} min}}" for h in hz) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(hz))),
             "Stage & " + " & ".join("MAE & gain" for _ in hz) + r" \\", r"\midrule"]
    for s in stages:
        row = [tex(s)]
        for h in hz:
            r = a[(a.stage == s) & (a.horizon_min == h)].iloc[0]
            row += [f"{r['validation MAE (mph)']:.3f}", f"{r['gain vs previous stage (%)']:.1f}"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (OUT / "tab_ablation.tex").write_text("\n".join(lines) + "\n")


def tuning_table() -> None:
    p = METRICS / "xgb_tuning_cv.csv"
    if not p.is_file():
        return
    c = pd.read_csv(p, index_col=0)
    s = pd.read_csv(METRICS / "xgb_tuning_search.csv", index_col=0)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Hyper-parameter study at 30 minutes. Rolling-origin cross-validation inside the training period for the default and the best random-search candidates (MAE of the residual, mph).}",
             r"\label{tab:tuning}", r"\small", r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Config. & depth & min.\ child & $\eta$ & CV MAE & vs.\ default (\%) \\", r"\midrule"]
    for name, r in c.iterrows():
        lines.append(f"{tex(name)} & {int(s.loc[name, 'max_depth'])} & {int(s.loc[name, 'min_child_weight'])} & {s.loc[name, 'learning_rate']:.2f} & {r['mean CV MAE']:.4f} & {r['vs default (%)']:+.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tab_tuning.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    for fn in (metrics_table, peak_table, improvement_table, ablation_table, tuning_table):
        fn()
        print("wrote", fn.__name__)
