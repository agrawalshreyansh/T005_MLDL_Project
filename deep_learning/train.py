"""Train one or more models and save checkpoint, history and val/test predictions.

    python -m deep_learning.train --models gru st_gat st_gat_noedge dcrnn --seeds 42
    TFC_RUN_MODE=smoke python -m deep_learning.train --models st_gat --max-epochs 2   # quick bug check

Outputs (run name ``dl_<model>_s<seed>``; ``_smoke`` appended in smoke mode):
    artifacts/models/<run>.pt                 best-validation weights + summary
    artifacts/metrics/<run>_history.csv       one row per epoch
    artifacts/metrics/<run>_summary.json
    artifacts/predictions/<run>__{val,test}.npz   [origins, 4 horizons, 207] mph, same format as Phase 1
Test predictions are written for later scoring; this script never prints a test metric.
"""
from __future__ import annotations

import argparse
import time

from src import config
from src.evaluation import save_predictions
from src.graph_utils import load_graph_artifacts
from src.preprocessing import load_context
from .dl import engine
from .dl.data import WindowData
from .dl.models import MODELS, build


def run_name(model: str, seed: int, tag: str = "") -> str:
    """Prediction files get the smoke tag from ``save_predictions``; models/metrics get it here."""
    return f"dl_{model}_s{seed}{tag}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train deep-learning forecasters on METR-LA")
    ap.add_argument("--models", nargs="+", choices=list(MODELS), required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[config.SEED])
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--width", type=int, help="d_model (GRU / ST-GAT) or hidden (DCRNN)")
    ap.add_argument("--tag", default="", help="suffix for the run name, e.g. a search setting")
    ap.add_argument("--device")
    args = ap.parse_args()

    device = engine.select_device(args.device)
    ctx = load_context()
    adj = load_graph_artifacts(expected_sensor_ids=ctx.sensor_ids).adjacency
    data = WindowData.from_context(ctx, device)
    print(f"device {device} | windows " + ", ".join(f"{k} {len(v)}" for k, v in data.origins.items()), flush=True)

    for model_name in args.models:
        for seed in args.seeds:
            engine.set_seed(seed)
            kwargs = {}
            if args.width:
                kwargs["hidden" if model_name == "dcrnn" else "d_model"] = args.width
            model = build(model_name, adj, **kwargs)
            params = {**MODELS[model_name][1], "seed": seed}
            for key, value in (("max_epochs", args.max_epochs), ("lr", args.lr)):
                if value is not None:
                    params[key] = value
            if config.IS_SMOKE:
                params.setdefault("max_epochs", 2)
            name = run_name(model_name, seed, args.tag)
            t0 = time.time()
            out = engine.train(model, data, params, out_dir=config.ARTIFACTS, name=config.tagged(name),
                               log=lambda s: print(s, flush=True))
            for split in ("val", "test"):
                z = engine.predict_scaled(model, data, split)[:, [h - 1 for h in config.HORIZONS], :]
                save_predictions(name, split, data.to_mph(z), ctx.origins[split], config.HORIZONS,
                                 meta={"run": name, "model": model_name, "seed": seed, "kwargs": kwargs,
                                       "best_epoch": out["best_epoch"], "best_val_mae_mph": out["best_val_mae_mph"]})
            print(f"{name}: done in {(time.time() - t0) / 60:.1f} min, best val MAE {out['best_val_mae_mph']:.3f} mph "
                  f"(epoch {out['best_epoch']})", flush=True)


if __name__ == "__main__":
    main()
