# Experiment notes: decisions, deviations and problems

This file records *why* things are the way they are, including what went wrong during development. Results are in `results_summary.md`.

## 1. Data provenance and integrity

* **Source of `metr-la.h5`.** The original file is only distributed through Google Drive / Baidu (links in the DCRNN README). The file used here came from a third-party Hugging Face mirror
  (`MintBruce/SkyTraffic`, 57,038,056 bytes). It was validated structurally (key `df`, 34,272 x 207, strictly increasing exact 5-minute grid, sensor order identical to DCRNN's `graph_sensor_ids.txt`)
  but **not** compared byte-for-byte with the original; checksums are in `data/README.md`.
* **Reading the HDF5.** `h5py` is used instead of `pandas.read_hdf`, because pandas' fixed format can unpickle object arrays and the file is third-party. The loader rejects any object-dtype dataset.
  (The sensor ids are fixed-width byte strings and timestamps int64 nanoseconds, so nothing needs unpickling.)
* **Missing = 0.** There are no NaNs; 8.1% of cells are exactly 0.0 and 2,148 time steps have *all* 207 sensors at exactly 0, which cannot be real traffic. Zeros are therefore missing (NaN + observed mask),
  never zero speed. The smallest positive reading is 0.33 mph and is kept.
* **Adjacency.** Rebuilt from `distances_la_2012.csv` with DCRNN's `gen_adj_mx` logic (read from the original source; not vendored). `adj_mx.pkl` was deliberately neither downloaded nor unpickled.
  The rebuilt matrix has 1,722 non-zero entries (207 self-loops + 1,515 directed edges), the count commonly quoted for this graph; this is a plausibility check, not a proof of identity with the pickle.
* **Peak definition** (weekdays 07-10 and 15-19 local, judged on the forecast target time) was fixed from the weekday speed profile in notebook 01 *before* any model was trained.
* **Time zone.** Timestamps are naive; Pacific local time is assumed, supported by the commute dips at ~08:00 and ~17:00. No daylight-saving handling is done.

## 2. Design decisions

| Decision | Reason |
|---|---|
| Day-aligned split (83 / 12 / 24 days = 69.7 / 10.1 / 20.2 %) | whole days per split; weekday/time-of-day structure never cut mid-day |
| A sample belongs to the split holding all its targets; inputs may reach into the earlier split | past context is legitimately available; training windows never touch validation/test rows |
| Global z-score + per-sensor training mean, fitted on training rows only | no leakage; verified by corrupting val/test and refitting |
| Anchor = last observed speed in the 60-min window (training sensor mean if none); learned models predict residuals from it | one shared formulation: models are compared on what they add beyond persistence |
| XGBoost: global model per horizon, direct strategy, `reg:absoluteerror`, categorical sensor id, no other-sensor features | 207 x 4 per-sensor models would be costly and data-starved; the L1 objective matches the graph model's MAE loss; no neighbour features keeps it an honest "independent" baseline |
| No seasonal-mean feature in XGBoost / GAT | a seasonal mean fitted on the training rows contains the training target itself, creating a train/test mismatch (leakage in-sample) |
| Graph attention implemented as masked dense attention (not PyG `GATConv`) | N = 207 is tiny; dense masked attention batches trivially and runs on MPS without scatter ops. Verified equal to `GATConv` (max abs difference ~1e-7) in tests and in notebook 06 |
| Separate in-edge and out-edge attention, summed | the graph is directed (only about 27% of edges are reciprocal); we do not symmetrise |
| Ablation `use_graph=False` (self-only attention, identical parameter count) | isolates the effect of the road graph from model size and training recipe |
| Predictions clipped to 0-70 mph before scoring, for **every** model | observed speeds are positive and capped at 70; identical rule for all models; raw model output stays on disk |
| MAPE only for true speed >= 1 mph, with its own valid count | avoids near-zero denominators (only 9 positive readings are below 1 mph) |
| Day-block bootstrap for improvement intervals | errors within a day are strongly correlated; resampling points would give over-confident intervals |
| Test metrics are not displayed in notebooks 05 / 06 | model selection uses validation only; the test comparison happens once, in notebook 07 |

## 3. Problems found and how they were handled

1. **XGBoost would not import on macOS** (missing OpenMP runtime `libomp.dylib`). The documented fix is `brew install libomp`, a system-level change that was not requested. Instead the venv's copy of `libxgboost.dylib` got an extra `rpath`
   to the `libomp.dylib` bundled inside PyTorch and was re-signed ad hoc. This only touches the project-local `.venv`; on another machine use `brew install libomp`.
2. **pandas 3 changed datetime resolution** (microseconds by default). A test wrote microsecond integers into a field the loader reads as nanoseconds. The loader now passes `unit="ns"` explicitly and the test helper converts. Also `DatetimeIndex[2-D array]` is no longer allowed; code goes through `.values`.
3. **A wrong claim caught by looking at a figure.** The first draft of notebook 03 said the adjacency heatmap was concentrated near the diagonal. The figure shows only the self-loop diagonal is structured and the edges are scattered; the text was corrected.
   Similar corrections: "gaps of weeks" (the longest gap is 5.8 days) and "many close pairs lack edges" (16% of pairs within 1 km). Interpretations that depend on results are now asserted in code or generated from the tables.
4. **Off-by-one in the signal controller.** The first version returned the *next* state's colour, so the very first green lasted `min_green - 1` seconds; the safety checker only inspected interior runs and missed it. Fixed (the controller shows the current state, then advances) and the tests now check the first green and the clearance times exactly.
5. **Tautological fairness checks** in the first draft of notebook 07 (two rows that were always `True`) were replaced by checks that can actually fail.
6. **Memory pressure on the 8 GB machine.** Running training jobs concurrently (ST-GAT ~3-4 GB on MPS, XGBoost ~2 GB, DCRNN ~2.5 GB) pushed the machine into swap: an ST-GAT epoch went from ~90-115 s to 460-600 s, and a first DCRNN smoke test with batch size 64 used ~4.7 GB.
   The first DCRNN smoke test was stopped, the batch size was cut to 16, and a later DCRNN run that shared the machine with the ablation was killed after 6 epochs and restarted alone. Jobs were afterwards run one at a time. XGBoost and the primary ST-GAT did overlap for a while (slower, but no results were affected: XGBoost reproduced identically on a second, uncontended execution).
7. **SUMO.** It was not installed at first, so notebook 09 initially only validated files and safety logic. Later SUMO 1.28 was installed into `.venv` with pip (`eclipse-sumo`, `traci`); the previously unrun TraCI glue needed three changes: locating the SUMO binaries (the Jupyter kernel's PATH does not include `.venv/bin`), silencing TraCI's connection-retry chatter and SUMO's locale warning, and an explicit per-step check that SUMO shows the controller's signal state (it always did). One episode of 3,600 s takes about 2 s; 20 seeds per policy were run. A later fairness check added a Webster-tuned time-of-day fixed plan: it beats the queue-aware rule on every seed, which reversed the earlier headline (queue-aware ~30% better than the untuned 30/30 plan).

## 4. What is *not* claimed

* No claim that attention weights are causal or that low speed identifies a particular incident.
* No claim that the DCRNN-style model reproduces the published DCRNN (it is a re-implementation with a reduced budget).
* No claim that the simulated intersection represents any real location or that the METR-LA forecasts were used by the signal controller; the fixed-time baseline was not tuned, so the simulated improvement is scenario-specific.
* No statement about generalisation beyond this city and these 119 days.

## 5. Provenance of the delivered run

* **Executed in full mode** (not smoke): notebooks 00-09 and 06b. Notebook 09 ran the SUMO experiment (20 seeds x 2 policies). Notebook 06b (optional DCRNN-style model) was retrained with the same 30-epoch / patience-6 budget as the ST-GAT (stride-3 windows, deterministic) and is included in notebook 07; an earlier 12-epoch run was superseded.
* **Training happened once, in earlier executions.** XGBoost (notebook 05) and the ST-GAT + ablation (notebook 06) were trained with `RETRAIN = True`; the delivered notebooks have `RETRAIN = False`, so re-executing them reloads the saved models/checkpoints and recomputes everything downstream (forecasts, scores, plots, attention). The original epoch-by-epoch logs are printed in notebook 06 on reload and kept in `artifacts/metrics/*_train.log`. This is why the delivered notebook 06 shows a reload message rather than the live training loop.
* **ST-GAT training:** 30 epochs (the configured maximum; early stopping did not trigger for either model), best epoch 28 for both, selected by validation MAE (graph model 2.814 mph, no-edge 2.952 mph over all 12 forecast steps). The models were still improving only marginally at the end.
* **Seeds:** the ST-GAT, its no-edge ablation (and later the DCRNN-style model) were trained with seeds 42, 1 and 2 (the extra seeds in a later execution of notebook 06 with `EXTRA_SEEDS = (1, 2)`, `RETRAIN = False`, so the seed-42 checkpoints were reloaded, not retrained). Seed-to-seed std of ST-GAT test MAE is about 0.01 mph; every seed beats XGBoost and its same-seed no-edge twin at every horizon (notebook 07, section 9, checked claims). Seed 2 of the ablation early-stopped at epoch 24 (best epoch 18); the others ran 26-30 epochs. XGBoost (deterministic) and the DCRNN-style model have one run each.
* **Interpretation checks** (`evaluation.report_claims`): every sentence of interpretation that depends on a result is checked against the tables in the same notebook and a warning is printed if it fails; in the delivered run none failed. They warn rather than raise so that a retrained run with other numbers is not blocked (`TFC_STRICT_CLAIMS=1` makes them errors).
* **Tests:** 100 pass (`python -m pytest tests -q`); every notebook, including the one-file version, is additionally validated structurally by `tests/test_notebooks.py`.
* **Smoke mode** (`TFC_RUN_MODE=smoke`) was run end to end for notebooks 04-08 on throwaway copies (about a minute in total) and all its artifacts were deleted afterwards; it never touches the real artifacts or `results_summary.md`.
* **DCRNN-style model:** one seed, every third training window (~8k of ~24k origins), batch size 16 for memory, at most 30 epochs (same budget as the ST-GAT). A first 30-epoch attempt early-stopped at epoch 16 because patience counted the noisy epochs while teacher forcing was still being phased out; the rule was fixed (patience and LR plateau only count epochs after teacher forcing reaches 0) and the model retrained: it ran to the 30-epoch limit with best epoch 27. Its validation MAE (2.837) is marginally better than a superseded 12-epoch run (2.839), yet its test MAE is a few hundredths of a mph worse at every horizon, a reminder that differences of that size are within run-to-run variation. Not a converged or tuned result and not the original DCRNN.
* **DCRNN seeds:** seeds 1 and 2 were trained later with the same budget (seed 1 early-stopped at epoch 24, best epoch 9; seed 2 ran to the epoch limit). Its seed-to-seed spread is several times that of the ST-GAT (up to ~0.15 mph MAE at 60 min); averaged over seeds the ST-GAT is better at every horizon.
* **Spatial XGBoost (notebook 05b):** identical to notebook 05 plus 10 neighbour features from the road graph (upstream/downstream weighted mean speed now and 15 min earlier, slowest neighbour, number observed). Best model at 5 minutes on all metrics; the ST-GAT is better at 15-60 minutes.
