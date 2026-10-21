# Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models

A notebook-first, reproducible study of multi-step traffic-speed forecasting on **METR-LA** (207 Los Angeles highway sensors, 5-minute data, March - June 2012).
It asks whether using the road graph helps forecasting compared with models that treat every sensor independently, and it ships an *optional, simulation-only* signal-control extension.

> **No result numbers are written in this README.** All metrics are produced by the executed notebooks, saved as CSV files in `artifacts/metrics/` and summarised in
> [`reports/results_summary.md`](reports/results_summary.md). Anything that was not run is labelled as such there.
> Signal control is **simulated only**; nothing in this project talks to real traffic infrastructure or live feeds.

## State of the delivered run

* **Executed in full mode:** notebooks 00-09 including 06b (optional DCRNN-style comparison) and 09 (optional SUMO simulation, SUMO 1.28 installed with pip) with real data. All unit/structure tests pass (`python -m pytest tests -q`).
* **Three training runs (seeds 42, 1, 2) of the ST-GAT and of its no-edge ablation** (notebook 06, `EXTRA_SEEDS = (1, 2)`). Notebook 07 reports the seed spread and per-seed paired comparisons; the main tables use seed 42. The bootstrap intervals cover test-day variation; the seed section covers training randomness (3 seeds is a small sample).
* **Training happened in earlier executions;** the delivered notebooks 05, 06 and 06b default to `RETRAIN = False`, i.e. they reload the saved models and recompute everything downstream (set `RETRAIN = True` to retrain; expect ~15-20 min for XGBoost, ~45 min per ST-GAT variant, ~25 min for DCRNN on an Apple M3). Original training logs are printed on reload and kept in `artifacts/metrics/`.
* **Not run / not claimed:** the original DCRNN code (only a PyTorch re-implementation with a reduced training budget); any link between the METR-LA forecasts and the simulated signal controller (the simulation is a separate synthetic intersection); real traffic-signal control of any kind; anything beyond this city and these 119 days.
* Interpretation text that depends on results is checked against the tables in the same notebook; a warning is printed if a claim does not hold in your run (see `reports/experiment_notes.md`).

## Research question

> Does incorporating spatial relationships between connected traffic sensors improve multi-step traffic forecasting compared with models that treat each sensor independently?

## What is implemented

| Component | Where | Uses the graph? |
|---|---|---|
| Persistence, 60-min rolling mean, seasonal (weekday x time-of-day) average | notebook 04, `src/baselines.py` | no |
| XGBoost: one global model per horizon, per-sensor lag / rolling / calendar features, residual target | notebook 05, `src/xgboost_model.py`, `src/feature_engineering.py` | no |
| **ST-GAT**: GRU temporal encoder + 2 graph-attention blocks (attention restricted to road-graph neighbours) + multi-horizon head | notebook 06, `src/gat_model.py` | **yes** |
| **ST-GAT without edges** (identical network and parameter count, every edge removed): the cleanest test of the graph itself | notebook 06 | no |
| DCRNN-style diffusion-convolution seq2seq (optional, PyTorch re-implementation, *not* the original code) | notebook 06b, `src/dcrnn_model.py` | yes |
| Evaluation: common valid mask, MAE / RMSE / MAPE, per horizon, peak vs off-peak, per sensor, bootstrap intervals | notebook 07, `src/evaluation.py`, `src/metrics.py` | - |
| Geographic congestion maps (static + interactive) | notebook 08 | - |
| SUMO fixed-time vs queue-aware signal control on one synthetic intersection (optional, simulation only) | notebook 09, `simulation/` | - |

## Dataset

Required files (not committed; see [`data/README.md`](data/README.md) for sources, sizes and SHA-256 checksums):

* `data/raw/metr-la.h5` - speeds, 34,272 five-minute steps x 207 sensors (pandas `HDFStore`, key `df`).
* `data/raw/sensor_graph/graph_sensor_ids.txt`, `graph_sensor_locations.csv`, `distances_la_2012.csv` - from the [DCRNN repository](https://github.com/liyaguang/DCRNN).

The reported results use a copy of `metr-la.h5` downloaded from a third-party Hugging Face mirror (the original is only on Google Drive / Baidu); its structure, time grid and sensor order were validated
against the DCRNN files (notebooks 01 and 03), but it was **not** compared byte-for-byte with the original.

**Dataset limitations** (details in notebook 01): speed only (no flow / occupancy); missing readings are encoded as `0.0` (about 8% of cells, including whole-network gaps) and are
treated as missing, never as zero speed; 119 days of one city in 2012; 207 of ~4,000 sensors; speeds capped at 70 mph; naive timestamps (Pacific local time assumed); no incident, weather or holiday labels.
The test period contains noticeably more missing data than the training period.

## Installation

Python 3.10+ (developed on 3.13.6, Apple M3, 8 GB RAM).

```bash
cd traffic-flow-control
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

* **macOS:** if `import xgboost` fails with a `libomp.dylib` error, run `brew install libomp`.
* **Devices:** the graph models use CUDA if available, then Apple MPS, then CPU. Nothing assumes a GPU.
* To use the notebooks with this environment: `python -m ipykernel install --user --name traffic-flow-control` and select that kernel, or start Jupyter from the activated environment.

## Execution order

Run the notebooks in `notebooks/` in this order; each one reloads what it needs from `data/processed/` and `artifacts/`, so every notebook is restart-safe.

| # | Notebook | Needs | Typical time* |
|---|---|---|---|
| 00 | `00_environment_and_setup` | - | seconds |
| 01 | `01_data_exploration` | raw data | ~1 min |
| 02 | `02_data_preprocessing` | raw data | ~1 min |
| 03 | `03_graph_construction` | 02 | seconds |
| 04 | `04_baseline_models` | 02, 03 | ~1 min |
| 05 | `05_xgboost_forecasting` | 02, 03 (04 for the validation comparison) | ~15-20 min (four models on ~2.3M rows each) |
| 06 | `06_graph_attention_network` | 02, 03 | ~45 min per model (30 epochs x ~90 s on Apple MPS); ST-GAT + ablation = ~1.5 h; each extra seed adds ~1.5 h |
| 06b | `06b_optional_dcrnn` (optional) | 02, 03 | up to ~1 h (at most 30 epochs x ~2 min, run alone on the machine) |
| 07 | `07_model_evaluation` | 04, 05, 06 (06b optional) | ~2 min |
| 08 | `08_congestion_visualization` | 05, 06 | ~1 min |
| 09 | `09_optional_sumo_simulation` (optional) | - | ~1-2 min with SUMO (40 one-hour episodes); seconds without |

\*Measured on an 8 GB Apple M3 laptop; they depend strongly on hardware.
`notebooks/OneFileConsistofEverythingInCells.ipynb` is the same content in one file: all notebooks 00-09 (including 06b) concatenated in this order, **with the outputs of the executed runs**. It was assembled from the executed notebooks and was **not** re-executed as a single file; running it top to bottom works the same way (each section has its own setup cell) and, with the default `RETRAIN = False`, reloads the saved models.

To regenerate the one-file notebook after editing/re-executing any notebook: `python tools/merge_notebooks.py` (it only concatenates the executed notebooks; nothing is re-run).

Run headless from the project root, for example:

```bash
cd notebooks
for nb in 00_environment_and_setup 01_data_exploration 02_data_preprocessing 03_graph_construction 04_baseline_models \
          05_xgboost_forecasting 06_graph_attention_network 07_model_evaluation 08_congestion_visualization 09_optional_sumo_simulation; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 $nb.ipynb
done
```

**Smoke mode.** `TFC_RUN_MODE=smoke` (set *before* starting Jupyter) shrinks every expensive step (tiny subsets, 2 epochs). Smoke outputs get a `_smoke` suffix and are **never** results.
Tests: `python -m pytest tests -q`.

## Graph construction

The graph is DCRNN's: directed road-network distances `d_ij` (metres, `distances_la_2012.csv`) restricted to the 207 sensors, `sigma` = std of the finite distances,
`A_ij = exp(-(d_ij / sigma)^2)`, entries below 0.1 set to 0. `A_ij > 0` is an edge `i -> j`; the graph is directed and **not** symmetrised. Sensors are matched to the
traffic columns by id (never by row position); the canonical sensor order is fingerprinted and every later loader asserts the fingerprint. Geographic coordinates are used only for maps and as
static features: proximity is not connectivity (notebook 03 quantifies this). The shipped `adj_mx.pkl` is deliberately not unpickled.

## Experimental design

* **Chronological split** at day boundaries: 83 days train / 12 validation / 24 test (about 70 / 10 / 20 %), exact timestamps in `data/processed/splits.json`. A sample belongs to the split containing *all* its targets; its 60-minute input window may reach back into the preceding split (past context only).
* **Windows:** 12 input steps (60 min) -> forecasts at 1, 3, 6, 12 steps (5, 15, 30, 60 min); all in `src/config.py`.
* **Missing data:** missing targets are excluded from every loss and metric; inputs keep an observed-mask; nothing is imputed from the future.
* **Scaling:** z-score fitted on training timestamps only.
* **Shared anchor / residual formulation:** persistence predicts the last observed speed in the window; XGBoost, ST-GAT and DCRNN predict the *change* from that anchor, so they are compared on what they add beyond persistence.
* **Fair comparison:** identical origins, horizons, targets and scoring code for every model; validation (never test) is used for early stopping and checkpoint selection; predictions of all models are clipped to 0-70 mph before scoring.
* **Peak / off-peak** fixed in advance: weekdays 07:00-10:00 and 15:00-19:00 (judged on the forecast target time); all other hours are off-peak.

## Metrics

MAE and RMSE in mph; MAPE in % for true speeds >= 1 mph (near-zero policy, with its own valid count). All metrics are pooled over valid points; every table reports `n_valid`.
`improvement = 100 x (error_reference - error_new) / error_reference`: **positive means the new model has lower error.** Intervals come from a day-block bootstrap.

## Generated artifacts

| Folder | Content |
|---|---|
| `data/processed/` | cleaned data, split boundaries, scaler, validated adjacency and sensor order |
| `artifacts/models/` | XGBoost JSON models, ST-GAT / DCRNN checkpoints (`.pt`) |
| `artifacts/predictions/` | `[samples, horizons, sensors]` forecasts in mph per model and split |
| `artifacts/metrics/` | metric tables (CSV), training histories and logs |
| `artifacts/figures/` | all figures (PNG) and interactive maps (HTML) |
| `reports/` | `results_summary.md` (generated from the metrics) and `experiment_notes.md` (decisions, deviations, problems) |

Large generated files are git-ignored.

## Computational requirements

An 8 GB laptop is sufficient for the main pipeline, **but run the training notebooks one at a time**: the ST-GAT (~3-4 GB on MPS), XGBoost (~2 GB) and DCRNN (~2.5 GB) together exceed 8 GB and the machine starts swapping, which made epochs 5x slower in our runs. XGBoost training peaks at a few GB of RAM; the ST-GAT has ~69k parameters and needs ~3 GB on MPS at batch size 64.
The optional DCRNN is memory-hungry because of its unrolled recurrent graph (use a smaller batch size on small machines).

## Reproducibility

Seeds (`config.SEED = 42`) are applied to Python, NumPy and PyTorch; all hyper-parameters are in `src/config.py` and are also stored next to each model.
GPU kernels (MPS / CUDA) are not guaranteed to be bit-wise deterministic, so repeated runs can differ slightly; notebook 06 can train extra seeds (`EXTRA_SEEDS`) and notebook 07 then reports the spread.
The raw data checksums are in `data/README.md`.

## Optional SUMO extension

`simulation/` contains a synthetic 4-way intersection, seeded Poisson demand, a fixed-time and a queue-aware policy, and a TraCI runner; the signal-safety logic
(min/max green, yellow, all-red, no conflicting greens) is enforced by a state machine and unit-tested, and every simulation step checks that SUMO shows exactly the
controller's signal state. Install SUMO with `pip install eclipse-sumo traci` (no admin rights) and run notebook 09; it compares the policies over 20 seeds with paired
statistics. This is a **simulation only**: METR-LA contains no intersection data, the simulated intersection is not part of the METR-LA network, the forecasts are not fed into the
controller, and the fixed-time baseline was not tuned to the (asymmetric) demand. See `simulation/README.md`.

## Known limitations

* One city, 119 days, speeds only; conclusions may not transfer.
* Only three seeds per neural model (the ST-GAT and its ablation) and one for the optional DCRNN-style model; the bootstrap covers test-day variation, the seed section training randomness.
* The test period has much more missing data than the training period.
* XGBoost treats sensors independently by design (no neighbour features), so XGBoost vs ST-GAT mixes the effect of the graph with the effect of a different model family; the no-edge ablation isolates the graph.
* The dense masked attention layer is verified against PyTorch Geometric's `GATConv` numerically, but PyG is not used for training.
* The DCRNN-style model is a re-implementation with a reduced training budget, not the original.
* Attention weights are not causal evidence.

## Project layout

```
README.md  requirements.txt  .gitignore
notebooks/   00 ... 09 (+ 06b) and the combined OneFileConsistofEverythingInCells.ipynb
src/         config, data_loader, preprocessing, graph_utils, feature_engineering, baselines, xgboost_model,
             gat_model, dcrnn_model, metrics, evaluation, visualization
tests/       data alignment, graph, metrics, splits / leakage, notebooks structure, GAT, DCRNN, signal policies
tools/       merge_notebooks.py (builds the one-file notebook from the executed ones)
data/        raw/ (not committed), processed/, README.md
artifacts/   models/ predictions/ figures/ metrics/
reports/     results_summary.md, experiment_notes.md
simulation/  signal_policies.py, run_sumo_experiment.py, sumo_config/, README.md
```
