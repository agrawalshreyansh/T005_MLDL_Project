# Deep learning on METR-LA

Phase 2 of the project. Question: **does letting each road sensor see its upstream and
downstream neighbours (a graph neural network) forecast speed better than models that
treat every sensor on its own?**

Everything here reuses the Phase 1 data pipeline in `src/` (cleaning, split, scaler,
adjacency, metrics), so deep-learning and Phase 1 models are scored on exactly the same
test points.

## Layout

```
deep_learning/
├── prepare_data.py      raw files -> data/processed (same steps as notebooks 02 + 03)
├── dl/                  data windows, training engine, models
├── train.py             training runs: python -m deep_learning.train --models st_gat --seeds 42 --lr 0.001 --width 128
├── kaggle/              push training runs to a Kaggle GPU and fetch the results
├── notebooks/           10-15: the step-by-step story
├── dashboard/app.py     streamlit run deep_learning/dashboard/app.py
├── presentation/        self-contained HTML deck
└── tests/               shapes, leakage, masks, tiny-overfit checks
```

## Result in one table

Test MAE in mph on the last 24 days (6,901 windows, common mask; DL = seed 42). Full tables, seeds,
peak/off-peak and bootstrap intervals: [`reports/dl_results_summary.md`](../reports/dl_results_summary.md).

| Model | 5 min | 15 min | 30 min | 60 min |
|---|---|---|---|---|
| Persistence | 2.82 | 3.49 | 4.19 | 5.35 |
| XGBoost (Phase 1, re-run) | 2.31 | 2.84 | 3.26 | 3.69 |
| XGBoost + neighbours (Phase 1, re-run) | 2.23 | 2.76 | 3.18 | 3.65 |
| GRU, time only | 2.32 | 2.83 | 3.22 | 3.63 |
| ST-GAT, edges removed | 2.31 | 2.81 | 3.19 | 3.57 |
| **ST-GAT** | 2.22 | 2.67 | **3.04** | **3.46** |
| DCRNN | **2.21** | **2.67** | 3.05 | 3.49 |

* The graph is what helps: the same ST-GAT with every edge removed loses 3–5% at every horizon (bootstrap CI excludes 0); seed spread is ~0.01 mph.
* Without a graph, a neural network only ties XGBoost.
* ST-GAT vs XGBoost + neighbours at 60 min: 5.3% lower MAE (95% CI 4.3–6.2); in rush hour 8.6%.

## Notebooks

| # | Notebook | Question it answers |
|---|---|---|
| 10 | `10_dl_data_pipeline` | How does the speed table become `[B, 12, 207, 6]` tensors, and is that correct? |
| 11 | `11_temporal_gru` | Without a graph, does a neural network beat XGBoost? |
| 12 | `12_st_gat_and_ablation` | Does graph attention help, and is it the edges or the extra parameters? |
| 13 | `13_dcrnn` | How does diffusion on the directed graph compare? |
| 14 | `14_dl_evaluation` | All models, one test set, error bars; writes the results summary |
| 15 | `15_congestion_maps` | Where and when does the graph help? |

Notebooks load saved weights and predictions; none of them trains. They run in a few minutes on a laptop.

## Reproduce

```bash
# 1. data (same split and scaler as Phase 1; checked against the committed json files)
python -m deep_learning.prepare_data --download

# 2. training on a Kaggle T4 (needs ~/.kaggle/access_token; the branch must be pushed)
python -m deep_learning.kaggle.launch push --slug metr-dl-search --models st_gat st_gat_noedge -- --lr 0.0003 0.001 0.003 --width 64 128 --max-epochs 25
python -m deep_learning.kaggle.launch push --slug metr-dl-dcrnn  --models dcrnn
python -m deep_learning.kaggle.launch push --slug metr-dl-final  --models gru st_gat st_gat_noedge --seeds 42 1 2 -- --lr 0.001 --width 128
python -m deep_learning.kaggle.launch fetch --slug <slug>       # copies models, predictions, histories into artifacts/

# 3. notebooks 10-15 in order, then the deck
python -m deep_learning.presentation.build                     # writes presentation/index.html
streamlit run deep_learning/dashboard/app.py
```

Kaggle allows one GPU session at a time, so the jobs queue. Measured on the T4: ST-GAT ~56 s per epoch,
no-edge twin ~42 s, GRU ~34 s, DCRNN ~159 s. Total GPU time for everything above: about 8 hours.
Training the same runs on an 8 GB Apple M3 would take 3–5× longer.

## Limits

* One city, 119 days in 2012, speed only. No claim beyond METR-LA.
* Phase 1 XGBoost was re-run so every model is scored on identical points; the re-runs are 0.01–0.05 mph worse than the committed Phase 1 numbers (cause not traced).
* DCRNN is our re-implementation (one seed, default size), not the authors' code.
* Attention weights describe where the model looks, not why it is right.
* Graph WaveNet-lite (optional rung 4) was not built: the ablation already answers the brief, and both graph models sit in the published range.
* Nothing controls real traffic; signal control in `phase2/simulation/` is simulated only.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r deep_learning/requirements-dl.txt
python -m deep_learning.prepare_data            # add --download on a fresh machine
python -m pytest deep_learning/tests -q
```

## How a deep-learning project is built

The commits on this branch follow these steps in order, so `git log` doubles as a tutorial.

| # | Step | What it means here | Where |
|---|---|---|---|
| 1 | Frame the question and the number to beat | Persistence, seasonal average, XGBoost and XGBoost + neighbour features set the bar per horizon | `notebooks/04`, `05`, `05b` |
| 2 | Freeze the data contract | Same split, scaler and sensor order on every machine; checked, not trusted | `prepare_data.py` |
| 3 | Turn data into tensors | Windows `[batch, 12 steps, 207 sensors, 6 features]`, targets `[batch, 12, 207]`, plus a mask | `dl/data.py` |
| 4 | Prove there is no leakage | Tests that no input or target crosses a split boundary | `tests/` |
| 5 | Write one training loop for every model | Loss, optimiser, early stopping, checkpoints, logs | `dl/engine.py` |
| 6 | Start with the simplest model | GRU over time only | `dl/models/temporal.py` |
| 7 | Overfit one batch before a long run | If a model can't memorise 8 samples, the bug is in the code, not the data | `tests/` |
| 8 | Add one idea at a time | + graph attention, then diffusion convolution | `dl/models/` |
| 9 | Ablate | Same network with the graph switched off isolates what the graph adds | notebook 12 |
| 10 | Evaluate once, on test, fairly | Common mask, per-horizon, peak vs off-peak, bootstrap intervals | notebook 14 |
