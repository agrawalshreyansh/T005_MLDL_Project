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
├── train.py             one training run: python -m deep_learning.train --model st_gat --seed 42
├── kaggle/              push training runs to a Kaggle GPU and fetch the results
├── notebooks/           10-15: the step-by-step story
├── dashboard/app.py     streamlit run deep_learning/dashboard/app.py
├── presentation/        self-contained HTML deck
└── tests/               shapes, leakage, masks, tiny-overfit checks
```

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
| 1 | Frame the question and the number to beat | Phase 1's XGBoost and XGBoost + neighbour features set the bar per horizon | `reports/phase1_results_summary.md` |
| 2 | Freeze the data contract | Same split, scaler and sensor order on every machine; checked, not trusted | `prepare_data.py` |
| 3 | Turn data into tensors | Windows `[batch, 12 steps, 207 sensors, 6 features]`, targets `[batch, 12, 207]`, plus a mask | `dl/data.py` |
| 4 | Prove there is no leakage | Tests that no input or target crosses a split boundary | `tests/` |
| 5 | Write one training loop for every model | Loss, optimiser, early stopping, checkpoints, logs | `dl/engine.py` |
| 6 | Start with the simplest model | GRU over time only | `dl/models/temporal.py` |
| 7 | Overfit one batch before a long run | If a model can't memorise 8 samples, the bug is in the code, not the data | `tests/` |
| 8 | Add one idea at a time | + graph attention, then diffusion convolution | `dl/models/` |
| 9 | Ablate | Same network with the graph switched off isolates what the graph adds | notebook 12 |
| 10 | Evaluate once, on test, fairly | Common mask, per-horizon, peak vs off-peak, bootstrap intervals | notebook 14 |
