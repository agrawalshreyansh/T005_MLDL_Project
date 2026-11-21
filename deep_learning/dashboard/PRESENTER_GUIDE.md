# Dashboard presenter guide (about 5 minutes)

## Before you present

```bash
cd ~/Desktop/DL/Project/T005_MLDL_Project
source .venv/bin/activate
streamlit run deep_learning/dashboard/app.py
```

* Opens at **http://localhost:8501**. Wait until the sidebar lists the loaded models (~10 s on first load).
* Open it **before** the talk and click through every page once, so everything is cached.
* Zoom the browser to 90% if the maps do not fit the projector.
* Every page opens on **ST-GAT**, our best model.

## One-line story

> Traffic jams spread along roads. A model that listens to upstream and downstream sensors
> (ST-GAT) forecasts better than models that look at each road alone, and switching the graph off removes the gain.

## Demo flow

| # | Page | What to do | What to say |
|---|---|---|---|
| 1 | **Replay** (≈90 s) | Scroll down, press **Play**. Pause around 08:30–09:00 or 16:30–17:30. | "Left: what the sensors measured. Right: what ST-GAT predicted for that moment, 30 minutes earlier. When the left map turns red, the right map should already be red. The title shows the network-wide error each frame." |
| 2 | **Network map** (≈45 s) | Switch *Colour by* to **error**. Drag the time slider through the evening rush. | "Grey means the sensor was offline: we never score those. Red is forecast too slow, blue too fast. Errors are largest where traffic is changing fast, at the edges of a jam." |
| 3 | **Leaderboard** (≈60 s) | Show the MAE table, then the **Peak vs off-peak** bars at 60 min. | "Same 6,901 test windows for every model. At 60 minutes ST-GAT is 3.46 mph, XGBoost with neighbours 3.65, persistence 5.35. Rush hour is where it matters: there ST-GAT is 8.6% better than XGBoost." |
| 4 | **Graph lens** (≈45 s) | Keep the first sensor (most connected). | "Blue: upstream sensors, orange: downstream. Dot size is how much ST-GAT listens to each one. It learned this; we never told it. A sensor keeps only about 15% of the attention for itself." |
| 5 | **Training** (≈30 s) | Leave the default runs selected. | "All trained on a Kaggle T4. Early stopping on validation only; the test set was used once, at the end." |
| 6 | **Sensor explorer** (optional) | Pick a sensor, show XGBoost vs ST-GAT for one day. | "One sensor, one day: both follow normal traffic; the difference is in how fast they react to a slowdown." |

## Numbers to remember

| | |
|---|---|
| Data | 207 sensors, 5-min speeds, 119 days (Mar–Jun 2012); 8.1% readings missing |
| Split | 83 / 12 / 24 days, chronological, cut at midnight |
| ST-GAT, 60 min | 3.46 mph MAE (XGBoost + neighbours 3.65, persistence 5.35) |
| vs XGBoost + neighbours | −5.3% at 60 min (95% CI 4.3–6.2), −8.6% in rush hour |
| Edges removed | 3–5% worse at every horizon; seed spread ~0.01 mph |
| Time-only GRU | ties XGBoost: the network alone is not the win, the graph is |
| Graph helps at | 76% of sensors |

## Likely questions

* **"Is the right map a live prediction?"** No. Forecasts were made by the trained model on the held-out test days and saved; the dashboard replays them. Nothing controls real traffic.
* **"Why are some dots grey?"** Sensor outage (reading 0). Masked in training and never scored.
* **"How do you know the graph causes the gain?"** Same network, same parameters, edges removed: 3–5% worse, bootstrap interval excludes zero.
* **"Why not DCRNN?"** It is just as good up to 30 min; ST-GAT is slightly better at 60 min and about 3× faster per training epoch.
* **"Does attention explain the forecast?"** It shows where the model looks, not why it is right.

## If something goes wrong

| Problem | Fix |
|---|---|
| Error about missing prediction files | run on the laptop that has `artifacts/predictions/` (not in git) |
| Port 8501 busy | `streamlit run deep_learning/dashboard/app.py --server.port 8502` |
| Maps blank (no internet) | map tiles need internet; dots and colours still work. Fall back to `deep_learning/presentation/presentation.pdf` |
| Replay slow | set **Frame every** to 15 min |
