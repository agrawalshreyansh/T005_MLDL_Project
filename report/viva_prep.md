# Viva preparation (Phase 1: machine learning)

Short answers; every number is from `reports/phase1_results_summary.md` and the notebooks it cites.

## The problem and the data
**What are we predicting?** Average speed of each of 207 METR-LA sensors 5, 15, 30 and 60 minutes ahead, from the previous 60 minutes.
**Why is zero "missing"?** 8.11% of readings are exactly 0.0, the smallest positive reading is 0.33 mph, and in 2,148 steps all 207 sensors read zero at once, which cannot be real traffic (notebook 01). Zeros become missing values plus a mask.
**Why is the test period harder?** It has 12.0% missing cells against 7.2% in training, and 9.6% of test windows have no recent observation for the sensor (training: 5.5%).
**Why no flow or occupancy?** The dataset has speed only; we did not invent the other channels.

## Validation and leakage
**Why a chronological split?** Windows that end five minutes apart share 11 of 12 inputs; shuffling would leak. Split: 83 train, 12 validation, 24 test days, whole days, in time order.
**Where could leakage still hide?** Scaling statistics (fitted on training only), per-sensor means (training only), a seasonal-mean feature (deliberately not used because it contains the training target), and neighbour features (only readings at or before the origin). Tests corrupt validation/test data and check that nothing changes.
**Why a gap in rolling-origin cross-validation?** A training window's farthest target is 12 steps ahead; without the gap it could overlap the first validation targets.
**Why a day-block bootstrap?** Errors within a day are strongly correlated; resampling points would give intervals that are too narrow.

## Models and theory
**Why predict the residual from the last observed speed?** Every model is then judged on what it adds beyond persistence, and trees only learn the change.
**How does gradient boosting work?** Additive trees, each fitted to the loss left by the previous ones, shrinkage `eta`, regularised leaf weights; split gain uses gradient and Hessian sums `G`, `H` (see report Section IV).
**Why absolute-error loss for XGBoost?** It matches the MAE metric; its minimiser is the conditional median.
**Why does the forest win on RMSE but lose on MAE?** Squared-error splits estimate a conditional mean (good for RMSE); XGBoost's absolute-error objective estimates a median (good for MAE). This is consistent with the numbers but was not tested by swapping the objectives, so say "consistent with", not "proves".
**Why is the forest's variance lower than a single tree's?** Averaging B trees with pairwise correlation `rho` gives `rho*s2 + (1-rho)*s2/B`; feature subsampling lowers `rho`. Bagging does not reduce bias.
**Why did the brief say "XGBoost per segment" and you did not do that?** One global model per horizon shares strength across sensors (a sensor id, coordinates and mean speed let it specialise) and avoids 207 x 4 models of about 10^5 correlated rows each. We did not compare against per-segment models, which is a stated limitation.
**Why did tuning change so little?** The default was within about 0.4% of the best searched value on validation; the adopted configuration gained 0.54% in cross-validation and 0.1 to 1.3% on test.

## Results
**Headline?** Boosted trees lower MAE by 18% (5 min) to 32% (60 min) against persistence; neighbour features add 3.6% (5 min) to 1.8% (60 min), larger in peak hours; the forest is 4.5 to 13% worse on MAE but ties on RMSE.
**Which features matter?** The last observed speed and lags first; rolling statistics add nothing measurable (0.05% and 0.02%, inside seed noise of 0.001 and 0.004 mph); at 60 minutes calendar features (+9.1%) and sensor identity (+10.3%) matter far more than neighbours (+1.7%).
**Is the forest at a fair setting?** It was chosen from a six-setting validation grid, but the best setting is at the grid edge, so a larger forest might narrow the gap.
**What is not established?** One city and period; single fits (seed noise only measured for the ablation); the test period has more missing data; importances and ablations are not causal.

## Likely traps
* "Your MAPE looks worse for the forest but RMSE better: which model is best?" It depends on the metric; report all three with sample sizes.
* "Did you tune on the test set?" No: early stopping and every choice used validation or training-period cross-validation; the test period is scored once, in notebook 07a.
* "Is persistence a fair baseline?" It is the standard naive reference and is also the anchor the learned models start from.
* "Neighbour features: how many?" Eight (older text said ten; the code and the ablation, 30 to 38 columns, say eight).
