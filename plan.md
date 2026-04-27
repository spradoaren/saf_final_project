## What needs to be done

### Task 0: Verify HMM/features.py runs in this repo

Now that HMM/features.py is in the repo, verify it imports cleanly:

    .venv/bin/python -c "from HMM.features import GKVolFeatures, RVForecastWF, walk_forward_har_rv; print('ok')"

If that fails, fix the import path issue (probably need HMM/__init__.py).

### Task A: Run the canonical HMM forecasting pipeline

File: new script `experiments/hmm_canonical_run.py` (or run her notebook).

Run RVForecastWF with order=1, n_states=3, train_window=252, refit_every=21
on SPY 2019-01-01 to 2024-12-31. Save predictions to a CSV with columns
(date, y_true_rv_gk, y_hmm_pred). Also run walk_forward_har_rv on the same
target. Save (date, y_har_pred) similarly.

This produces the canonical track's reference outputs.

### Task B: Realign IOHMM to forecast rv_gk at horizon-1

File: IOHMM/experiments/spy_vol_regime.py

Change the target from build_log_rv_target(close, horizon=5) to
GKVolFeatures.compute_gk(raw_df, ticker). Drop the horizon=5 5-step GARCH
averaging and revert to horizon=1 GARCH variance forecast.

Adapt the OOS evaluation period and refit cadence to match the HMM track:
2019-01-01 to 2024-12-31, rolling 252-day window, 21-day refit.

Output: CSV with (date, y_true_rv_gk, y_iohmm_pred, y_har_pred, y_garch_pred).

### Task C: Realign MS-AR to forecast rv_gk at horizon-1

File: Markov_Switching_AR/model_utils.py

The current MS-AR forecasts log returns. This is incompatible with the
canonical target. Modify the data-prep function to feed it rv_gk instead
of returns. Verify MarkovAutoregression accepts a positive-valued target
(it should; it's just a Gaussian autoregression).

If MarkovAutoregression on rv_gk doesn't fit cleanly (numerical issues
with positive-only series), fit on log(rv_gk) instead and exp the
predictions.

Output: CSV with (date, y_true_rv_gk, y_msar_pred).

### Task D: Build unified comparison table

File: new script `experiments/unified_table.py`

Read all four CSVs. Inner-join on date. Verify y_true_rv_gk is bit-equal
across all sources. Compute RMSE, MAE, QLIKE, directional accuracy, MZ R^2
on the volatility-percentage scale (sqrt(rv) * 100) for each model.

Save to results/unified_metrics.csv.

## Order

Task 0: 5 minutes
Task A: 30 minutes (just running the existing code)
Task B: 1.5-2 hours (real code change in IOHMM)
Task C: 1-1.5 hours (MS-AR is a more invasive change)
Task D: 30 minutes

Total: ~4 hours of code work tomorrow morning.
