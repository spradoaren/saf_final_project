# SAF Final Project: Regime-Switching Models for SPY Volatility

## Overview

End-to-end study of regime-switching models for forecasting SPY daily
realized volatility (and short-term returns). Three regime-switching
tracks: Gaussian HMM, Markov-Switching AR, Input-Output HMM are
benchmarked against ARMA, HAR-RV, and GARCH(1,1) baselines under an
expanding-window out-of-sample protocol with MSE / MAE / QLIKE and
the Diebold-Mariano test.

## Repository Layout

```
data_preprocessing/
    data_adapter.py            Canonical YFinance adapter (parquet cache,
                               MultiIndex (Price, Ticker), auto_adjust=True).
    price_utils.py             extract_adjusted_close helper.
01_Data_Preprocessing_EDA/
    data_adapter.ipynb         Fetches SPY, writes train/test CSVs.
    spy_in_sample_return_eda.ipynb
                               Normality / stationarity / ARCH / leverage EDA.
ARMA/
    spy_return_benchmark_model.ipynb
                               M0 mean, M1 RW, M2 AR(p), M3 ARMA(p,q),
                               rolling 1-step forecast, Theil's U.
HMM_weekly_vol/
    model_utils.py             hmmlearn GaussianHMM wrapper, IC helpers,
                               rolling in-sample IC, backtest with naive +
                               AR(1) benchmarks.
    results.ipynb              Consolidated notebook for the weekly-vol
                               HMM track.
HMM_daily_return/
    return_regime_utils.py     hmmlearn GaussianHMM wrapper for daily
                               return regimes plus regime persistence,
                               stress alignment, and interpretability
                               diagnostics.
    model.ipynb                Daily-return regime walkthrough.
Markov_Switching_AR/
    model_utils.py             statsmodels MarkovAutoregression wrapper,
                               grid search, rolling forecast.
    results.ipynb              Consolidated notebook for the MS-AR track.
utils/
    metrics.py                 Shared rmse / mae / mse / directional
                               accuracy / qlike / dm_stat.
    diagnostics.py             Shared summarize_regimes (DataFrame and
                               fitted-model dispatch).
IOHMM/
    regimes/features.py        build_vol_iohmm_dataset: external features
                               and y = log(r_{t+1}^2 + eps).
    regimes/iohmm.py           GaussianIOHMM: linear Gaussian emissions,
                               softmax transitions with reference-state
                               identifiability, EM with multistart, BIC/ICL.
    regimes/diagnostics.py     summarize_regimes, check_regime_stationarity.
    experiments/spy_vol_regime.py
                               K sweep + expanding-window evaluation against
                               HAR-RV (RidgeCV) and GARCH(1,1).
    experiments/spy_vol_regime_analysis.ipynb
                               Result presentation.
```

## Installation

- Python 3.10 or newer.
- `pip install -r requirements.txt`

Required packages (minimum versions):

```
numpy>=1.24
pandas>=2.0
scipy>=1.10
scikit-learn>=1.3
statsmodels>=0.14
hmmlearn>=0.3
arch>=6.2
yfinance>=0.2.40
matplotlib>=3.7
pyarrow>=14.0
```

If running the HMM module with multithreaded BLAS, set
`OMP_NUM_THREADS=1` before launching Python (e.g., `export
OMP_NUM_THREADS=1`); `hmmlearn` is sensitive to nested parallelism.

## Data

- Source: Yahoo Finance via `yfinance`, with `auto_adjust=True` (Close
  is split- and dividend-adjusted).
- Tickers: SPY (volatility target), TLT, HYG, UUP, GLD (external
  regime features for IOHMM); ^GSPC for the HMM track.
- Default date range: 2017-06-01 to 2025-01-31 for IOHMM;
  2019-01-01 to 2026-01-01 for HMM; 2019-01-01 to 2025-01-01 for MS-AR.
- Cache: `data/yfinance_cache/<TICKER>_<md5>.parquet`, one file per
  (ticker, start_date, end_date) tuple.

## Models

### ARMA Benchmarks (M0–M3)

- Inputs: SPY daily log returns.
- Targets: next-day log return.
- Estimation: M0 unconditional mean; M1 random walk; M2 AR(p) with p
  selected by AIC over a (0..15) grid (selected p=9); M3 ARMA(p,q) with
  (p,q) selected by AIC over a (0..15)^2 grid (selected ARMA(9,12)).
- Evaluation: rolling one-step-ahead forecast; metric Theil's U.

### Gaussian HMM

- Inputs: ^GSPC weekly OHLC log returns (multivariate, 4 features).
- Estimation: `hmmlearn.GaussianHMM(covariance_type="full")` via
  Baum-Welch (`n_iter=100`, `tol=1e-4`).
- K selection: `rolling_ic_train_window` reports AIC, BIC, HQC, CAIC
  on overlapping in-sample windows.
- Evaluation: `backtest()` runs one-step-ahead expectation
  `transmat[state_now] @ means` and reports `mse_hmm` against `mse_naive`
  (last observed) and `mse_ar1` (AR(1) fit on the trailing 252 training
  observations, per feature).

### Markov-Switching AR

- Inputs: SPY daily log returns (`Close` from canonical adapter, which
  is adjusted).
- Estimation: `statsmodels.tsa.regime_switching.MarkovAutoregression`
  with `switching_ar=False` (AR coefficients common across regimes —
  Hamilton 1989 simplification), `switching_variance=True`, `trend="c"`.
- K selection: `grid_search_msar` sweeps (k_regimes, order,
  switching_variance) and selects by BIC then AIC. Currently restricted
  to `k_regimes=2` (an explicit `NotImplementedError` is raised
  otherwise to flag the limitation).
- Evaluation: `rolling_forecast_msar` refits at each step;
  `evaluate_forecasts` reports MSE, MAE, RMSE, directional accuracy.

### Input-Output HMM (IOHMM)

- Inputs: external features only — for each of {TLT, HYG, UUP, GLD},
  lag-1 return, lag-1 log realized vol (5-day rolling vol, annualized),
  and lag-5 cumulative return.
- Target: `y_t = log(r_{t+1}^2 + 1e-8)` (log realized variance proxy).
- Estimation: EM in log space (`logsumexp`) with `n_init=10` random
  restarts, sigma² floor `max(1e-4, 0.01 * Var(y))`, EM-divergence
  guard, and softmax reference-state identifiability
  (`W[:, K-1, :] = 0`). Post-fit states are reordered by ascending σ²
  with consistent permutation of betas, transition tensor, and π.
- K selection: K∈{2, 3, 4} fit on the initial training window;
  ICL = BIC + 2·Entropy(γ) selects the final K. Free-parameter count
  is `(K-1)·K·(F+1) + K·(F+1) + (K-1)`.
- Evaluation: expanding window with `MIN_TRAIN=504`, `REFIT_FREQ=21`.
  IOHMM forecasts use `forecast()` (forward filter + one-step transition
  through `x_next`). Baselines: HAR-RV (RidgeCV with α∈{0.01, 0.1, 1,
  10, 100}, cv=5) and GARCH(1,1) refit at every test day. Metrics:
  MSE, MAE, QLIKE, Diebold-Mariano vs HAR and vs GARCH; per-regime
  metrics by argmax filtered state probability.

## How to Run

End-to-end sequence (from repository root):

1. `01_Data_Preprocessing_EDA/data_adapter.ipynb` — populate the cache
   and write `spy_train_*.csv` / `spy_test_*.csv`.
2. `01_Data_Preprocessing_EDA/spy_in_sample_return_eda.ipynb` — EDA.
3. `ARMA/spy_return_benchmark_model.ipynb` — ARMA baselines.
4. `HMM_weekly_vol/results.ipynb`.
5. `Markov_Switching_AR/results.ipynb`.
6. `python -m IOHMM.experiments.spy_vol_regime` — run the IOHMM
   expanding-window experiment.
7. `IOHMM/experiments/spy_vol_regime_analysis.ipynb` — render results.

## Outputs

Written from the IOHMM experiment to the current working directory
(typically `IOHMM/experiments/`):

- `spy_vol_iohmm_results.csv` — per-day OOS predictions (`date`,
  `y_true`, `y_iohmm`, `y_har`, `y_garch`, `dom_state`).
- `spy_vol_iohmm_metrics.csv` — MSE, MAE, QLIKE per model.
- `spy_vol_iohmm_per_regime.csv` — IOHMM MSE and QLIKE conditional on
  the dominant filtered state.
- `spy_vol_iohmm_kselect.csv` — log-likelihood, BIC, ICL across K.

Other tracks emit results inside their notebooks; the EDA notebook
writes train/test CSVs under `01_Data_Preprocessing_EDA/`.

Adapter cache: `data/yfinance_cache/<TICKER>_<md5>.parquet`.

## Statistical Notes

- **EM initialization sensitivity.** EM is non-convex; the IOHMM uses
  `n_init=10` random restarts and keeps the run with the highest final
  log-likelihood. Runs whose log-likelihood decreases beyond `tol` are
  discarded as numerically diverged. The HMM track relies on
  `hmmlearn`'s default initialization and a single fit; multiple
  restarts are recommended if results are sensitive.
- **ICL vs BIC.** ICL = BIC + 2·Entropy(γ) penalizes weakly separated
  regimes. The IOHMM K is selected by ICL; BIC is reported alongside.
  The free-parameter count `(K-1)·K·(F+1) + K·(F+1) + (K-1)` reflects
  softmax reference-state identifiability and the (K-1) free entries
  in π.
- **Forward filter vs smoother.** IOHMM `forecast()` uses the forward
  filter only — it conditions on `y_{1:T}` and `x_{1:T}` and propagates
  one step through the input-dependent transition at `x_next`, with no
  future-y leakage. `predict_state_proba` defaults to `smoothed=False`
  (forward filter); `smoothed=True` runs full forward-backward and is
  intended for in-sample regime visualization only.
- **GARCH single-step refitting protocol.** GARCH(1,1) is refit at
  every test day inside the expanding-window inner loop and forecasts
  one step ahead. Returns are passed in percent (×100) for numerical
  conditioning; the variance forecast is rescaled by 1e-4 to convert
  back to fractional units before mapping to the log realized variance
  scale.
- **MS-AR `switching_ar=False` limitation.** AR coefficients are
  constrained to be common across regimes; only the intercept and
  variance switch. This is a deliberate Hamilton 1989 simplification.
  `k_regimes>2` with `switching_ar=False` is untested and raises
  `NotImplementedError`; use `k_regimes=2` or set `switching_ar=True`.
- **HMM backtest unconditional-mean caveat.** `backtest()` forecasts
  next-step OHLC log returns as `transmat[state_now] @ means`, where
  `means` is the unconditional regime mean. For zero-mean log returns
  this baseline is only weakly informative; the function therefore
  reports MSE alongside two trivial benchmarks (`mse_naive` and
  `mse_ar1`) so the result can be interpreted in relative terms.

## References

- Zucchini, W. and MacDonald, I. L. (2009). *Hidden Markov Models for
  Time Series: An Introduction Using R*. Chapman & Hall/CRC.
- Hamilton, J. D. (1989). A new approach to the economic analysis of
  nonstationary time series and the business cycle. *Econometrica*
  57(2), 357–384.
- Bengio, Y. and Frasconi, P. (1996). Input-output HMMs for sequence
  processing. *IEEE Transactions on Neural Networks* 7(5), 1231–1249.
- Corsi, F. (2009). A simple approximate long-memory model of realized
  volatility. *Journal of Financial Econometrics* 7(2), 174–196.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect
  volatility proxies. *Journal of Econometrics* 160(1), 246–256.
- Ang, A. and Timmermann, A. (2012). Regime changes and financial
  markets. *Annual Review of Financial Economics* 4, 313–337.
- Guidolin, M. (2011). Markov switching models in empirical finance.
  *Advances in Econometrics* 27B, 1–86.
