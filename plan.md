# SAF Final Project — Implementation Plan

## 1. High-Level Plan Overview

This plan corrects findings from the repository audit and unifies all model tracks under a single volatility target. It is sequenced in five phases. Phase 1 removes the bugs that invalidate current numerical results. Phase 2 makes evaluation procedures statistically defensible. Phase 3 retargets every model to the same realized-volatility quantity and adds the missing baselines. Phase 4 unifies the evaluation harness so cross-track comparisons become valid. Phase 5 establishes reproducibility and reliability infrastructure.

Phases must be applied in order. Phase 4 depends on Phases 1–3. Phase 5 can be partially applied early (e.g., `requirements.txt`) but the unification-driven refactors should follow Phase 4.

| Phase | Theme | Blocking Risk if Skipped |
|---|---|---|
| 1 | Critical correctness fixes | All current numbers are biased |
| 2 | Statistical corrections | Metrics and comparisons are not defensible |
| 3 | Model retargeting + baselines | Tracks predict different objects, not comparable |
| 4 | Evaluation framework unification | Cross-track claims are not supported |
| 5 | Refactoring & reliability | Project is hard to reproduce, no tests, brittle imports |

---

## 2. Target Convention

All models in this project predict the **same target**:

```
y_t = log(RV²_{t+1} + 1e-8)
```

where `RV²_{t+1}` is the **5-day forward realized variance**:

```
RV²_{t+1} = (1/5) · Σ_{i=t+1}^{t+5} r_i²
r_i = log(Close_i / Close_{i-1})
```

**Variance vs volatility.** Realized volatility is `RV = √RV²`; realized variance is `RV²`. They differ by a factor of 2 on the log scale (`log RV² = 2·log RV`) and are equivalent for model ranking. We target log-variance because (a) the QLIKE loss is defined on variance forecasts, (b) GARCH outputs variance natively (no awkward conversions), and (c) the Corsi HAR-RV / Andersen-Bollerslev literature works on the variance scale. The standard term "realized volatility forecasting" refers to this object.

Rationale for the 5-day window: a single squared daily return is unbiased but extremely noisy (`Var(r²) ≈ 2σ⁴`). A 5-day forward window smooths the noise while remaining a one-step volatility forecast at the model's decision horizon. The `+ 1e-8` and `log` are kept for numerical stability and approximate Gaussianity of the target distribution.

**This target is the single source of truth.** Every model — ARMA, MS-AR, HMM (weekly and daily), IOHMM, HAR-RV, GARCH, persistence — predicts this object. Return-prediction models (`r_{t+1}` forecasts) are kept only as an appendix for sanity-checking; they are not part of the headline cross-track comparison.

---

## 3. Phase-by-Phase Action Plan

### Phase 1 — Critical Fixes

#### 1.1 Define the canonical target builder
- **File:** new module `utils/targets.py`
- **Issue:** No shared target-builder exists; each track defines its own RV proxy. The IOHMM target uses a single squared return, which is noisy.
- **Action:** Create `build_log_rv_target(close: pd.Series, horizon: int = 5, eps: float = 1e-8) -> pd.Series` that returns `log((1/horizon) · Σ_{i=t+1}^{t+horizon} r_i² + eps)` indexed at time `t`. Add a complementary `build_log_rv_features(close, lags=(1,5,22))` for HAR-style lagged-RV features (rolling means of squared returns, log-transformed, shifted by 1).
- **Validation:** Unit test on a synthetic series with known constant volatility — output should be approximately constant at `log(σ²)`. NaN count at the tail equals `horizon`.

#### 1.2 Fix `rv_w` / `rv_m` definitions in IOHMM features
- **File:** `IOHMM/regimes/features.py:74-75`
- **Issue:** `rv_w = r.rolling(5).mean()` and `rv_m = r.rolling(22).mean()` compute rolling means of returns rather than rolling means of squared returns.
- **Action:** Change both to `rv_d.rolling(5).mean()` and `rv_d.rolling(22).mean()`. Replace the inline target construction with a call to `utils.targets.build_log_rv_target(close_target, horizon=5)`.
- **Validation:** `rv_w` and `rv_m` are non-negative for all input. Output of `build_vol_iohmm_dataset` matches the new `build_log_rv_target` for the target column.

#### 1.3 Fix GARCH target scale
- **File:** `IOHMM/experiments/spy_vol_regime.py:160-164`
- **Issue:** GARCH forecast is `0.5·log(σ²/1e4)` (i.e., `log σ`), not on the `log RV²` scale used by IOHMM and HAR.
- **Action:** Replace the conversion. GARCH(1,1) must forecast multi-step variance: call `garch_res.forecast(horizon=5)`, sum the 5 conditional variances, divide by 5 to match the rolling-mean form of `RV²`, divide by 1e4 to undo the `*100` rescaling, then emit `log(σ²_5step_avg + 1e-8)`. Document the unit chain in a comment.
- **Validation:** On any single date, print `(y_h_iohmm, y_h_har, y_h_garch, y_true)`. All four values must lie in the same numeric range (typically −12 to −6 for SPY). Mean of `y_h_garch` over the OOS period is within ±1 of the mean of `y_true`.

#### 1.4 Rewrite the broken HMM_weekly_vol notebook
- **File:** `HMM_weekly_vol/results.ipynb`
- **Issue:** Cell 5 references `data['Adj Close']` which does not exist in the canonical MultiIndex frame. Cell 27 refits inside the loop without preserving state-label identity.
- **Action:** Use `extract_adjusted_close(data, '^GSPC')` from `data_preprocessing.price_utils`. Replace the in-loop full refit with a rolling refit that warm-starts via `init_params=""` to keep state IDs stable. Replace the trading-rule cell's headline with a `log RV²` forecast plot.
- **Validation:** Notebook runs end-to-end against the canonical adapter. Across two consecutive refits, `model.means_` rows are within tolerance (state labels preserved).

#### 1.5 MS-AR rolling forecast: track failures explicitly
- **File:** `Markov_Switching_AR/model_utils.py:288-339`
- **Issue:** Failed forecasts insert `np.nan` and are silently dropped, biasing reported MSE if failures correlate with regimes.
- **Action:** Inside the loop, accumulate `(date, error_type, message)` for every failure. Return both a clean DataFrame and a `failures` DataFrame. Report MSE under both drop and forward-fill imputation.
- **Validation:** Output includes `n_failed`, `failures_df`, and two MSE values. If `n_failed == 0`, both MSE values are equal.

#### 1.6 HMM weekly `backtest()` — relabel or remove
- **File:** `HMM_weekly_vol/model_utils.py:96-129`
- **Issue:** Predicts `transmat[state_now] @ means`, which for zero-mean returns is essentially zero. Cannot meaningfully outperform any baseline.
- **Action:** Remove from the publishable pipeline. Replace with a `log RV²` forecast computed as `E[log RV² | filtered state] = (filtered_probs @ transmat) · regime_log_rv_means`, where `regime_log_rv_means` is the mean of the `log RV²` target conditional on each regime in the training set.
- **Validation:** New function returns predictions in the `log RV²` range. Old `backtest()` is either renamed with a clear "sanity-check only" docstring or deleted.

---

### Phase 2 — Statistical Corrections

#### 2.1 HAC-adjusted Diebold-Mariano
- **File:** `utils/metrics.py:43-45`
- **Issue:** `dm_stat` uses i.i.d. variance, no HAC adjustment, no p-value.
- **Action:** Replace with a function returning `(stat, p_value, h)`. Use Newey-West with truncation lag `h-1` and the Harvey-Leybourne-Newbold small-sample correction. Return a two-sided p-value from a t-distribution with `T-1` degrees of freedom.
- **Validation:** Synthetic equal-loss test → stat near zero, p > 0.5. Synthetic unequal-loss test → p < 0.05 for sufficient T.

#### 2.2 Held-out K-selection for HMM tracks
- **Files:** `HMM_weekly_vol/model_utils.py:54-67`, `HMM_daily_return/return_regime_utils.py:113-128`
- **Issue:** K is selected by in-sample IC only or implicitly fixed.
- **Action:** Add a function that fits each candidate K on a training window and scores on a validation window. Return `(K, train_ll, val_ll, BIC, ICL)`. Use ICL as the primary criterion to match the IOHMM track.
- **Validation:** Each HMM track produces a `K_selection.csv`. Selected K is reported alongside alternatives.

#### 2.3 Constrain ARMA grid
- **File:** `ARMA/spy_return_benchmark_model.ipynb`, cells 12 and 15
- **Issue:** Grid `(0..15)²` is overfit-prone with ~1500 daily obs.
- **Action:** Reduce to `(0..5)²`. Add validation-set log-likelihood as secondary criterion. Break ties by parsimony. Note: ARMA in Phase 3 fits on `log RV²`, not returns.
- **Validation:** Notebook prints AIC and validation-LL tables. Selected `(p, q)` is the joint AIC-then-validation winner.

#### 2.4 Log ARMA fallback rate
- **File:** `ARMA/spy_return_benchmark_model.ipynb`, cell 16
- **Issue:** Silent estimator switch from `innovations_mle` to `statespace` is not logged.
- **Action:** Track `fallback_count` and `fallback_dates`. Print at end of rolling forecast. Flag the model as unstable if fallback rate > 5%.
- **Validation:** Notebook output contains the fallback summary.

#### 2.5 Standardize the metrics surface
- **File:** `utils/metrics.py`
- **Issue:** Each track reports a different metric subset.
- **Action:** Add `compute_all_metrics(y_true, y_pred, baseline_pred=None) -> dict` returning `{mse, mae, rmse, qlike, r2_oos_vs_zero, ic, dm_stat_vs_baseline, dm_pvalue_vs_baseline}`. Every track must use this single function.
- **Validation:** All tracks emit the same metrics columns. A single concat produces a clean comparison table.

---

### Phase 3 — Model Retargeting and Baselines

All actions in this phase change the prediction target to `y_t = log(RV²_{t+1} + 1e-8)` from `utils.targets.build_log_rv_target`.

#### 3.1 Retarget ARMA M0–M3 to log RV²
- **File:** `ARMA/spy_return_benchmark_model.ipynb`
- **Issue:** ARMA currently fits on log returns; the project is about volatility.
- **Action:** Compute `y = build_log_rv_target(close, horizon=5)`. Fit M0 (mean of `y`), M1 (persistence: `y_hat_t = y_{t-1}`), M2 (AR(p) on `y`), M3 (ARMA(p,q) on `y`). Replace residual diagnostics to operate on `y` residuals. Optionally keep the original return-target run as an appendix labeled "Return-forecasting sanity check (not part of cross-track comparison)".
- **Validation:** Notebook predictions are in `log RV²` range. Forecast plot shows reasonable tracking of `y_true`. M1 (persistence) typically dominates M0 (mean).

#### 3.2 Retarget MS-AR to log RV²
- **File:** `Markov_Switching_AR/model_utils.py`
- **Issue:** MS-AR currently fits on log returns.
- **Action:** Replace `prepare_return_data` calls with a `prepare_log_rv_data` function that returns `y = build_log_rv_target(close, horizon=5)`. Fit `MarkovAutoregression(endog=y, k_regimes=2, order=1, switching_variance=True, trend="c", switching_ar=False)`. Adjust `one_step_forecast_from_result` documentation to reflect the new target.
- **Validation:** Predictions are in `log RV²` range. Filtered probabilities behave sensibly across known volatility events (e.g., March 2020).

#### 3.3 Retarget HMM_weekly_vol to log RV²
- **File:** `HMM_weekly_vol/model_utils.py`
- **Issue:** Track currently models multivariate OHLC log returns; predictions are unconditional means.
- **Action:** Keep OHLC log returns as the HMM observation features (so the regime structure is identified from market dynamics), but separate the **prediction target** from the **observation features**. Compute `y_target = build_log_rv_target(close, horizon=5)` aligned to the HMM observation index. Compute per-regime mean of `y_target` on training data. At forecast time, predict `E[y_target | filtered state] = (filtered_probs @ transmat) · regime_y_means`.
- **Validation:** Predictions in `log RV²` range. Per-regime means of `y_target` on training data are monotone in regime volatility.

#### 3.4 Retarget HMM_daily_return to log RV²
- **File:** `HMM_daily_return/return_regime_utils.py:182-207`
- **Issue:** Track predicts `r_{t+1}` regime-conditionally.
- **Action:** Replace `state_return_means = model.means_[:, 0]` with `state_y_means = (training y_target grouped by argmax filtered state).mean()`. Forecast `regime_pred = next_state_probs @ state_y_means`. Keep features (`return`, `abs_return`, `rv_21d_annualised`) for HMM fitting, change target only.
- **Validation:** Predictions in `log RV²` range. Comparable in scale to IOHMM.

#### 3.5 Universal baselines: persistence + HAR-RV
- **File:** new module `utils/baselines.py`
- **Issue:** Persistence (`y_hat_t = y_{t-1}`) does not exist anywhere. HAR-RV is buried inside the IOHMM experiment.
- **Action:** Implement `PersistenceBaseline` and `HARRVBaseline` (RidgeCV with `α ∈ {0.01, 0.1, 1, 10, 100}`, `cv=5`) as standalone classes with `.fit(X, y)` and `.predict(x_next)` interfaces. Both consume `build_log_rv_features` from `utils.targets`. Used by every track.
- **Validation:** Both baselines produce predictions in `log RV²` range. Persistence on `y_t` is `y_{t-1}` exactly.

#### 3.6 Multistart EM for hmmlearn HMMs
- **Files:** `HMM_weekly_vol/model_utils.py:15-29`, `HMM_daily_return/return_regime_utils.py:113-128`
- **Issue:** Single-seed fit; EM is non-convex.
- **Action:** Wrap each HMM fit in an `n_init` loop (default 5) with different `random_state` values. Keep the run with highest `model.score(X)`. Persist the winning seed.
- **Validation:** Two runs of the wrapper with the same `n_init` and seed sequence are bit-identical.

#### 3.7 State-label coherence in HMM_daily_return
- **File:** `HMM_daily_return/return_regime_utils.py:182-207`
- **Issue:** Full refit each step yields arbitrary state ordering.
- **Action:** After each refit, reorder states by ascending `means_[:, 0]` (return mean) and apply the same permutation to `transmat_` and posteriors.
- **Validation:** Per-state mean of `actual_return` is monotonically increasing in state ID across refits.

#### 3.8 Robust MS-AR parameter extraction
- **File:** `Markov_Switching_AR/model_utils.py:241-262`
- **Issue:** String-matching `f"const[{r}]"` / `f"intercept[{r}]"` is brittle across statsmodels versions.
- **Action:** Prefer `result.params_regime` where available. Fallback to a robust regex matching both `const[0]` and `const.regime[0]` patterns. Add a unit test pinning expected parameter names for the supported statsmodels version.
- **Validation:** Unit test fits a 2-regime MS-AR(1) on synthetic switching data and checks `intercepts.shape == (2,)`.

#### 3.9 Document IOHMM forecast convention
- **File:** `IOHMM/regimes/iohmm.py:480-509`
- **Issue:** Convention for `x_next` is undocumented.
- **Action:** Add a docstring section "Conventions" stating: `X[t]` drives the `s_{t-1} → s_t` transition and the emission at t; `forecast(X_hist, y_hist, x_next)` predicts one step ahead where `x_next` is the input observed at t+1.
- **Validation:** Docstring renders. No code change required.

#### 3.10 IOHMM external-features log-transform consistency
- **File:** `IOHMM/regimes/features.py:91-93`
- **Issue:** The `lag1_log_rv_d_{ticker}` column wraps in `np.log(... + eps)`; the sibling `lag1_log_rv_w_{ticker}` and `lag1_log_rv_m_{ticker}` columns do NOT. The names imply a log-transform; the values are raw rolling means. Dormant under `strictly_external_inputs=True` (the default), surfaced when external inputs are turned off.
- **Action:** Wrap both in `np.log(... + DEFAULT_EPS).shift(1)` to match the daily sibling.
- **Validation:** Bit-equal to `build_har_features`'s `log_rv_w_lag1` / `log_rv_m_lag1` columns on the same input.

---

### Phase 4 — Evaluation Framework Unification

#### 4.1 Pin a single dataset window
- **Files:** all entry points
- **Issue:** Date ranges differ across tracks.
- **Action:** Create `config/experiment_config.py` with constants `START_DATE = "2017-06-01"`, `END_DATE = "2025-01-31"`, `MIN_TRAIN = 504`, `REFIT_FREQ = 21`, `RV_HORIZON = 5`. Every track imports from here.
- **Validation:** Grep for hard-coded dates returns no matches outside this config file.

#### 4.2 Pin a single OOS protocol
- **Files:** all rolling-forecast functions
- **Issue:** Refit cadences vary from 1 day to never.
- **Action:** All tracks use `MIN_TRAIN=504`, `REFIT_FREQ=21`. Same warmup, same step, same horizon.
- **Validation:** `n_oos_obs` is identical across all tracks.

#### 4.3 Shared evaluator
- **File:** new module `utils/evaluation.py`
- **Issue:** Each track has its own loop logic and output schema.
- **Action:** Define `ExpandingWindowEvaluator(min_train, refit_freq, target_fn, predictor_factory)` that yields `(train_X, train_y, test_X, test_y, test_dates)` slices. Define a uniform output schema: `{date, y_true, y_pred, model_name}`. Every track wraps its model in `predictor_factory` and emits this schema.
- **Validation:** Five CSVs (one per track) with identical columns. A single `pd.concat` produces a clean long-format DataFrame.

#### 4.4 Cross-track results script
- **File:** new entry point `experiments/run_all.py`
- **Issue:** No single place to reproduce all results.
- **Action:** Invoke every track's evaluator, concatenate the long-format outputs, emit `results/unified_metrics.csv` (one row per model, columns from `compute_all_metrics`) and `results/dm_matrix.csv` (pairwise DM with HAC-adjusted p-values, with HAR-RV and persistence as primary baselines).
- **Validation:** Script runs end-to-end. Both CSVs reproducible bit-for-bit from a fixed seed.

#### 4.5 Per-regime decomposition
- **File:** `IOHMM/experiments/spy_vol_regime.py:32-50`
- **Issue:** Per-regime metrics already exist for IOHMM but are not produced for the other regime models.
- **Action:** Extend `per_regime_metrics` to accept `(y_true, y_hat, dom_state, K, model_name)` and run it for every regime model (HMM_weekly, HMM_daily, MS-AR, IOHMM). Emit `results/per_regime_metrics.csv`.
- **Validation:** CSV contains rows for all four regime models, with state ID, observation count, MSE, and QLIKE.

---

### Phase 5 — Refactoring and Reliability

#### 5.1 Add `requirements.txt`
- **File:** new file `requirements.txt`
- **Action:** Pin exact versions for `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `hmmlearn`, `arch`, `yfinance`, `matplotlib`, `pyarrow`. Lower bounds from README; upper bounds from current installed versions.
- **Validation:** `pip install -r requirements.txt` succeeds in a clean venv.

#### 5.2 Make project pip-installable
- **File:** new `pyproject.toml`
- **Action:** Declare the project as a package exporting `data_preprocessing`, `utils`, `IOHMM`, `HMM_weekly_vol`, `HMM_daily_return`, `Markov_Switching_AR`, `config`, `experiments`. Remove any `sys.path` manipulation.
- **Validation:** `pip install -e .` succeeds. Notebooks run from any working directory.

#### 5.3 Resolve duplicate adapter
- **Files:** `01_Data_Preprocessing_EDA/data_adapter.ipynb`, `Markov_Switching_AR/adapter.py`
- **Issue:** Three adapter import paths.
- **Action:** Notebook imports `YFinanceAdapter` from `data_preprocessing.data_adapter` instead of redefining it. Decide whether to keep `Markov_Switching_AR/adapter.py` re-export.
- **Validation:** Grep for `class YFinanceAdapter` returns exactly one match.

#### 5.3a Migrate `01_Data_Preprocessing_EDA/data_adapter.ipynb` to canonical adapter (deferred follow-up to §5.3)
- **File:** `01_Data_Preprocessing_EDA/data_adapter.ipynb`
- **Issue:** The notebook currently defines its own `YFinanceAdapter` (flat-column, CSV cache, no retries). Cells 3–6 depend on the notebook's flat `'SPY'` column convention to produce `spy_train_*.csv` / `spy_test_*.csv`. Substituting canonical (which returns a `(Price, Ticker)` MultiIndex) without adapting the consumer cells would silently change the on-disk CSV schema that downstream tracks read.
- **Action:** Edit cells 3–6 to extract `Close` from canonical's MultiIndex output while preserving the existing on-disk CSV schema. Then replace the inline `class YFinanceAdapter` definition (cell 1) with `from data_preprocessing.data_adapter import YFinanceAdapter`.
- **Validation:** Re-run the notebook and diff the resulting CSVs against the current ones — they must be byte-equivalent (or equivalent-after-known-schema-changes documented here). After this lands, the §5.3 grep validation (`class YFinanceAdapter` returns exactly one match) finally passes.
- **Note:** Naturally bundled with task 1.4 (rewriting the HMM weekly notebook against canonical) since the same migration pattern applies.

#### 5.4 Unit tests
- **File:** new `tests/` directory
- **Action:** Add pytest tests for: (a) `extract_adjusted_close` on canonical MultiIndex; (b) `build_log_rv_target` on synthetic constant-vol series; (c) `build_vol_iohmm_dataset` row counts and target alignment to t+1; (d) IOHMM forward filter equality with `_forward_backward` marginals on a 2-state synthetic; (e) `compute_all_metrics` on synthetic input with known answers; (f) `dm_stat` HAC equal-loss and unequal-loss tests; (g) MS-AR parameter extraction.
- **Validation:** `pytest tests/` passes locally.

#### 5.5 CI
- **File:** new `.github/workflows/ci.yml`
- **Action:** Run `pytest`, `ruff check`, and a notebook-execution smoke test on PRs against `main`. Cache `data/yfinance_cache` between runs.
- **Validation:** Green CI on a test PR.

#### 5.6 Centralize failure logging
- **Files:** `Markov_Switching_AR/model_utils.py`, `IOHMM/experiments/spy_vol_regime.py`
- **Action:** Replace `warnings.warn` in failure paths with structured logging via the standard `logging` module. Every failure record gets `(date, model, error_type, message)` appended to `failures.csv`.
- **Validation:** Each track output directory contains `failures.csv` (possibly empty). `n_oos_obs + n_failures == expected_test_days`.

#### 5.7 Update README
- **File:** `README.md`
- **Action:** Add a "Target Convention" subsection stating that all models predict `log(RV²_{t+1} + 1e-8)` with 5-day forward RV. Update each model section to reflect the new target. Add a "Cross-track comparability" subsection describing the unified evaluation harness. Remove or restructure any text claiming current cross-track results are valid.
- **Validation:** README renders correctly. No reference to the old per-track targets in the headline narrative.

---

## 4. Ordered Execution Steps

Apply in strict order. Do not start step N+1 until step N validates.

1. **5.1** Create `requirements.txt`.
2. **5.2** Add `pyproject.toml`. Make project pip-installable.
3. **5.3** Resolve duplicate adapter.
4. **1.1** Create `utils/targets.py` with `build_log_rv_target` and `build_log_rv_features`.
5. **1.2** Fix `rv_w` / `rv_m` in IOHMM features. Switch IOHMM target to `build_log_rv_target`.
6. **1.3** Fix GARCH target scale.
7. **1.4** Rewrite HMM_weekly_vol notebook against canonical adapter.
8. **1.5** Add failure tracking to MS-AR rolling forecast.
9. **1.6** Replace HMM weekly `backtest()` with `log RV²` forecaster.
10. **2.1** Implement HAC-adjusted DM with p-value.
11. **2.5** Implement `compute_all_metrics`.
12. **2.2** Implement held-out K-selection for HMM tracks.
13. **2.3** Constrain ARMA grid.
14. **2.4** Log ARMA fallback rate.
15. **3.5** Implement `PersistenceBaseline` and `HARRVBaseline` in `utils/baselines.py`.
16. **3.6** Add multistart EM to hmmlearn HMMs.
17. **3.7** Tighten state-label coherence in HMM_daily_return.
18. **3.8** Make MS-AR parameter extraction robust.
19. **3.9** Document IOHMM forecast convention.
20. **3.1** Retarget ARMA to `log RV²`.
21. **3.2** Retarget MS-AR to `log RV²`.
22. **3.3** Retarget HMM_weekly_vol to `log RV²`.
23. **3.4** Retarget HMM_daily_return to `log RV²`.
24. **4.1** Create `config/experiment_config.py` with shared constants.
25. **4.2** Apply shared OOS protocol across all tracks.
26. **4.3** Implement `ExpandingWindowEvaluator` in `utils/evaluation.py`.
27. **4.5** Per-regime metrics for all regime models.
28. **4.4** Implement `experiments/run_all.py`. Produce `unified_metrics.csv` and `dm_matrix.csv`.
29. **5.6** Centralize failure logging.
30. **5.4** Add unit tests in `tests/`.
31. **5.5** Add CI workflow.
32. **5.7** Update README.

---

## 5. Final Validation Checklist

### Required outputs

After execution, the repository must contain:

- `requirements.txt`
- `pyproject.toml`
- `config/experiment_config.py`
- `utils/targets.py`, `utils/baselines.py`, `utils/evaluation.py`
- `tests/` directory with passing pytest suite
- `.github/workflows/ci.yml`
- `experiments/run_all.py`
- `results/unified_metrics.csv` — one row per model
- `results/dm_matrix.csv` — pairwise DM with HAC-adjusted p-values
- `results/per_regime_metrics.csv` — per-regime decomposition for the four regime models
- Per-track long-format CSVs: `{date, y_true, y_pred, model_name}`
- Per-track `failures.csv`
- Updated `README.md` reflecting the unified target

### Required recomputed metrics

Every model in `unified_metrics.csv` reports:

- `mse`, `mae`, `rmse`
- `qlike`
- `r2_oos_vs_zero`
- `ic`
- `dm_stat_vs_har`, `dm_pvalue_vs_har`
- `dm_stat_vs_persistence`, `dm_pvalue_vs_persistence`

### Confirmation criteria

**Consistent targets**

- All models predict `y_t = log(RV²_{t+1} + 1e-8)` from `utils.targets.build_log_rv_target` with `horizon=5`.
- Cross-track table has one prediction column per model, all in the same numeric range.
- Spot-check on any single OOS date: predictions from all models lie within ±2 of `y_true`.
- No file outside `utils/targets.py` defines its own RV proxy.

**No data leakage**

- `build_log_rv_target` uses only `r_{t+1}, ..., r_{t+horizon}` and is indexed at `t`. Predictions at time `t` use only data up to and including `t`.
- All HAR-style features are computed from rolling means of squared past returns and shifted by 1.
- IOHMM `forecast()` uses the forward filter only (verified by code inspection).
- IOHMM standardization is recomputed per refit on training data only.
- Regression test: deliberately remove a `.shift(1)` somewhere; metrics should improve dramatically; revert.

**Valid comparisons**

- All tracks evaluated on the same canonical date range after warmup.
- All tracks use `MIN_TRAIN=504`, `REFIT_FREQ=21`, `RV_HORIZON=5` from the shared config.
- All tracks emit identical CSV schema; concatenation yields a clean long-format DataFrame.
- All metrics computed by `compute_all_metrics`.
- DM tests use HAC-adjusted variance and report p-values.
- `unified_metrics.csv` and `dm_matrix.csv` are reproducible bit-for-bit from a fixed seed across two clean checkouts.

**Reproducibility**

- `pip install -r requirements.txt && pip install -e .` succeeds in a clean venv.
- `python experiments/run_all.py` produces identical CSVs across two clean runs with the same seed.
- `pytest tests/` passes.
- CI is green.