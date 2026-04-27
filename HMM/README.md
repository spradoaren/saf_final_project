# High-Order HMM on SPY: Return Regimes and Realized Volatility

This directory contains two research-style notebook experiments:

- `return_prediction.ipynb`: hidden-state discovery on SPY returns and a regime-conditioned trading backtest.
- `rv_prediction.ipynb`: hidden-state modeling of realized volatility and one-step-ahead volatility forecasting.

Together, these notebooks test a common thesis: financial dynamics are not homogeneous through time, and a latent-state process can capture persistent market regimes (trend vs stress, low-vol vs high-vol) better than a single linear model.

## Research Motivation

Classical time-series models assume stationary parameters over long periods. In practice, equity markets switch between qualitatively different environments: momentum-driven expansion, panic drawdown, and transitional consolidation. Hidden Markov models (HMMs) are a natural way to represent this behavior, because they combine:

1. **A latent Markov chain** for regime transitions.
2. **State-conditional emission distributions** for observed features.
3. **Probabilistic filtering** for online state inference and forecasting.

This project extends the baseline first-order HMM to a second-order formulation, allowing transition dynamics to depend on two previous states instead of one. That design specifically targets persistence and path-dependence effects often observed in volatility clustering and trend continuation.

## Mathematical Setup

### 1) First-Order Gaussian HMM

Let \(q_t \in \{1,\dots,K\}\) denote hidden state and \(x_t \in \mathbb{R}^d\) the observed feature vector.

\[
P(q_t \mid q_{1:t-1}) = P(q_t \mid q_{t-1}) = A_{q_{t-1},q_t}
\]

\[
x_t \mid (q_t=k) \sim \mathcal{N}(\mu_k,\Sigma_k)
\]

Joint likelihood:

\[
P(x_{1:T}, q_{1:T})
= \pi_{q_1}\, \mathcal{N}(x_1;\mu_{q_1},\Sigma_{q_1})
\prod_{t=2}^{T} A_{q_{t-1},q_t}\,\mathcal{N}(x_t;\mu_{q_t},\Sigma_{q_t})
\]

### 2) Second-Order HMM

Second-order dynamics are:

\[
P(q_t \mid q_{1:t-1}) = P(q_t \mid q_{t-1}, q_{t-2})
= A^{(2)}_{q_{t-2},q_{t-1},q_t}
\]

In implementation, this is handled with state augmentation (compatible with Gaussian-HMM training APIs), then mapped back to original regimes for interpretation.

### 3) Realized Volatility Proxy (Garman-Klass)

From OHLC prices:

\[
\widehat{\sigma}^2_{\text{GK},t}
= \frac{1}{2}\left(\ln\frac{H_t}{L_t}\right)^2
- (2\ln 2 - 1)\left(\ln\frac{C_t}{O_t}\right)^2
\]

Annualized RV is then transformed to log scale for modeling stability.

### 4) HAR-Style RV Features

For realized variance \(RV_t\):

\[
\log RV^{(d)}_t = \log(RV_t),\quad
\log RV^{(w)}_t = \log\left(\frac{1}{5}\sum_{i=0}^{4}RV_{t-i}\right),\quad
\log RV^{(m)}_t = \log\left(\frac{1}{22}\sum_{i=0}^{21}RV_{t-i}\right)
\]

These form the observation vector used by volatility-regime HMMs.

### 5) Forecast and Evaluation

Walk-forward evaluation uses rolling refits and one-step-ahead prediction. For volatility forecasts \(\hat v_t\) vs actual \(v_t\), QLIKE is:

\[
\text{QLIKE}_t = \frac{\hat v_t}{v_t} - \log\!\left(\frac{\hat v_t}{v_t}\right) - 1
\]

This loss is robust and standard for variance forecast comparison.

## Notebook A: Return Regime Prediction (`return_prediction.ipynb`)

### Data and Features

- SPY daily prices transformed into return-based features.
- Regime labels are interpreted post-fit (Bull/Bear/Neutral) from empirical state statistics.

### Models

- First-order Gaussian HMM (`order=1`).
- Second-order HMM (`order=2`) using augmented-state representation.

### Backtest Design

- Walk-forward protocol with fixed train window (`TRAIN_WINDOW = 252`).
- Periodic refit (`refit_every = 21`) to mimic monthly model updates.
- Strategy converts predicted state into position and compares against buy-and-hold.

### Result Narrative

The notebook typically shows three consistent patterns:

1. **State separation in return distribution**: one regime has clearly lower/negative mean return and higher downside concentration.
2. **Transition persistence**: second-order transition heatmaps reveal memory effects not visible in first-order transitions.
3. **Strategy-level impact**: regime-filtered exposure tends to improve drawdown control; Sharpe and annualized return comparisons are reported in the performance table.

Visual evidence comes from:

- hidden-state overlays on price/return,
- per-state return histograms,
- transition heatmaps,
- cumulative return and drawdown panels.

## Notebook B: RV Prediction (`rv_prediction.ipynb`)

### Data and Features

- Garman-Klass realized variance from SPY OHLC.
- Log-RV daily/weekly/monthly components (`log_rv_d`, `log_rv_w`, `log_rv_m`).

### Models and Baselines

- First-order and second-order Gaussian HMM regime forecasts.
- Baseline comparators include naive/random-walk style and HAR-based walk-forward forecast.

### Evaluation Design

- One-step-ahead walk-forward forecasting.
- Metric family includes trajectory plots and cumulative QLIKE competition.
- Additional diagnostics: predicted-vs-actual scatter in log scale.

### Result Narrative

The RV notebook usually supports the following conclusions:

1. **Volatility regimes are persistent** (high-vol clusters and low-vol clusters are visibly segmented).
2. **Second-order dynamics better capture clustering** when two consecutive high-vol states increase probability of continued stress.
3. **Forecast ranking is metric-dependent**: HMM variants may win during stress transitions while HAR can remain competitive in calm periods.

The cumulative QLIKE plot is the most direct "horse-race" summary over the out-of-sample path.

## Key Code Components (`features.py`)

- `ReturnFeatures`, `GKVolFeatures`: feature engineering for return and RV pipelines.
- `SecondOrderHMM`: first/second-order unified model wrapper.
- `WalkForwardHMM`: rolling training and one-step-ahead state prediction engine.
- `ReturnSignalWF`, `ReturnForecastWF`, `RVForecastWF`: task-level walk-forward wrappers.
- `walk_forward_har_rv`: HAR baseline implementation for RV forecast benchmarking.

## Reproducibility

### Environment

- Python 3.10+
- Install dependencies from repo root:

```bash
pip install -r requirements.txt
```

### Run Order

From `cu_saf` root:

1. Run `HMM/return_prediction.ipynb` from top to bottom.
2. Run `HMM/rv_prediction.ipynb` from top to bottom.

Use a clean kernel for each notebook to avoid leakage from previous variables.

## Results and Artifacts

- Figures are saved to `HMM/pic/` (state overlays, distributions, transition heatmaps, forecast/backtest curves).
- Quantitative tables are generated in notebook outputs (performance metrics and forecast comparisons).
- If you modify sample windows (for example restricting walk-forward to 2023-2025), rerun all downstream cells to keep metrics and figures aligned.

## Interpretation Notes

- HMM regime labels are semantic and assigned after fitting based on observed state behavior; labels are not supervised targets.
- Better state separation does not always imply superior trading PnL; transaction costs and turnover sensitivity matter.
- For volatility, model superiority can change by market phase, so full-path cumulative loss plots are more informative than a single aggregate point metric.
