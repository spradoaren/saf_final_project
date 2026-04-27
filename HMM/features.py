"""
HMM feature builders and walk-forward framework.

Public API
----------
Feature builders
    ReturnFeatures        – triple-variate return observations (Zhang et al. 2019)
    GKVolFeatures         – HAR-style log-RV features from Garman-Klass variance

HMM utilities
    build_second_order_tensor   – estimate A2[i,j,k] from a Viterbi sequence
    SecondOrderHMM              – GaussianHMM with integrated scaler + 2nd-order prediction
    WalkForwardHMM              – rolling-window refit/predict loop (base class)
    ReturnSignalWF              – walk-forward → long/flat signal  (return notebook)
    RVForecastWF                – walk-forward → RV point forecast (vol notebook)

Backward compatibility
    build_features        – module-level alias for ReturnFeatures().fit_transform(df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from hmmlearn.hmm import GaussianHMM


# ══════════════════════════════════════════════════════════════════════════════
# Feature builders
# ══════════════════════════════════════════════════════════════════════════════

class ReturnFeatures:
    """
    Triple-variate return observations from Zhang et al. (2019), Physica A 517.

    Columns produced
    ----------------
    simple_return   (P_t - P_{t-1}) / P_{t-1}
    log_return_1d   log(P_t / P_{t-1})
    log_return_5d   log(P_t / P_{t-long_window})
    """

    def __init__(self, long_window: int = 5):
        self.long_window = long_window

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : DataFrame with columns 'price', 'simple_return', 'log_return'.

        Returns
        -------
        obs : DataFrame with three columns, NaN rows dropped.
        """
        obs = pd.DataFrame(index=df.index)
        obs["simple_return"] = df["simple_return"]
        obs["log_return_1d"] = df["log_return"]
        obs["log_return_5d"] = np.log(df["price"] / df["price"].shift(self.long_window))
        return obs.dropna()


def make_return_dataframe(price: pd.Series) -> pd.DataFrame:
    """
    Build the canonical return-input DataFrame from a price series.

    Parameters
    ----------
    price : Series of prices.

    Returns
    -------
    DataFrame with columns: price, log_return, simple_return.
    """
    df = price.rename("price").to_frame()
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["simple_return"] = df["price"].pct_change()
    return df


def build_return_observations(price: pd.Series, long_window: int = 5) -> pd.DataFrame:
    """
    Convenience wrapper: price series -> return HMM observation triple.
    """
    return ReturnFeatures(long_window=long_window).fit_transform(make_return_dataframe(price))


class GKVolFeatures:
    """
    HAR-style log-realized-variance features based on the Garman-Klass (1980) estimator.

    Columns produced
    ----------------
    log_rv_d   log(GK_t * 252)                         — daily
    log_rv_w   log(rolling(weekly_window).mean * 252)  — weekly
    log_rv_m   log(rolling(monthly_window).mean * 252) — monthly
    """

    def __init__(self, weekly_window: int = 5, monthly_window: int = 21):
        self.weekly_window = weekly_window
        self.monthly_window = monthly_window

    @staticmethod
    def compute_gk(raw_df: pd.DataFrame, ticker: str) -> pd.Series:
        """
        Garman-Klass (1980) annualised variance from YFinanceAdapter OHLC output.

        GK = 0.5*(ln H/L)^2 - (2*ln2 - 1)*(ln C/O)^2,  then *252 to annualise.

        Parameters
        ----------
        raw_df : MultiIndex DataFrame from YFinanceAdapter.get_data()
        ticker : e.g. 'SPY'

        Returns
        -------
        rv_gk : Series of daily annualised variance, clipped to [1e-10, inf).
        """
        O = raw_df["Open"][ticker]
        H = raw_df["High"][ticker]
        L = raw_df["Low"][ticker]
        C = raw_df["Close"][ticker]
        rv = (
            0.5 * np.log(H / L) ** 2
            - (2 * np.log(2) - 1) * np.log(C / O) ** 2
        ).clip(lower=1e-10) * 252
        rv.name = "rv_gk"
        return rv

    def fit_transform(self, rv_gk: pd.Series) -> pd.DataFrame:
        """
        Parameters
        ----------
        rv_gk : daily annualised GK variance series (from compute_gk).

        Returns
        -------
        obs : DataFrame with three log-scale columns, NaN rows dropped.
        """
        obs = pd.DataFrame(index=rv_gk.index)
        obs["log_rv_d"] = np.log(rv_gk)
        obs["log_rv_w"] = np.log(rv_gk.rolling(self.weekly_window).mean())
        obs["log_rv_m"] = np.log(rv_gk.rolling(self.monthly_window).mean())
        return obs.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# HMM utilities
# ══════════════════════════════════════════════════════════════════════════════

def build_second_order_tensor(
    states: np.ndarray,
    n_states: int,
    smoothing: float = 1e-6,
) -> np.ndarray:
    """
    Estimate A2[i, j, k] = P(q_t = k | q_{t-2} = i, q_{t-1} = j) by counting
    3-grams in a Viterbi-decoded state sequence (Laplace-smoothed).

    Parameters
    ----------
    states    : 1-D integer array of decoded hidden states.
    n_states  : number of distinct states.
    smoothing : Laplace pseudo-count added to every cell before normalisation.

    Returns
    -------
    A2 : ndarray of shape (n_states, n_states, n_states), rows sum to 1.
    """
    A2 = np.ones((n_states, n_states, n_states)) * smoothing
    for t in range(2, len(states)):
        A2[states[t - 2], states[t - 1], states[t]] += 1
    A2 /= A2.sum(axis=2, keepdims=True)
    return A2


class SecondOrderHMM:
    """
    Gaussian HMM with integrated StandardScaler and optional 2nd-order prediction.

    Differences from raw hmmlearn.GaussianHMM
    ------------------------------------------
    - Scaler is fitted and stored alongside the HMM (single training call).
    - 2nd-order transition tensor A2 is estimated from the Viterbi sequence when
      order=2, enabling richer next-state prediction.
    - Helpers for emission-mean inspection and state ranking by any feature column.

    Parameters
    ----------
    n_states     : number of hidden states.
    order        : Markov order for next-state prediction (1 or 2).
    n_iter       : Baum-Welch EM iterations.
    random_state : reproducibility seed.
    """

    def __init__(
        self,
        n_states: int = 3,
        order: int = 1,
        n_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.order = order
        self.n_iter = n_iter
        self.random_state = random_state

        self.hmm_: GaussianHMM | None = None
        self.scaler_: StandardScaler | None = None
        self.A2_: np.ndarray | None = None

    def fit(self, X_raw: np.ndarray) -> SecondOrderHMM:
        """
        Fit StandardScaler then GaussianHMM on raw (unscaled) observations.
        When order=2, also estimate the 2nd-order transition tensor A2.
        """
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_raw)

        hmm = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        hmm.fit(X_sc)

        self.hmm_ = hmm
        self.scaler_ = sc

        if self.order == 2:
            states = hmm.predict(X_sc)
            self.A2_ = build_second_order_tensor(states, self.n_states)

        return self

    def predict_states(self, X_raw: np.ndarray) -> np.ndarray:
        """Viterbi-decode raw observations using the fitted scaler."""
        return self.hmm_.predict(self.scaler_.transform(X_raw))

    def predict_next_state(self, X_raw: np.ndarray) -> int:
        """
        Predict the next hidden state from context observations X_raw.

        1st-order : argmax over the transition row of the last decoded state.
        2nd-order : argmax over A2[s_{t-2}, s_{t-1}, :].
        Falls back to 1st-order when the context contains fewer than 2 steps.
        """
        states = self.predict_states(X_raw)
        if self.order == 2 and self.A2_ is not None and len(states) >= 2:
            return int(np.argmax(self.A2_[states[-2], states[-1]]))
        return int(np.argmax(self.hmm_.transmat_[states[-1]]))

    def emission_means_original(self) -> np.ndarray:
        """
        Return emission means in the original (pre-scaling) feature space.
        Shape: (n_states, n_features).
        """
        return self.scaler_.inverse_transform(self.hmm_.means_)

    def state_order(self, col: int = 0) -> list[int]:
        """
        State indices sorted by emission mean of feature column ``col``, ascending.

        Examples
        --------
        Return notebook (col=1 → log_return_1d):
            order = model.state_order(col=1)
            bull_state   = order[-1]   # highest expected return
            bear_state   = order[0]    # lowest expected return

        Volatility notebook (col=0 → log_rv_d):
            order = model.state_order(col=0)
            high_vol_state = order[-1]
            low_vol_state  = order[0]
        """
        means = self.emission_means_original()[:, col]
        return list(np.argsort(means))


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward framework
# ══════════════════════════════════════════════════════════════════════════════

class WalkForwardHMM:
    """
    Rolling-window walk-forward driver for HMM-based forecasting.

    Shared mechanics
    ----------------
    - Refit ``SecondOrderHMM`` every ``refit_every`` steps on a fixed
      ``train_window``-day look-back window.
    - Decode the current context and predict the next hidden state.
    - Delegate the task-specific output to ``_output_value`` (override in subclasses).

    Parameters
    ----------
    n_states     : number of hidden states.
    order        : 1 or 2 (Markov order for next-state prediction).
    train_window : look-back window in trading days (default 252).
    refit_every  : refit the HMM every N steps (default 21 ≈ monthly).
    n_iter       : Baum-Welch EM iterations (default 100).
    random_state : reproducibility seed.
    """

    def __init__(
        self,
        n_states: int = 3,
        order: int = 1,
        train_window: int = 252,
        refit_every: int = 21,
        n_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_states = n_states
        self.order = order
        self.train_window = train_window
        self.refit_every = refit_every
        self.n_iter = n_iter
        self.random_state = random_state

    def _output_value(
        self,
        model: SecondOrderHMM,
        predicted_state: int,
        obs_df: pd.DataFrame,
        t: int,
    ) -> float:
        """
        Produce the scalar output for time step t given the fitted model and
        predicted next state.  Override in subclasses.
        Default: return the raw state index (useful for debugging).
        """
        return float(predicted_state)

    def run(self, obs_df: pd.DataFrame) -> pd.Series:
        """
        Execute the walk-forward loop over all rows of obs_df.

        Returns
        -------
        pd.Series of outputs aligned with obs_df.index.
        The first ``train_window`` entries are NaN (warm-up period).
        """
        n     = len(obs_df)
        X_all = obs_df.values
        out   = pd.Series(np.nan, index=obs_df.index)
        model: SecondOrderHMM | None = None

        for t in range(self.train_window, n):
            # ── refit ──────────────────────────────────────────────────────
            if model is None or (t - self.train_window) % self.refit_every == 0:
                X_raw = X_all[t - self.train_window : t]
                model = SecondOrderHMM(
                    n_states=self.n_states,
                    order=self.order,
                    n_iter=self.n_iter,
                    random_state=self.random_state,
                )
                try:
                    model.fit(X_raw)
                except Exception:
                    model = None
                    continue

            # ── predict ────────────────────────────────────────────────────
            ctx = X_all[max(0, t - self.train_window) : t]
            nxt = model.predict_next_state(ctx)
            out.iloc[t] = self._output_value(model, nxt, obs_df, t)

        return out


class ReturnSignalWF(WalkForwardHMM):
    """
    Walk-forward → long/flat trading signal for return-prediction notebooks.

    Signal = 1.0 (long) when the predicted next state is the bull state
    (highest emission mean for ``signal_col``), else 0.0 (flat).

    Parameters
    ----------
    signal_col : feature column index used to rank states.
                 Default 1 = log_return_1d in the ReturnFeatures triple.
    **kwargs   : forwarded to WalkForwardHMM.
    """

    def __init__(self, signal_col: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.signal_col = signal_col

    def _output_value(self, model, predicted_state, obs_df, t):
        bull_state = model.state_order(col=self.signal_col)[-1]
        return 1.0 if predicted_state == bull_state else 0.0


class ReturnForecastWF(WalkForwardHMM):
    """
    Walk-forward -> continuous next-day return forecast for return notebooks.

    Forecast = emission mean of ``forecast_col`` for the predicted next state.
    For the default ``forecast_col=1``, this returns predicted ``log_return_1d``.

    Parameters
    ----------
    forecast_col : feature column index whose emission mean becomes the forecast.
                   Default 1 = log_return_1d in the ReturnFeatures triple.
    **kwargs     : forwarded to WalkForwardHMM.
    """

    def __init__(self, forecast_col: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.forecast_col = forecast_col

    def _output_value(self, model, predicted_state, obs_df, t):
        means = model.emission_means_original()[:, self.forecast_col]
        return float(means[predicted_state])


class RVForecastWF(WalkForwardHMM):
    """
    Walk-forward → continuous RV point forecast for volatility notebooks.

    Forecast = exp(emission mean of ``forecast_col`` for the predicted state),
    converting from log-variance back to the original variance scale.

    Parameters
    ----------
    forecast_col : feature column index whose emission mean becomes the forecast.
                   Default 0 = log_rv_d in the GKVolFeatures triple.
    **kwargs     : forwarded to WalkForwardHMM.
    """

    def __init__(self, forecast_col: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.forecast_col = forecast_col

    def _output_value(self, model, predicted_state, obs_df, t):
        log_means = model.emission_means_original()[:, self.forecast_col]
        return float(np.exp(log_means[predicted_state]))


def walk_forward_har_rv(
    obs_df: pd.DataFrame,
    train_window: int = 252,
    refit_every: int = 21,
) -> pd.Series:
    """
    Walk-forward HAR-RV forecast (Corsi 2009, log version).

    OLS: log_rv_d(t) ~ log_rv_d(t-1) + log_rv_w(t-1) + log_rv_m(t-1)
    """
    n = len(obs_df)
    forecasts = pd.Series(np.nan, index=obs_df.index)
    lr: LinearRegression | None = None
    y_all = obs_df["log_rv_d"].values
    X_all = obs_df.values

    for t in range(train_window, n):
        if lr is None or (t - train_window) % refit_every == 0:
            X_tr = X_all[t - train_window : t - 1]
            y_tr = y_all[t - train_window + 1 : t]
            lr = LinearRegression().fit(X_tr, y_tr)
        forecasts.iloc[t] = np.exp(lr.predict(X_all[t - 1 : t])[0])

    return forecasts


# ══════════════════════════════════════════════════════════════════════════════
# Backward-compatible module-level alias
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, long_window: int = 5) -> pd.DataFrame:
    """Alias for ReturnFeatures(long_window).fit_transform(df). Kept for compatibility."""
    return ReturnFeatures(long_window=long_window).fit_transform(df)