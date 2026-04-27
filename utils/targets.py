"""Canonical target builder for the SPY volatility forecasting project.

The whole project unifies on a single forecasting target:

    y_t = log( (1/H) * sum_{i=t+1}^{t+H} r_i^2  +  eps )

where:
  - r_t = log(p_t / p_{t-1}) is the natural-log return, observed at t.
  - H is the forward-window length (default 5 trading days).
  - eps = DEFAULT_EPS = 1e-8 is the project-wide numerical regularizer
    inside the log, preventing log(0) when realized variance degenerates.

Two helpers are exposed:

  build_log_rv_target(close, horizon, eps) -> pd.Series
      The canonical target series.

  build_log_rv_features(close, lags, eps) -> pd.DataFrame
      HAR-style lagged log-RV regressors, .shift(1)-aligned so that the
      feature row at time t uses past data only (r^2_{t-L}, ..., r^2_{t-1}).

Conventions
-----------
- Target uses FUTURE returns only: y_t depends on r_{t+1}, ..., r_{t+H}
  and never on r_t or any prior return.
- HAR features use PAST returns only: feature_t for window L depends on
  r^2_{t-L}, ..., r^2_{t-1}. The .shift(1) is what enforces this; without
  it, the feature would peek at r^2_t (which is observed at t, on the
  same calendar bar as the prediction is made).
- y.iloc[0] is masked to NaN even though r_1, ..., r_H are observable.
  Reason: at t=0 the model has no past return history, so no
  past-data-only forecast is meaningful at that index. This convention
  aligns with how downstream evaluators index the test set.
- Float64 throughout. Inputs are cast to float64 at function entry.
- Input NaNs are NOT handled, NOT raised, NOT warned. The caller is
  responsible for cleaning. Input NaNs propagate through the compute
  chain to output NaNs.
- DEFAULT_EPS = 1e-8 is the project-wide regularizer. Do not change
  without auditing every consumer (every model, every metric).
"""

from typing import Tuple

import numpy as np
import pandas as pd

DEFAULT_EPS: float = 1e-8


def build_log_rv_target(
    close: pd.Series,
    horizon: int = 5,
    eps: float = DEFAULT_EPS,
) -> pd.Series:
    """Build the canonical log-realized-variance target.

    For each index t the function computes

        y_t = log( (1/H) * sum_{i=t+1}^{t+H} r_i^2  +  eps )

    where r_i = log(close_i / close_{i-1}).

    Parameters
    ----------
    close : pd.Series
        1-D series of close prices indexed ascending by date. The series
        is cast to float64 internally; the original dtype is not
        preserved on the output. Input NaNs are not handled — they
        propagate to output NaNs. The caller is responsible for cleaning.
    horizon : int, default 5
        Forward-window length H. Must be a positive integer.
    eps : float, default DEFAULT_EPS (1e-8)
        Numerical regularizer added inside the log.

    Returns
    -------
    pd.Series
        Same index as `close`, length len(close), dtype float64.
        - y.iloc[0] is NaN by convention (see module docstring).
        - y.iloc[N-H], ..., y.iloc[N-1] are NaN (no future window).
        - Otherwise finite if `close` is clean over the window.

    Notes
    -----
    Step-by-step:
        r       = log(close / close.shift(1))
        r2      = r ** 2
        fwd_avg = mean of r2.iloc[t+1 : t+1+H] for each t
        y       = log(fwd_avg + eps)
        y.iloc[0] = NaN   # convention; see module docstring
    """
    if not isinstance(close, pd.Series):
        raise TypeError(f"close must be pd.Series, got {type(close).__name__}")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError(f"horizon must be a positive int, got {horizon!r}")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
        raise ValueError(f"eps must be a positive float, got {eps!r}")

    close_f64 = close.astype(np.float64, copy=False)

    r = np.log(close_f64 / close_f64.shift(1))
    r2 = r ** 2

    # Forward-window mean: at index t, mean of r2.iloc[t+1 : t+1+H].
    # rolling(H).mean() at index k gives mean(r2.iloc[k-H+1 : k+1]).
    # Setting k = t+H gives mean(r2.iloc[t+1 : t+H+1]) = the desired window.
    # Then shift(-H) places that value at index t.
    fwd_avg = (
        r2.rolling(window=horizon, min_periods=horizon)
        .mean()
        .shift(-horizon)
    )

    y = np.log(fwd_avg + eps).astype(np.float64, copy=False)

    # Convention: at t=0 the model has no past history, so no forecast
    # is meaningful. Mask y.iloc[0] even though the future window is
    # technically computable. See module docstring "Conventions".
    if len(y) > 0:
        y.iloc[0] = np.nan

    return y


def build_log_rv_features(
    close: pd.Series,
    lags: Tuple[int, ...] = (1, 5, 22),
    eps: float = DEFAULT_EPS,
) -> pd.DataFrame:
    """Build HAR-style lagged log-realized-variance features.

    For each lag L in `lags` the function computes

        rv_L_t      = mean(r^2_{t-L+1}, ..., r^2_t)
        log_rv_L_t  = log(rv_L_t + eps)
        feature_t   = log_rv_L_{t-1}        # via .shift(1)

    The .shift(1) ensures the feature row at time t uses only past data
    (r^2_{t-L}, ..., r^2_{t-1}); it never peeks at r^2_t.

    Parameters
    ----------
    close : pd.Series
        1-D series of close prices. Cast to float64 internally. Input
        NaNs are not handled.
    lags : tuple of int, default (1, 5, 22)
        Window lengths. Must be a non-empty tuple of strictly ascending
        positive integers. (1, 5, 22) corresponds to the canonical Corsi
        HAR-RV daily / weekly / monthly horizons.
    eps : float, default DEFAULT_EPS (1e-8)
        Numerical regularizer.

    Returns
    -------
    pd.DataFrame
        Columns "log_rv_lag_{L}" for each L in `lags`, in the order
        given. Same index and length as `close`. dtype float64 per
        column. Leading NaN per column equals L (rolling needs L
        observations and .shift(1) adds one).
    """
    if not isinstance(close, pd.Series):
        raise TypeError(f"close must be pd.Series, got {type(close).__name__}")
    if not isinstance(lags, tuple):
        raise TypeError(f"lags must be a tuple, got {type(lags).__name__}")
    if len(lags) == 0:
        raise ValueError("lags must be non-empty")
    for L in lags:
        if not isinstance(L, int) or isinstance(L, bool) or L < 1:
            raise ValueError(f"all lags must be positive ints, got {lags!r}")
    if any(lags[i] >= lags[i + 1] for i in range(len(lags) - 1)):
        raise ValueError(f"lags must be strictly ascending, got {lags!r}")
    if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps <= 0:
        raise ValueError(f"eps must be a positive float, got {eps!r}")

    close_f64 = close.astype(np.float64, copy=False)

    r = np.log(close_f64 / close_f64.shift(1))
    r2 = r ** 2

    cols = {}
    for L in lags:
        rv_L = r2.rolling(window=L, min_periods=L).mean()
        log_rv_L = np.log(rv_L + eps)
        feature = log_rv_L.shift(1).astype(np.float64, copy=False)
        cols[f"log_rv_lag_{L}"] = feature

    return pd.DataFrame(cols, index=close.index)
