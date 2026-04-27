"""Shared canonical-data loaders for the cross-track comparison.

Single source of ``rv_gk`` (and the matching HMM observation matrix) for the
four-track evaluation (HMM, IOHMM, MS-AR, HAR) so that ``y_true_rv_gk`` is
bit-equal across all per-task CSV outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_preprocessing.data_adapter import YFinanceAdapter
from HMM.features import GKVolFeatures


def get_canonical_rv_gk(
    ticker: str = "SPY",
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    cache_dir: str = "experiments/cache",
) -> pd.Series:
    """Return canonical rv_gk Series for the cross-track comparison.

    Loads from cache if present; otherwise fetches OHLC via
    YFinanceAdapter, computes Garman-Klass annualized variance,
    caches to parquet, and returns.
    """
    cache_path = Path(cache_dir) / f"rv_gk_{ticker}_{start}_{end}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        rv = df["rv_gk"].astype(np.float64)
        rv.index = pd.to_datetime(rv.index)
        rv.name = "rv_gk"
        return rv

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw = YFinanceAdapter().get_data(
        tickers=[ticker],
        start_date=start,
        end_date=end,
        force_refresh=False,
    )
    rv = GKVolFeatures.compute_gk(raw, ticker).astype(np.float64)
    rv.name = "rv_gk"
    rv.to_frame().to_parquet(cache_path)
    return rv


def get_canonical_obs(
    ticker: str = "SPY",
    start: str = "2019-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Return GKVolFeatures().fit_transform(rv_gk) for HMM observations."""
    rv = get_canonical_rv_gk(ticker=ticker, start=start, end=end)
    return GKVolFeatures().fit_transform(rv)


def _forward_5_mean(rv: pd.Series) -> pd.Series:
    """Forward 5-day rolling mean: y[t] = mean(rv[t+1..t+5]).

    Implementation: rv.rolling(5).mean().shift(-5). The unshifted rolling
    mean at index i is mean(rv[i-4..i]); shifting by -5 moves the value at
    index i+5 to index i, giving y[i] = mean(rv[i+1..i+5]). Last 5 rows
    are NaN by construction. Assumes rv has no NaN; otherwise NaN
    propagates from the rolling step.
    """
    return rv.rolling(window=5, min_periods=5).mean().shift(-5)


def get_canonical_rv_gk_h5(
    ticker: str = "SPY",
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    cache_dir: str = "experiments/cache",
) -> pd.Series:
    """Return canonical 5-day forward-average rv_gk target.

    y_t = (1/5) * sum(rv_gk[t+1..t+5])

    Built from the same cached ``get_canonical_rv_gk`` series so the
    underlying daily rv_gk values are bit-equal to the h=1 track. Last 5
    rows are NaN by construction.
    """
    cache_path = Path(cache_dir) / f"rv_gk_h5_{ticker}_{start}_{end}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        y = df["rv_gk_h5"].astype(np.float64)
        y.index = pd.to_datetime(y.index)
        y.name = "rv_gk_h5"
        return y

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rv = get_canonical_rv_gk(ticker=ticker, start=start, end=end)
    y = _forward_5_mean(rv).astype(np.float64)
    y.name = "rv_gk_h5"
    y.to_frame().to_parquet(cache_path)
    return y
