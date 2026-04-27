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
