from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass
class IOHMMPreparedData:
    X: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    feature_names: List[str]
    frame: pd.DataFrame


def _as_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _check_adapter_df(df: pd.DataFrame) -> None:
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Expected adapter data with MultiIndex columns: (Field, Ticker).")
    if df.columns.nlevels != 2:
        raise ValueError("Expected 2-level MultiIndex columns: (Field, Ticker).")


def _col(df: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    key = (field, ticker)
    if key not in df.columns:
        raise KeyError(f"Missing column {key} in adapter dataframe.")
    return _as_float_series(df[key])


def build_vol_iohmm_dataset(
    df: pd.DataFrame,
    target_ticker: str = "SPY",
    external_tickers: Sequence[str] = ("TLT", "HYG", "UUP", "GLD"),
    rv_window_target: int = 10,
    rv_window_external: int = 5,
    strictly_external_inputs: bool = True,
) -> IOHMMPreparedData:
    """
    Build X/y for SPY volatility-regime IOHMM from DataAdapter-formatted data.

    Target:
        y_t = log(annualized rolling realized vol of SPY)

    External features:
        lagged daily return
        lagged log realized vol
        lagged 5-day cumulative return

    Optional internal features:
        lagged target realized vol
        lagged abs return
        lagged volume z-score
    """
    _check_adapter_df(df)

    close_target = _col(df, "Close", target_ticker)
    vol_target = _col(df, "Volume", target_ticker) if ("Volume", target_ticker) in df.columns else None

    ret_target = np.log(close_target / close_target.shift(1))
    rv_target = ret_target.rolling(rv_window_target).std() * np.sqrt(252.0)
    y = np.log(rv_target + 1e-8)

    out = pd.DataFrame(index=df.index)
    out["y_log_rv"] = y

    for ticker in external_tickers:
        close = _col(df, "Close", ticker)
        ret = np.log(close / close.shift(1))
        rv = ret.rolling(rv_window_external).std() * np.sqrt(252.0)

        out[f"lag1_ret_{ticker}"] = ret.shift(1)
        out[f"lag1_log_rv_{ticker}"] = np.log(rv + 1e-8).shift(1)
        out[f"lag5_cumret_{ticker}"] = ret.rolling(5).sum().shift(1)

    if not strictly_external_inputs:
        out[f"lag1_y_log_rv_{target_ticker}"] = y.shift(1)
        out[f"lag1_abs_ret_{target_ticker}"] = ret_target.abs().shift(1)

        if vol_target is not None:
            vol_z = (vol_target - vol_target.rolling(20).mean()) / vol_target.rolling(20).std()
            out[f"lag1_vol_z_{target_ticker}"] = vol_z.shift(1)

    out = out.dropna().copy()

    feature_names = [c for c in out.columns if c != "y_log_rv"]
    X = out[feature_names].to_numpy(dtype=float)
    y_arr = out["y_log_rv"].to_numpy(dtype=float)

    return IOHMMPreparedData(
        X=X,
        y=y_arr,
        dates=pd.DatetimeIndex(out.index),
        feature_names=feature_names,
        frame=out,
    )