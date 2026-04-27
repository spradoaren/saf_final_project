from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from data_preprocessing.price_utils import extract_adjusted_close
from utils.targets import build_log_rv_target


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


def _close(df: pd.DataFrame, ticker: str) -> pd.Series:
    return _as_float_series(extract_adjusted_close(df, ticker))


def _volume(df: pd.DataFrame, ticker: str) -> pd.Series:
    key = ("Volume", ticker)
    if key not in df.columns:
        raise KeyError(f"Missing column {key} in adapter dataframe.")
    return _as_float_series(df[key])


def build_har_features(close: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    r = np.log(close).diff()
    rv_d = r ** 2
    rv_w = rv_d.rolling(5).mean()
    rv_m = rv_d.rolling(22).mean()
    feat = pd.DataFrame(
        {
            "log_rv_d_lag1": np.log(rv_d + 1e-8).shift(1),
            "log_rv_w_lag1": np.log(rv_w + 1e-8).shift(1),
            "log_rv_m_lag1": np.log(rv_m + 1e-8).shift(1),
        }
    )
    return feat.reindex(dates).to_numpy()


def build_vol_iohmm_dataset(
    df: pd.DataFrame,
    target_ticker: str = "SPY",
    external_tickers: Sequence[str] = ("TLT", "HYG", "UUP", "GLD"),
    rv_window_external: int = 5,
    strictly_external_inputs: bool = True,
) -> IOHMMPreparedData:
    _check_adapter_df(df)

    # adjusted prices via auto_adjust=True
    close_target = _close(df, target_ticker)
    vol_target = _volume(df, target_ticker) if ("Volume", target_ticker) in df.columns else None

    r = np.log(close_target).diff()
    rv_d = r ** 2
    rv_w = rv_d.rolling(5).mean()
    rv_m = rv_d.rolling(22).mean()
    y = build_log_rv_target(close_target, horizon=5)

    out = pd.DataFrame(index=df.index)
    out["y_log_rv"] = y

    for ticker in external_tickers:
        close = _close(df, ticker)
        ret = np.log(close).diff()
        rv = ret.rolling(rv_window_external).std() * np.sqrt(252.0)

        out[f"lag1_ret_{ticker}"] = ret.shift(1)
        out[f"lag1_log_rv_{ticker}"] = np.log(rv + 1e-8).shift(1)
        out[f"lag5_cumret_{ticker}"] = ret.rolling(5).sum().shift(1)

    if not strictly_external_inputs:
        out[f"lag1_log_rv_d_{target_ticker}"] = np.log(rv_d + 1e-8).shift(1)
        out[f"lag1_log_rv_w_{target_ticker}"] = rv_w.shift(1)
        out[f"lag1_log_rv_m_{target_ticker}"] = rv_m.shift(1)
        out[f"lag1_abs_ret_{target_ticker}"] = r.abs().shift(1)

        if vol_target is not None:
            vol_z = (vol_target - vol_target.rolling(20).mean()) / vol_target.rolling(20).std()
            out[f"lag1_vol_z_{target_ticker}"] = vol_z.shift(1)

    n_before = len(out)
    out = out.dropna().copy()
    n_dropped = n_before - len(out)
    if n_dropped > 0:
        warnings.warn(
            f"build_vol_iohmm_dataset: dropped {n_dropped} rows containing NaN.",
            stacklevel=2,
        )

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
