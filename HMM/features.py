import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, long_window: int = 5) -> pd.DataFrame:
    """
    Triple-variate observation matrix from Zhang et al. (2019), Physica A 517:1-12.

    Inputs
    ------
    df : DataFrame with columns 'price', 'simple_return', 'log_return'
         (as produced by result.ipynb cell 1).
    long_window : look-back for the multi-day log-return (paper uses 5).

    Returns
    -------
    obs : DataFrame with three observation columns, NaN rows dropped.
        simple_return  – fractional price change  (P_t - P_{t-1}) / P_{t-1}
        log_return_1d  – 1-day log return          log(P_t / P_{t-1})
        log_return_5d  – 5-day log return          log(P_t / P_{t-5})
    """
    obs = pd.DataFrame(index=df.index)
    obs["simple_return"] = df["simple_return"]
    obs["log_return_1d"] = df["log_return"]
    obs["log_return_5d"] = np.log(df["price"] / df["price"].shift(long_window))
    return obs.dropna()