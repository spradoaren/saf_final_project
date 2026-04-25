"""Shared regime diagnostics.

Unifies ``summarize_regimes`` (IOHMM, DataFrame-based) and the
formerly British-spelled ``summarise_regimes`` (HMM_updated,
fitted-hmmlearn-model-based) under a single ``summarize_regimes`` name.

Both code paths preserve their original formulas exactly. Dispatch is
by the type of the first argument:

* ``pd.DataFrame`` -> IOHMM-style summary (occupancy + ``y`` stats +
  durations).
* otherwise        -> HMM_updated-style summary (per-state return /
  volatility / annualised approximations).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def summarize_regimes(*args, **kwargs) -> pd.DataFrame:
    if len(args) >= 1 and isinstance(args[0], pd.DataFrame):
        return _summarize_from_results(*args, **kwargs)
    return _summarize_from_model(*args, **kwargs)


def _state_run_lengths(states: np.ndarray, target_state: int) -> list[int]:
    runs: list[int] = []
    current = 0
    for s in states:
        if s == target_state:
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0
    if current > 0:
        runs.append(current)
    return runs


def _summarize_from_results(results: pd.DataFrame) -> pd.DataFrame:
    """IOHMM-style regime summary from a results DataFrame.

    Expected columns:
        y
        state
        p_state_0, ...
    """
    if "state" not in results.columns or "y" not in results.columns:
        raise ValueError("results must contain 'state' and 'y' columns.")

    summaries = []
    for state, grp in results.groupby("state"):
        durations = _state_run_lengths(results["state"].to_numpy(), int(state))
        summaries.append(
            {
                "state": int(state),
                "n_obs": len(grp),
                "fraction": len(grp) / len(results),
                "y_mean": grp["y"].mean(),
                "y_std": grp["y"].std(),
                "avg_duration": np.mean(durations) if durations else np.nan,
                "max_duration": np.max(durations) if durations else np.nan,
            }
        )

    out = pd.DataFrame(summaries).sort_values("state").reset_index(drop=True)
    return out


def _summarize_from_model(
    model,
    X: np.ndarray,
    index: pd.Index,
    return_series: pd.Series,
) -> pd.DataFrame:
    """HMM_updated-style regime summary from a fitted hmmlearn model."""

    logprob, posteriors = model.score_samples(X)
    _ = logprob
    states = posteriors.argmax(axis=1)

    df = pd.DataFrame(index=index)
    df["state"] = states
    df["return"] = return_series.reindex(index)

    rows = []

    for s in range(model.n_components):
        mask = df["state"] == s
        sub = df.loc[mask, "return"].dropna()

        mean_ret = float(sub.mean()) if len(sub) else np.nan
        vol = float(sub.std()) if len(sub) else np.nan
        sharpe_like = np.nan if vol in [0, np.nan] else mean_ret / vol

        rows.append(
            {
                "state": s,
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "avg_return": mean_ret,
                "volatility": vol,
                "annualised_return_approx": float(mean_ret * 252) if len(sub) else np.nan,
                "annualised_vol_approx": float(vol * np.sqrt(252)) if len(sub) else np.nan,
                "return_to_vol_ratio": sharpe_like,
            }
        )

    return pd.DataFrame(rows).sort_values("avg_return").reset_index(drop=True)
