from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def summarize_regimes(results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize regime occupancy and target distribution from results frame.

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