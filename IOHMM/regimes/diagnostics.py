from __future__ import annotations

import warnings

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


def check_regime_stationarity(model, X: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    if not getattr(model, "is_fitted_", False):
        raise RuntimeError("Model is not fitted.")

    X = np.asarray(X, dtype=float)
    Xs = model._transform_X(X)
    log_trans = model.transition_model.log_transition_tensor(Xs)
    mean_transmat = np.exp(log_trans).mean(axis=0)

    eigvals, eigvecs = np.linalg.eig(mean_transmat.T)
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    stat = np.real(eigvecs[:, idx])
    s = stat.sum()
    if abs(s) < 1e-12:
        warnings.warn("Stationary eigenvector sums to ~0; transition matrix may be degenerate.", stacklevel=2)
        return np.full(model.n_states, np.nan)
    stat = stat / s

    for k, p in enumerate(stat):
        if p < threshold:
            warnings.warn(
                f"Regime {k} stationary probability {p:.4f} < {threshold}: near-absorbing or near-transient.",
                stacklevel=2,
            )
    return stat


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
