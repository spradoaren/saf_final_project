from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from utils.diagnostics import summarize_regimes

__all__ = ["summarize_regimes", "check_regime_stationarity", "_state_run_lengths"]


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
