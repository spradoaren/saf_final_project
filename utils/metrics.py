"""Shared forecasting metrics.

Formulas are preserved exactly as they appeared in the per-model
``evaluate_forecasts`` helpers and in
``IOHMM/experiments/spy_vol_regime.py``.
"""

from __future__ import annotations

import numpy as np


def mse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return float((err ** 2).mean())


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return float(np.abs(err).mean())


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def directional_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def qlike(rv_true: np.ndarray, rv_hat: np.ndarray) -> float:
    rv_hat = np.maximum(rv_hat, 1e-12)
    rv_true = np.maximum(rv_true, 1e-12)
    return float(np.mean(rv_true / rv_hat - np.log(rv_true / rv_hat) - 1.0))


def dm_stat(e1: np.ndarray, e2: np.ndarray) -> float:
    d = e1 - e2
    return float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))


def dm_stat_hac(e1: np.ndarray, e2: np.ndarray, h: int) -> float:
    """Diebold-Mariano statistic with Newey-West HAC variance and the
    Harvey-Leybourne-Newbold (1997) small-sample correction.

    Tests H0: E[e1 - e2] = 0 (equal predictive accuracy).
    Sign convention matches ``dm_stat``: negative => model 1 has lower
    average loss.

    Newey-West HAC long-run variance (Bartlett kernel) with truncation
    lag q = h - 1:
        sigma2_LR = gamma_0 + 2 * sum_{k=1..q} (1 - k/(q+1)) * gamma_k
        gamma_k   = (1/T) * sum_{t=k+1..T} (d_t - d_bar)(d_{t-k} - d_bar)

    HLN correction (Harvey, Leybourne & Newbold 1997, Int. J. Forecasting):
        DM* = DM * sqrt((T + 1 - 2h + h(h-1)/T) / T)
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    d = e1 - e2
    T = len(d)
    d_bar = float(np.mean(d))
    d_dev = d - d_bar

    q = max(h - 1, 0)
    gamma0 = float(np.sum(d_dev ** 2)) / T
    s = gamma0
    for k in range(1, q + 1):
        gamma_k = float(np.sum(d_dev[k:] * d_dev[:-k])) / T
        weight = 1.0 - k / (q + 1)
        s += 2.0 * weight * gamma_k

    if s <= 0:
        return float("nan")
    dm = d_bar / np.sqrt(s / T)
    correction = float(np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T))
    return float(dm * correction)
