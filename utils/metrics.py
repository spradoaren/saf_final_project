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
