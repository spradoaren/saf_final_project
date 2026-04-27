"""Unified cross-track comparison table.

Reads the four canonical per-track CSVs, verifies y_true_rv_gk
bit-equality, inner-joins on date, adds a naive persistence baseline,
computes five metrics under a single canonical formulation, and writes
results/unified_metrics.csv (sorted by RMSE) plus a long-format
results/unified_predictions.csv for downstream analysis.

Canonical metric formulations (used uniformly across all 6 models):
  - RMSE on volatility-percent scale: sqrt(mean((vol_t - vol_p)^2))
  - MAE on volatility-percent scale:  mean(|vol_t - vol_p|)
  - QLIKE on variance scale (project convention; teammate notebook):
      mean(rv_pred / rv_true - log(rv_pred / rv_true) - 1)
    NOTE: This is the rv_pred/rv_true asymmetry, opposite of the
    Patton 2011 form used by utils/metrics.qlike. Penalizes
    overprediction more heavily than underprediction.
  - Directional accuracy: sign(vol_pred[t] - vol_true[t-1])
                        == sign(vol_true[t] - vol_true[t-1])
  - MZ R^2 on log scale: corrcoef(log(rv_pred), log(rv_true))^2
"""
from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._shared import get_canonical_rv_gk
from utils.metrics import dm_stat


EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS_DIR / name, parse_dates=["date"])


def _vol_pct(rv: np.ndarray) -> np.ndarray:
    return np.sqrt(rv) * 100.0


def _qlike_pred_over_true(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
    """Project-convention QLIKE: mean(p/t - log(p/t) - 1)."""
    r = rv_pred / rv_true
    return float(np.mean(r - np.log(r) - 1.0))


def _directional_accuracy(vol_true: np.ndarray, vol_pred: np.ndarray) -> float:
    actual = np.sign(vol_true[1:] - vol_true[:-1])
    pred = np.sign(vol_pred[1:] - vol_true[:-1])
    return float(np.mean(actual == pred))


def _mz_r2_log(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
    return float(np.corrcoef(np.log(rv_pred), np.log(rv_true))[0, 1] ** 2)


def _all_metrics(rv_true: np.ndarray, rv_pred: np.ndarray) -> dict:
    vol_t = _vol_pct(rv_true)
    vol_p = _vol_pct(rv_pred)
    err_pct = vol_t - vol_p
    return {
        "rmse_volpct": float(np.sqrt(np.mean(err_pct ** 2))),
        "mae_volpct": float(np.mean(np.abs(err_pct))),
        "qlike": _qlike_pred_over_true(rv_true, rv_pred),
        "diracc": _directional_accuracy(vol_t, vol_p),
        "mz_r2_log": _mz_r2_log(rv_true, rv_pred),
    }


def main() -> None:
    print("=== Reading per-track CSVs ===")
    hmm = _read_csv("hmm_canonical_predictions.csv")
    har = _read_csv("har_canonical_predictions.csv")
    iohmm = _read_csv("iohmm_canonical_predictions.csv")
    msar = _read_csv("msar_canonical_predictions.csv")

    sources = [
        ("HMM (Task A)", hmm),
        ("HAR (Task A)", har),
        ("IOHMM (Task B)", iohmm),
        ("MS-AR (Task C)", msar),
    ]
    for name, df in sources:
        print(f"  {name:<18} rows={len(df):>4}  "
              f"{df['date'].min().date()} → {df['date'].max().date()}  "
              f"cols={list(df.columns)}")
    print()

    print("=== Bit-equality of y_true_rv_gk across all four sources ===")
    truth_dfs = [(name, df[["date", "y_true_rv_gk"]]) for name, df in sources]
    any_diff = False
    for (n1, d1), (n2, d2) in combinations(truth_dfs, 2):
        merged = d1.merge(d2, on="date", suffixes=("_1", "_2"))
        diff = (merged["y_true_rv_gk_1"] - merged["y_true_rv_gk_2"]).abs()
        max_diff = float(diff.max())
        flag = "OK" if max_diff == 0.0 else "DIFFERS"
        print(f"  {n1:<16} vs {n2:<16}  n={len(merged):>4}  max|diff|={max_diff:.3e}  [{flag}]")
        if max_diff != 0.0:
            any_diff = True
    if any_diff:
        raise SystemExit("STOP: y_true_rv_gk is not bit-equal across all sources.")
    print()

    print("=== Inner-joining on date ===")
    joined = (
        hmm[["date", "y_true_rv_gk", "y_hmm_pred"]]
        .merge(har[["date", "y_har_pred"]], on="date")
        .merge(
            iohmm[["date", "y_iohmm_pred", "y_har_pred", "y_garch_pred"]],
            on="date", suffixes=("", "_iohmm"),
        )
        .merge(msar[["date", "y_msar_pred"]], on="date")
    )

    har_diff = (joined["y_har_pred"] - joined["y_har_pred_iohmm"]).abs().max()
    print(f"  cross-track HAR check: max|y_har_pred(A) - y_har_pred(B)| = {har_diff:.3e}")
    if har_diff != 0.0:
        raise SystemExit("STOP: HAR predictions disagree between Task A and Task B.")
    joined = joined.drop(columns=["y_har_pred_iohmm"])

    rv_gk = get_canonical_rv_gk()
    persistence = rv_gk.shift(1).reindex(pd.DatetimeIndex(joined["date"]))
    joined["y_persistence_pred"] = persistence.to_numpy(dtype=np.float64)

    n_persistence_nan = int(joined["y_persistence_pred"].isna().sum())
    if n_persistence_nan > 0:
        print(f"  WARNING: {n_persistence_nan} persistence NaN — will not occur unless joined contains rv_gk's first date")
    print(f"  joined rows: {len(joined)}, "
          f"date range {joined['date'].iloc[0].date()} → {joined['date'].iloc[-1].date()}")
    print()

    pred_cols = {
        "HMM": "y_hmm_pred",
        "IOHMM": "y_iohmm_pred",
        "MS-AR": "y_msar_pred",
        "GARCH(1,1)": "y_garch_pred",
        "HAR-RV": "y_har_pred",
        "Persistence": "y_persistence_pred",
    }

    rv_true = joined["y_true_rv_gk"].to_numpy(dtype=np.float64)
    n = len(rv_true)
    print(f"=== Computing metrics for {len(pred_cols)} models on n={n} observations ===")

    rows = []
    for model_name, col in pred_cols.items():
        rv_pred = joined[col].to_numpy(dtype=np.float64)
        m = _all_metrics(rv_true, rv_pred)
        m["model"] = model_name
        m["n"] = n
        rows.append(m)
    metrics_df = pd.DataFrame(rows)
    metrics_df = metrics_df[["model", "n", "rmse_volpct", "mae_volpct", "qlike", "diracc", "mz_r2_log"]]
    metrics_df = metrics_df.sort_values("rmse_volpct", ascending=True).reset_index(drop=True)
    print()
    print(metrics_df.to_string(index=False))
    print()

    print("=== Best model per metric ===")
    best_descriptors = {
        "rmse_volpct": ("min", "lower is better"),
        "mae_volpct": ("min", "lower is better"),
        "qlike": ("min", "lower is better"),
        "diracc": ("max", "higher is better"),
        "mz_r2_log": ("max", "higher is better"),
    }
    for metric, (direction, hint) in best_descriptors.items():
        if direction == "min":
            best_idx = metrics_df[metric].idxmin()
        else:
            best_idx = metrics_df[metric].idxmax()
        winner = metrics_df.loc[best_idx]
        print(f"  {metric:<12}  ({hint})  →  {winner['model']:<14}  value={winner[metric]:.4f}")
    print()

    print("=== Diebold-Mariano vs HAR-RV (i.i.d. variance — NOT HAC corrected) ===")
    err_har = (rv_true - joined["y_har_pred"].to_numpy(dtype=np.float64)) ** 2
    for model_name, col in pred_cols.items():
        if model_name == "HAR-RV":
            continue
        err_m = (rv_true - joined[col].to_numpy(dtype=np.float64)) ** 2
        dm = dm_stat(err_m, err_har)
        sign = "model better" if dm < 0 else "HAR better"
        print(f"  DM({model_name:<14} − HAR-RV) on rv-scale squared error  =  {dm:+8.4f}  [{sign}]")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / "unified_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"wrote {metrics_path}")

    long_rows = []
    for model_name, col in pred_cols.items():
        for date, yt, yp in zip(joined["date"], joined["y_true_rv_gk"], joined[col]):
            long_rows.append({
                "date": date,
                "model": model_name,
                "y_true_rv_gk": yt,
                "y_pred": yp,
            })
    long_df = pd.DataFrame(long_rows)
    long_path = RESULTS_DIR / "unified_predictions.csv"
    long_df.to_csv(long_path, index=False)
    print(f"wrote {long_path}  ({len(long_df)} rows = {len(pred_cols)} models × {n} dates)")


if __name__ == "__main__":
    main()
