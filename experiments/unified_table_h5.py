"""Unified cross-track h=5 comparison table.

Reads the four canonical h=5 per-track CSVs, verifies y_true_rv_gk_h5
bit-equality, inner-joins on date, adds a backward 5-day-mean
persistence baseline, computes five metrics under a single canonical
formulation, and adds HAC-DM (Newey-West, truncation lag h-1=4, with
HLN small-sample correction) of each model vs HAR-RV.

Persistence at h=5 (design decision):
    y_pred[t] = mean(rv_gk[t-4..t])  -- backward 5-day mean of daily
    annualized variance.
This is the natural analogue of the h=1 persistence baseline
``y_pred[t] = rv_gk[t-1]`` rescaled to match the target's window
length. We do NOT also report the stale-daily ``rv_gk[t-1]`` variant.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from experiments._shared import get_canonical_rv_gk
from utils.metrics import dm_stat, dm_stat_hac


EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"

H = 5  # forecast horizon — Newey-West truncation lag is H-1 = 4


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(EXPERIMENTS_DIR / name, parse_dates=["date"])


def _vol_pct(rv: np.ndarray) -> np.ndarray:
    return np.sqrt(rv) * 100.0


def _qlike_pred_over_true(rv_true: np.ndarray, rv_pred: np.ndarray) -> float:
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
    print("=== Reading per-track h=5 CSVs ===")
    hmm = _read_csv("hmm_canonical_h5_predictions.csv")
    har = _read_csv("har_canonical_h5_predictions.csv")
    iohmm = _read_csv("iohmm_canonical_h5_predictions.csv")
    msar = _read_csv("msar_canonical_h5_predictions.csv")

    sources = [
        ("HMM h5",   hmm),
        ("HAR h5",   har),
        ("IOHMM h5", iohmm),
        ("MS-AR h5", msar),
    ]
    for name, df in sources:
        print(f"  {name:<12} rows={len(df):>4}  "
              f"{df['date'].min().date()} → {df['date'].max().date()}  "
              f"cols={list(df.columns)}")
    print()

    print("=== Bit-equality of y_true_rv_gk_h5 across all four sources ===")
    truth_dfs = [(name, df[["date", "y_true_rv_gk_h5"]]) for name, df in sources]
    any_diff = False
    for (n1, d1), (n2, d2) in combinations(truth_dfs, 2):
        merged = d1.merge(d2, on="date", suffixes=("_1", "_2"))
        diff = (merged["y_true_rv_gk_h5_1"] - merged["y_true_rv_gk_h5_2"]).abs()
        max_diff = float(diff.max())
        flag = "OK" if max_diff == 0.0 else "DIFFERS"
        print(f"  {n1:<10} vs {n2:<10}  n={len(merged):>4}  "
              f"max|diff|={max_diff:.3e}  [{flag}]")
        if max_diff != 0.0:
            any_diff = True
    if any_diff:
        raise SystemExit("STOP: y_true_rv_gk_h5 is not bit-equal across all sources.")
    print()

    print("=== Inner-joining on date ===")
    joined = (
        hmm[["date", "y_true_rv_gk_h5", "y_hmm_h5_pred"]]
        .merge(har[["date", "y_har_h5_pred"]], on="date")
        .merge(iohmm[["date", "y_iohmm_h5_pred", "y_garch_h5_pred"]], on="date")
        .merge(msar[["date", "y_msar_h5_pred"]], on="date")
    )

    # Persistence baseline at h=5: y_pred[t] = mean(rv_gk[t-4..t])
    rv_gk = get_canonical_rv_gk()
    persistence_full = rv_gk.rolling(window=5, min_periods=5).mean()
    persistence = persistence_full.reindex(pd.DatetimeIndex(joined["date"]))
    joined["y_persistence_h5_pred"] = persistence.to_numpy(dtype=np.float64)

    n_persistence_nan = int(joined["y_persistence_h5_pred"].isna().sum())
    if n_persistence_nan > 0:
        print(f"  WARNING: {n_persistence_nan} persistence NaN — should not occur "
              f"unless joined contains the first 4 dates of rv_gk")
    print(f"  joined rows: {len(joined)}, "
          f"date range {joined['date'].iloc[0].date()} → {joined['date'].iloc[-1].date()}")
    print()

    pred_cols = {
        "HMM h5":         "y_hmm_h5_pred",
        "IOHMM h5":       "y_iohmm_h5_pred",
        "MS-AR h5":       "y_msar_h5_pred",
        "GARCH(1,1) h5":  "y_garch_h5_pred",
        "HAR-RV h5":      "y_har_h5_pred",
        "Persistence h5": "y_persistence_h5_pred",
    }

    rv_true = joined["y_true_rv_gk_h5"].to_numpy(dtype=np.float64)
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

    # ── HAC-DM (and standard DM for comparison) vs HAR-RV ────────────────
    err_har = (rv_true - joined["y_har_h5_pred"].to_numpy(dtype=np.float64)) ** 2
    dm_iid: dict = {}
    dm_hac: dict = {}
    for model_name, col in pred_cols.items():
        if model_name == "HAR-RV h5":
            dm_iid[model_name] = np.nan
            dm_hac[model_name] = np.nan
            continue
        err_m = (rv_true - joined[col].to_numpy(dtype=np.float64)) ** 2
        dm_iid[model_name] = dm_stat(err_m, err_har)
        dm_hac[model_name] = dm_stat_hac(err_m, err_har, h=H)

    metrics_df["dm_vs_har_iid"] = metrics_df["model"].map(dm_iid)
    metrics_df["dm_vs_har_hac"] = metrics_df["model"].map(dm_hac)

    metrics_df = metrics_df[[
        "model", "n", "rmse_volpct", "mae_volpct", "qlike", "diracc",
        "mz_r2_log", "dm_vs_har_iid", "dm_vs_har_hac",
    ]]
    metrics_df = metrics_df.sort_values("rmse_volpct", ascending=True).reset_index(drop=True)
    print()
    print(metrics_df.to_string(index=False))
    print()

    print("=== Best model per metric ===")
    best_descriptors = {
        "rmse_volpct": ("min", "lower is better"),
        "mae_volpct":  ("min", "lower is better"),
        "qlike":       ("min", "lower is better"),
        "diracc":      ("max", "higher is better"),
        "mz_r2_log":   ("max", "higher is better"),
    }
    for metric, (direction, hint) in best_descriptors.items():
        if direction == "min":
            best_idx = metrics_df[metric].idxmin()
        else:
            best_idx = metrics_df[metric].idxmax()
        winner = metrics_df.loc[best_idx]
        print(f"  {metric:<12}  ({hint})  →  {winner['model']:<16}  value={winner[metric]:.4f}")
    print()

    print("=== Diebold-Mariano vs HAR-RV (h=5, both i.i.d. and HAC) ===")
    print("  HAC: Newey-West truncation lag = h-1 = 4, HLN small-sample correction")
    print("  Sign convention: negative => model better than HAR-RV")
    print()
    for model_name, col in pred_cols.items():
        if model_name == "HAR-RV h5":
            continue
        d_iid = dm_iid[model_name]
        d_hac = dm_hac[model_name]
        sign_hac = "model better" if d_hac < 0 else "HAR better"
        print(f"  {model_name:<16}  iid DM={d_iid:+8.4f}   "
              f"HAC DM={d_hac:+8.4f}   [{sign_hac}, by HAC]")
    print()

    RESULTS_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / "unified_metrics_h5.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"wrote {metrics_path}")

    long_rows = []
    for model_name, col in pred_cols.items():
        for date, yt, yp in zip(joined["date"], joined["y_true_rv_gk_h5"], joined[col]):
            long_rows.append({
                "date": date,
                "model": model_name,
                "y_true_rv_gk_h5": yt,
                "y_pred": yp,
            })
    long_df = pd.DataFrame(long_rows)
    long_path = RESULTS_DIR / "unified_predictions_h5.csv"
    long_df.to_csv(long_path, index=False)
    print(f"wrote {long_path}  "
          f"({len(long_df)} rows = {len(pred_cols)} models × {n} dates)")


if __name__ == "__main__":
    main()
