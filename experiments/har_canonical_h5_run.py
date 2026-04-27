"""HAR canonical h=5 track: walk_forward_har_rv on rv_gk_h5.

Same structure as the HAR baseline embedded in hmm_canonical_run.py, but
obs is built from rv_gk_h5 (via GKVolFeatures) instead of rv_gk so the
HAR-RV regression targets the h=5 forward-mean quantity directly.
train_window=252, refit_every=21.

Sanity expectation: HAR-RV RMSE at h=5 should be lower than at h=1
(target is smoother — averaging reduces noise).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._shared import get_canonical_rv_gk_h5
from HMM.features import GKVolFeatures, walk_forward_har_rv


OUT_DIR = Path(__file__).resolve().parent
SEED = 42  # walk_forward_har_rv is deterministic OLS; recorded for documentation


def _vol_pct_metrics(name: str, df: pd.DataFrame, pred_col: str) -> dict:
    y_true_var = df["y_true_rv_gk_h5"].to_numpy(dtype=np.float64)
    y_pred_var = df[pred_col].to_numpy(dtype=np.float64)
    y_true_pct = np.sqrt(y_true_var) * 100.0
    y_pred_pct = np.sqrt(y_pred_var) * 100.0
    n = len(df)
    rmse = float(np.sqrt(np.mean((y_true_pct - y_pred_pct) ** 2)))
    mae = float(np.mean(np.abs(y_true_pct - y_pred_pct)))
    qlike = float(np.mean(y_true_var / y_pred_var - np.log(y_true_var / y_pred_var) - 1.0))
    d_actual = np.sign(y_true_pct[1:] - y_true_pct[:-1])
    d_pred = np.sign(y_pred_pct[1:] - y_true_pct[:-1])
    diracc = float(np.mean(d_actual == d_pred))
    print(f"{name:<8} n={n:>4}  RMSE={rmse:7.4f}  MAE={mae:7.4f}  QLIKE={qlike:7.4f}  DirAcc={diracc:.4f}")
    return {"n": n, "rmse": rmse, "mae": mae, "qlike": qlike, "diracc": diracc}


def main() -> None:
    print(f"seed: {SEED}")
    print()

    rv_gk_h5 = get_canonical_rv_gk_h5()
    obs = GKVolFeatures().fit_transform(rv_gk_h5)
    print(f"rv_gk_h5: len={len(rv_gk_h5)}, range {rv_gk_h5.index.min().date()} → {rv_gk_h5.index.max().date()}, NaN={int(rv_gk_h5.isna().sum())}")
    print(f"obs:      len={len(obs)}, cols={list(obs.columns)}, range {obs.index.min().date()} → {obs.index.max().date()}")
    print()

    print("Running walk_forward_har_rv (train_window=252, refit_every=21) on h=5 obs...")
    har_pred = walk_forward_har_rv(obs, train_window=252, refit_every=21)
    print()

    rv_aligned = rv_gk_h5.reindex(obs.index).astype(np.float64)
    n_truth_nan = int(rv_aligned.isna().sum())
    print(f"rv_gk_h5 reindexed to obs.index: {len(rv_aligned)} rows, {n_truth_nan} NaN (expected 0)")

    n_har_nan = int(har_pred.isna().sum())
    print(f"har_pred NaN count: {n_har_nan} (expected 252 = warmup)")
    print()

    har_df = pd.DataFrame(
        {
            "date": obs.index,
            "y_true_rv_gk_h5": rv_aligned.to_numpy(dtype=np.float64),
            "y_har_h5_pred": har_pred.to_numpy(dtype=np.float64),
        }
    ).dropna(subset=["y_har_h5_pred"]).reset_index(drop=True)

    print(f"HAR h5 CSV: {len(har_df)} rows, {har_df['date'].iloc[0].date()} → {har_df['date'].iloc[-1].date()}")
    print()

    print("=== OOS metrics (volatility-percentage scale, sqrt(rv)*100) ===")
    res = _vol_pct_metrics("HAR h5", har_df, "y_har_h5_pred")
    print()

    print("Sanity check vs h=1 HAR-RV reference (RMSE=6.81 from results/unified_metrics.csv):")
    print(f"  HAR h=5 RMSE = {res['rmse']:.4f}  "
          f"({'OK — lower than h=1' if res['rmse'] < 6.81 else 'UNEXPECTED — higher than h=1; investigate'})")
    print()

    out_path = OUT_DIR / "har_canonical_h5_predictions.csv"
    har_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
