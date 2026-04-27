"""HMM canonical track: SecondOrderHMM (order=1, K=3) + HAR-RV baseline.

Walk-forward on Garman-Klass annualized variance over 2019-01-01 → 2024-12-31,
train_window=252, refit_every=21. Single canonical rv_gk source from
experiments/_shared.py so the truth column is bit-equal across all four
per-task CSV outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._shared import get_canonical_rv_gk, get_canonical_obs
from HMM.features import RVForecastWF, walk_forward_har_rv


OUT_DIR = Path(__file__).resolve().parent
SEED = 42


def _vol_pct_metrics(name: str, df: pd.DataFrame, pred_col: str) -> dict:
    """Compute RMSE/MAE/QLIKE/DirAcc on the volatility-percentage scale.

    QLIKE is computed on the variance scale (canonical Patton 2011 form).
    Directional accuracy: sign(pred_t - true_{t-1}) == sign(true_t - true_{t-1}).
    """
    y_true_var = df["y_true_rv_gk"].to_numpy(dtype=np.float64)
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

    rv_gk = get_canonical_rv_gk()
    obs = get_canonical_obs()
    print(f"rv_gk: len={len(rv_gk)}, range {rv_gk.index.min().date()} → {rv_gk.index.max().date()}")
    print(f"obs:   len={len(obs)}, cols={list(obs.columns)}, range {obs.index.min().date()} → {obs.index.max().date()}")
    print()

    wf = RVForecastWF(
        n_states=3,
        order=1,
        train_window=252,
        refit_every=21,
        n_iter=100,
        random_state=SEED,
        forecast_col=0,
    )
    print("Running RVForecastWF (n_states=3, order=1, train_window=252, refit_every=21)...")
    hmm_pred = wf.run(obs)

    print("Running walk_forward_har_rv (train_window=252, refit_every=21)...")
    har_pred = walk_forward_har_rv(obs, train_window=252, refit_every=21)
    print()

    rv_aligned = rv_gk.reindex(obs.index).astype(np.float64)
    n_truth_nan = int(rv_aligned.isna().sum())
    print(f"rv_gk reindexed to obs.index: {len(rv_aligned)} rows, {n_truth_nan} NaN (expected 0)")

    n_hmm_nan = int(hmm_pred.isna().sum())
    n_har_nan = int(har_pred.isna().sum())
    print(f"hmm_pred NaN count: {n_hmm_nan} (expected 252 = warmup)")
    print(f"har_pred NaN count: {n_har_nan} (expected 252 = warmup)")
    print()

    hmm_df = pd.DataFrame(
        {
            "date": obs.index,
            "y_true_rv_gk": rv_aligned.to_numpy(dtype=np.float64),
            "y_hmm_pred": hmm_pred.to_numpy(dtype=np.float64),
        }
    ).dropna(subset=["y_hmm_pred"]).reset_index(drop=True)

    har_df = pd.DataFrame(
        {
            "date": obs.index,
            "y_true_rv_gk": rv_aligned.to_numpy(dtype=np.float64),
            "y_har_pred": har_pred.to_numpy(dtype=np.float64),
        }
    ).dropna(subset=["y_har_pred"]).reset_index(drop=True)

    print(f"HMM CSV: {len(hmm_df)} rows, {hmm_df['date'].iloc[0].date()} → {hmm_df['date'].iloc[-1].date()}")
    print(f"HAR CSV: {len(har_df)} rows, {har_df['date'].iloc[0].date()} → {har_df['date'].iloc[-1].date()}")
    print()

    print("=== OOS metrics (volatility-percentage scale, sqrt(rv)*100) ===")
    _vol_pct_metrics("HMM", hmm_df, "y_hmm_pred")
    _vol_pct_metrics("HAR", har_df, "y_har_pred")
    print()

    hmm_path = OUT_DIR / "hmm_canonical_predictions.csv"
    har_path = OUT_DIR / "har_canonical_predictions.csv"
    hmm_df.to_csv(hmm_path, index=False)
    har_df.to_csv(har_path, index=False)
    print(f"wrote {hmm_path}")
    print(f"wrote {har_path}")


if __name__ == "__main__":
    main()
