"""HMM canonical h=5 track: SecondOrderHMM (order=1, K=3) on rv_gk_h5.

Walk-forward on the 5-day forward-average Garman-Klass annualized variance
target (rv_gk_h5) over 2019-01-01 → 2024-12-31, train_window=252,
refit_every=21. Same architecture as hmm_canonical_run.py; obs is built
from rv_gk_h5 (via GKVolFeatures) instead of rv_gk so the HMM emission
means target the h=5 quantity directly. K=3 is held fixed (matching the
h=1 spec / Zhang et al.).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._shared import get_canonical_rv_gk_h5
from HMM.features import GKVolFeatures, RVForecastWF


OUT_DIR = Path(__file__).resolve().parent
SEED = 42


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

    wf = RVForecastWF(
        n_states=3,
        order=1,
        train_window=252,
        refit_every=21,
        n_iter=100,
        random_state=SEED,
        forecast_col=0,
    )
    print("Running RVForecastWF (n_states=3, order=1, train_window=252, refit_every=21) on h=5 obs...")
    hmm_pred = wf.run(obs)

    rv_aligned = rv_gk_h5.reindex(obs.index).astype(np.float64)
    n_truth_nan = int(rv_aligned.isna().sum())
    print(f"rv_gk_h5 reindexed to obs.index: {len(rv_aligned)} rows, {n_truth_nan} NaN (expected 0)")

    n_hmm_nan = int(hmm_pred.isna().sum())
    print(f"hmm_pred NaN count: {n_hmm_nan} (expected 252 = warmup)")
    print()

    hmm_df = pd.DataFrame(
        {
            "date": obs.index,
            "y_true_rv_gk_h5": rv_aligned.to_numpy(dtype=np.float64),
            "y_hmm_h5_pred": hmm_pred.to_numpy(dtype=np.float64),
        }
    ).dropna(subset=["y_hmm_h5_pred"]).reset_index(drop=True)

    print(f"HMM h5 CSV: {len(hmm_df)} rows, {hmm_df['date'].iloc[0].date()} → {hmm_df['date'].iloc[-1].date()}")
    print()

    print("=== OOS metrics (volatility-percentage scale, sqrt(rv)*100) ===")
    _vol_pct_metrics("HMM h5", hmm_df, "y_hmm_h5_pred")
    print()

    out_path = OUT_DIR / "hmm_canonical_h5_predictions.csv"
    hmm_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
