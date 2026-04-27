"""MS-AR canonical h=5 track:
MarkovAutoregression(k=2, order=1, sw_var=True, trend='c') on
log(rv_gk_h5), rolling 252-day window, refit every 21 days. Predictions
exp()-ed to rv_gk_h5 units for cross-track comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._shared import get_canonical_rv_gk_h5
from Markov_Switching_AR.model_utils import rolling_forecast_msar_rv


OUT_DIR = Path(__file__).resolve().parent
SEED = 42


def _vol_pct_metrics(name: str, df: pd.DataFrame, pred_col: str) -> dict:
    yt = df["y_true_rv_gk_h5"].to_numpy(dtype=np.float64)
    yp = df[pred_col].to_numpy(dtype=np.float64)
    yt_pct = np.sqrt(yt) * 100.0
    yp_pct = np.sqrt(yp) * 100.0
    n = len(df)
    rmse = float(np.sqrt(np.mean((yt_pct - yp_pct) ** 2)))
    mae = float(np.mean(np.abs(yt_pct - yp_pct)))
    qlike = float(np.mean(yt / yp - np.log(yt / yp) - 1.0))
    da = float(np.mean(np.sign(yt_pct[1:] - yt_pct[:-1]) == np.sign(yp_pct[1:] - yt_pct[:-1])))
    print(f"  {name:<22} n={n:>4} RMSE={rmse:7.4f} MAE={mae:7.4f} QLIKE={qlike:7.4f} DirAcc={da:.4f}")
    return {"name": name, "n": n, "rmse": rmse, "mae": mae, "qlike": qlike, "diracc": da}


def main() -> None:
    print(f"seed: {SEED} (MarkovAutoregression MLE is deterministic; recorded for documentation)")
    print()

    rv_gk_h5 = get_canonical_rv_gk_h5()
    # Drop the trailing 5 NaN before taking log (rolling_forecast_msar_rv
    # also dropna-s internally).
    log_rv_h5 = np.log(rv_gk_h5.dropna()).rename("log_rv_gk_h5")
    print(f"rv_gk_h5: len={len(rv_gk_h5)}, NaN={int(rv_gk_h5.isna().sum())} (expected 5 trailing)")
    print(f"log_rv_h5 (after dropna): len={len(log_rv_h5)}, "
          f"range {log_rv_h5.index.min().date()} → {log_rv_h5.index.max().date()}")
    print()

    print("Running rolling_forecast_msar_rv on log(rv_gk_h5)")
    print("  k_regimes=2, order=1, switching_variance=True, trend='c'")
    print("  rolling train_window=252, refit_every=21")
    out = rolling_forecast_msar_rv(
        log_rv_series=log_rv_h5,
        train_window=252,
        refit_every=21,
        k_regimes=2,
        order=1,
        switching_variance=True,
        trend="c",
    )

    n_refits = out.attrs["n_refits"]
    n_failed = out.attrs["n_failed"]
    n_non_converged = out.attrs["n_non_converged"]
    fail_rate = n_failed / max(n_refits, 1)
    nc_rate = n_non_converged / max(n_refits, 1)
    print(f"refits: {n_refits}")
    print(f"  hard failures (raised exception):  {n_failed} ({fail_rate:.1%})")
    print(f"  soft non-converged (params used):  {n_non_converged} ({nc_rate:.1%})")
    if fail_rate > 0.05:
        print(f"WARNING: hard failure rate {fail_rate:.1%} exceeds 5% threshold")
    print()

    valid_dates = out["y_msar_pred_ffill"].dropna().index
    df = pd.DataFrame(
        {
            "date": valid_dates,
            "y_true_rv_gk_h5": rv_gk_h5.reindex(valid_dates).to_numpy(dtype=np.float64),
            "y_msar_h5_pred": out.loc[valid_dates, "y_msar_pred_ffill"].to_numpy(dtype=np.float64),
        }
    )
    print(f"OOS rows: {len(df)}, range {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print()

    print("=== OOS metrics (volatility-percentage scale, sqrt(rv)*100) ===")
    _vol_pct_metrics("MS-AR h5 (ffill)", df, "y_msar_h5_pred")

    if n_failed > 0:
        clean_dates = out["y_msar_pred"].dropna().index
        df_clean = pd.DataFrame(
            {
                "date": clean_dates,
                "y_true_rv_gk_h5": rv_gk_h5.reindex(clean_dates).to_numpy(dtype=np.float64),
                "y_msar_h5_pred": out.loc[clean_dates, "y_msar_pred"].to_numpy(dtype=np.float64),
            }
        )
        _vol_pct_metrics("MS-AR h5 (clean)", df_clean, "y_msar_h5_pred")
    else:
        print("  (no fit failures — clean and ffill columns are identical)")

    out_path = OUT_DIR / "msar_canonical_h5_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
