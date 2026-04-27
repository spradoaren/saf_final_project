"""MS-AR alternative spec robustness check:
MarkovAutoregression(k=2, order=1, switching_ar=True, switching_variance=True, trend='c')
on log(rv_gk), rolling 252-day window, refit every 21 days. Predictions
exp()-ed to rv_gk units for cross-track comparison.

Compared to msar_canonical_run.py, this allows the AR coefficient to vary
by regime — testing whether the degenerate fit observed in the canonical
spec (sigma2[0] -> 0, ar.L1 ~ 0.6 collapsed onto regime 1) is a model-
class limitation or a specification artifact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments._shared import get_canonical_rv_gk
from Markov_Switching_AR.model_utils import (
    fit_msar_model,
    rolling_forecast_msar_rv,
)


OUT_DIR = Path(__file__).resolve().parent
SEED = 42


def _vol_pct_metrics(name: str, df: pd.DataFrame, pred_col: str) -> dict:
    yt = df["y_true_rv_gk"].to_numpy(dtype=np.float64)
    yp = df[pred_col].to_numpy(dtype=np.float64)
    yt_pct = np.sqrt(yt) * 100.0
    yp_pct = np.sqrt(yp) * 100.0
    n = len(df)
    rmse = float(np.sqrt(np.mean((yt_pct - yp_pct) ** 2)))
    mae = float(np.mean(np.abs(yt_pct - yp_pct)))
    qlike = float(np.mean(yt / yp - np.log(yt / yp) - 1.0))
    da = float(np.mean(np.sign(yt_pct[1:] - yt_pct[:-1]) == np.sign(yp_pct[1:] - yt_pct[:-1])))
    print(f"  {name:<28} n={n:>4} RMSE={rmse:7.4f} MAE={mae:7.4f} QLIKE={qlike:9.4f} DirAcc={da:.4f}")
    return {"name": name, "n": n, "rmse": rmse, "mae": mae, "qlike": qlike, "diracc": da}


def _print_first_refit_params(log_rv: pd.Series, train_window: int, start_offset: int = 0) -> None:
    """Try to fit MS-AR (switching_ar=True) on a representative window and
    print params. Falls forward through start_offset values until a fit
    succeeds (the first 2019-only window often hits SVD/EM init issues
    with the extra regime-specific AR parameter)."""
    import warnings as _w
    for offset in range(start_offset, len(log_rv) - train_window, 21):
        train = log_rv.iloc[offset: offset + train_window]
        print(f"=== Representative fit on dates "
              f"{train.index[0].date()} → {train.index[-1].date()} (offset={offset}) ===")
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                res = fit_msar_model(
                    returns=train,
                    k_regimes=2,
                    order=1,
                    switching_ar=True,
                    switching_variance=True,
                    trend="c",
                )
        except Exception as e:
            print(f"  fit raised {type(e).__name__}: {e}; trying next window…")
            print()
            continue
        params = pd.Series(res.params, index=res.model.param_names)
        print("Param names + values:")
        for name, val in params.items():
            print(f"  {name:<20} = {val:.6e}")
        mle = getattr(res, "mle_retvals", {})
        print(f"converged: {mle.get('converged')},  warnflag: {mle.get('warnflag')}")
        print()
        return
    print("  no representative window succeeded across the whole series.")
    print()


def main() -> None:
    print(f"seed: {SEED} (MarkovAutoregression MLE is deterministic; recorded for documentation)")
    print()

    rv_gk = get_canonical_rv_gk()
    log_rv = np.log(rv_gk).rename("log_rv_gk")
    print(f"log(rv_gk): len={len(log_rv)}, range {log_rv.index.min().date()} → {log_rv.index.max().date()}")
    print()

    # Representative fit BEFORE the walk-forward so we can compare to the
    # canonical-spec fitted params from Task C.
    _print_first_refit_params(log_rv, train_window=252)

    print("Running rolling_forecast_msar_rv (k=2, order=1, switching_ar=True, sw_var=True, trend='c')")
    print("  rolling train_window=252, refit_every=21")
    out = rolling_forecast_msar_rv(
        log_rv_series=log_rv,
        train_window=252,
        refit_every=21,
        k_regimes=2,
        order=1,
        switching_variance=True,
        switching_ar=True,
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
            "y_true_rv_gk": rv_gk.reindex(valid_dates).to_numpy(dtype=np.float64),
            "y_msar_alt_pred": out.loc[valid_dates, "y_msar_pred_ffill"].to_numpy(dtype=np.float64),
        }
    )
    print(f"OOS rows: {len(df)}, range {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
    print()

    print("=== OOS metrics (volatility-percentage scale, sqrt(rv)*100) ===")
    _vol_pct_metrics("MS-AR alt (ffill)", df, "y_msar_alt_pred")
    if n_failed > 0:
        clean_dates = out["y_msar_pred"].dropna().index
        df_clean = pd.DataFrame(
            {
                "date": clean_dates,
                "y_true_rv_gk": rv_gk.reindex(clean_dates).to_numpy(dtype=np.float64),
                "y_msar_alt_pred": out.loc[clean_dates, "y_msar_pred"].to_numpy(dtype=np.float64),
            }
        )
        _vol_pct_metrics("MS-AR alt (clean)", df_clean, "y_msar_alt_pred")
    else:
        print("  (no fit failures — clean and ffill columns are identical)")
    print()

    # Prediction distribution diagnostic
    yt = df["y_true_rv_gk"].to_numpy(dtype=np.float64)
    yp = df["y_msar_alt_pred"].to_numpy(dtype=np.float64)
    print("Prediction distribution (vol-pct scale):")
    print(f"  truth mean={np.sqrt(yt).mean()*100:6.2f}%  std={np.sqrt(yt).std()*100:6.2f}%  "
          f"max={np.sqrt(yt).max()*100:6.2f}%")
    print(f"  pred  mean={np.sqrt(yp).mean()*100:6.2f}%  std={np.sqrt(yp).std()*100:6.2f}%  "
          f"max={np.sqrt(yp).max()*100:6.2f}%")
    ratio = yt / yp
    print(f"  truth/pred ratio: median={np.median(ratio):.2f}  q95={np.quantile(ratio, 0.95):.2f}  "
          f"max={ratio.max():.2f}")

    out_path = OUT_DIR / "msar_alt_canonical_predictions.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
