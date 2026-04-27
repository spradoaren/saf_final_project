"""IOHMM canonical h=5 track + GARCH(1,1) h=5 baseline.

Mirrors IOHMM/experiments/spy_vol_regime.py but targets log(rv_gk_h5)
(the 5-day forward-mean of GK annualized variance). K is re-selected by
ICL on the first 504 dates of the new prepared dataset (whatever K it
picks is reported and used; documented in the script output).

GARCH at h=5: forecasts the next 5 daily conditional variances and
averages them, then converts pct² → annualized fractional variance to
match the rv_gk_h5 scale.

Same canonical evaluation protocol as h=1: 2019-01-01 → 2024-12-31,
rolling 252-day train window, refit every 21 days.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from IOHMM.regimes.features import build_vol_iohmm_dataset
from IOHMM.regimes.iohmm import GaussianIOHMM
from data_preprocessing.data_adapter import YFinanceAdapter
from data_preprocessing.price_utils import extract_adjusted_close
from experiments._shared import get_canonical_rv_gk_h5


TRAIN_WINDOW = 252
REFIT_FREQ = 21
INIT_TRAIN = 504
K_VALUES = (2, 3, 4)
SEED = 42
HORIZON = 5  # forecast horizon used for the GARCH 5-day-mean forecast

_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    print(f"seed: {SEED}")
    print(f"train_window: {TRAIN_WINDOW}, refit_freq: {REFIT_FREQ}, "
          f"init_train (K-sel): {INIT_TRAIN}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    adapter = YFinanceAdapter()
    tickers = ["SPY", "TLT", "HYG", "UUP", "GLD"]
    raw = adapter.get_data(
        tickers=tickers,
        start_date="2018-06-01",
        end_date="2024-12-31",
        force_refresh=False,
    )

    rv_gk_h5 = get_canonical_rv_gk_h5()
    log_rv_gk_h5 = np.log(rv_gk_h5).rename("log_rv_gk_h5")
    print(f"rv_gk_h5: len={len(rv_gk_h5)}, NaN={int(rv_gk_h5.isna().sum())} "
          f"(expected 5 trailing)")
    print(f"log target: range {log_rv_gk_h5.dropna().index.min().date()} → "
          f"{log_rv_gk_h5.dropna().index.max().date()}")

    prepared = build_vol_iohmm_dataset(
        raw,
        target_ticker="SPY",
        external_tickers=("TLT", "HYG", "UUP", "GLD"),
        rv_window_external=5,
        strictly_external_inputs=True,
        target=log_rv_gk_h5,
    )
    X = prepared.X
    y = prepared.y
    dates = prepared.dates
    T = len(y)

    spy_close = pd.to_numeric(extract_adjusted_close(raw, "SPY"), errors="coerce").astype(float)
    r_raw = np.log(spy_close).diff().reindex(dates).to_numpy()

    print(f"\nIOHMM observations: T={T}, "
          f"range {dates[0].date()} → {dates[-1].date()}")
    print()

    truth_check = log_rv_gk_h5.reindex(dates).to_numpy()
    assert np.allclose(y, truth_check, equal_nan=False), "injected target mismatch"

    # ── K selection (one-shot on first INIT_TRAIN dates) ──────────────────
    bic_icl: dict = {}
    init_end = min(INIT_TRAIN, T)
    X_tr0 = X[:init_end]
    y_tr0 = y[:init_end]
    print(f"K selection on first {init_end} dates "
          f"({dates[0].date()} → {dates[init_end - 1].date()}):")

    for K in K_VALUES:
        m = GaussianIOHMM(n_states=K, max_iter=100, n_init=10, random_state=SEED)
        try:
            m.fit(X_tr0, y_tr0)
            bic_icl[K] = {"bic": m.bic_, "icl": m.icl_, "ll": m.best_loglik_}
            print(f"  K={K}: ll={m.best_loglik_:.2f}  BIC={m.bic_:.2f}  ICL={m.icl_:.2f}")
        except Exception as e:
            warnings.warn(f"K={K} fit failed: {e}")
            bic_icl[K] = {"bic": np.inf, "icl": np.inf, "ll": -np.inf}

    best_K = min(bic_icl, key=lambda k: bic_icl[k]["icl"])
    print(f"Selected K={best_K} by ICL on h=5 target "
          f"(documented choice; may differ from h=1's selection)")
    print()

    # ── Walk-forward: rolling 252 / refit every 21 ────────────────────────
    y_true_acc: list = []
    y_hat_iohmm: list = []
    y_hat_garch: list = []
    dates_acc: list = []

    iohmm = GaussianIOHMM(n_states=best_K, max_iter=100, n_init=10, random_state=SEED)

    print(f"Walk-forward: t in [{TRAIN_WINDOW}, {T}) stepping by {REFIT_FREQ}...")
    n_refits = 0
    for t in range(TRAIN_WINDOW, T, REFIT_FREQ):
        X_tr, y_tr = X[t - TRAIN_WINDOW:t], y[t - TRAIN_WINDOW:t]
        end = min(t + REFIT_FREQ, T)
        X_te, y_te = X[t:end], y[t:end]
        dates_te = dates[t:end]

        if len(X_te) == 0:
            break

        try:
            iohmm.fit(X_tr, y_tr)
        except Exception as e:
            warnings.warn(f"IOHMM fit failed at t={t}: {e}")
            continue

        # GARCH(1,1) on rolling 252-day return window. Inputs in percent
        # (×100); arch_model's forecast.variance is therefore in pct².
        # For h=5: forecast horizon=5 daily variances, average them, then
        # convert to annualized fractional variance to match rv_gk_h5
        # units (annualized fractional).
        r_tr = r_raw[t - TRAIN_WINDOW:t]
        r_tr = r_tr[~np.isnan(r_tr)]
        garch_res = None
        try:
            garch_res = arch_model(
                r_tr * 100, vol="Garch", p=1, q=1, dist="normal"
            ).fit(disp="off", show_warning=False)
        except Exception as e:
            warnings.warn(f"GARCH fit failed at t={t}: {e}")

        n_refits += 1

        for i, (xi, yi_true) in enumerate(zip(X_te, y_te)):
            hist_lo = t + i - TRAIN_WINDOW
            hist_hi = t + i
            X_hist = X[hist_lo:hist_hi]
            y_hist = y[hist_lo:hist_hi]
            try:
                y_h, _, _ = iohmm.forecast(X_hist, y_hist, xi)
            except Exception as e:
                warnings.warn(f"IOHMM forecast failed at t={t}+{i}: {e}")
                continue

            date_i = dates_te[i]

            if garch_res is None:
                y_h_garch_log = np.nan
            else:
                # horizon=5: 5 daily variance forecasts in pct²; average,
                # then convert pct² → annualized fractional variance:
                #   daily fractional variance      = pct² / 1e4
                #   annualized fractional variance = daily * 252
                garch_var_pct2_5d = garch_res.forecast(
                    horizon=HORIZON, reindex=False
                ).variance.values[-1, :HORIZON]
                garch_var_pct2_mean = float(np.mean(garch_var_pct2_5d))
                garch_var_annualized = garch_var_pct2_mean / 1e4 * 252.0
                y_h_garch_log = float(np.log(garch_var_annualized))

            y_true_acc.append(float(yi_true))
            y_hat_iohmm.append(float(y_h))
            y_hat_garch.append(y_h_garch_log)
            dates_acc.append(date_i)

    print(f"Completed {n_refits} refits, {len(y_true_acc)} predictions.")
    print()

    # ── Aggregate / metrics ───────────────────────────────────────────────
    y_true_log = np.asarray(y_true_acc, dtype=float)
    y_iohmm_log = np.asarray(y_hat_iohmm, dtype=float)
    y_garch_log = np.asarray(y_hat_garch, dtype=float)

    y_true_rv = np.exp(y_true_log)
    y_iohmm_rv = np.exp(y_iohmm_log)
    y_garch_rv = np.exp(y_garch_log)

    valid = ~(
        np.isnan(y_true_rv)
        | np.isnan(y_iohmm_rv)
        | np.isnan(y_garch_rv)
    )
    n_dropped = int((~valid).sum())
    print(f"Predictions: {len(valid)} total, {int(valid.sum())} kept, "
          f"{n_dropped} dropped (NaN).")

    yt_rv = y_true_rv[valid]
    yi_rv = y_iohmm_rv[valid]
    yg_rv = y_garch_rv[valid]
    dates_v = pd.DatetimeIndex([d for d, v in zip(dates_acc, valid) if v])

    def vol_pct_metrics(name: str, y_pred_rv: np.ndarray) -> dict:
        y_true_pct = np.sqrt(yt_rv) * 100.0
        y_pred_pct = np.sqrt(y_pred_rv) * 100.0
        rmse = float(np.sqrt(np.mean((y_true_pct - y_pred_pct) ** 2)))
        mae = float(np.mean(np.abs(y_true_pct - y_pred_pct)))
        ql = float(np.mean(yt_rv / y_pred_rv - np.log(yt_rv / y_pred_rv) - 1.0))
        d_actual = np.sign(y_true_pct[1:] - y_true_pct[:-1])
        d_pred = np.sign(y_pred_pct[1:] - y_true_pct[:-1])
        diracc = float(np.mean(d_actual == d_pred))
        return {"model": name, "n": len(yt_rv), "rmse": rmse,
                "mae": mae, "qlike": ql, "diracc": diracc}

    metrics_df = pd.DataFrame(
        [
            vol_pct_metrics("IOHMM h5", yi_rv),
            vol_pct_metrics("GARCH(1,1) h5", yg_rv),
        ]
    )
    print("\n=== OOS metrics (volatility-percentage scale, sqrt(rv)*100) ===")
    print(metrics_df.to_string(index=False))
    print()

    # ── CSV: cross-track CSV with iohmm + garch h5 predictions ────────────
    # Use canonical rv_gk_h5 source for y_true_rv_gk_h5 (not exp(y_true_log))
    # so it is bit-equal to the truth column in the other h=5 CSVs.
    yt_rv_canonical = rv_gk_h5.reindex(dates_v).to_numpy(dtype=np.float64)
    canonical_df = pd.DataFrame(
        {
            "date": dates_v,
            "y_true_rv_gk_h5": yt_rv_canonical,
            "y_iohmm_h5_pred": yi_rv,
            "y_garch_h5_pred": yg_rv,
        }
    )
    canonical_path = os.path.join(_EXPERIMENT_DIR, "iohmm_canonical_h5_predictions.csv")
    canonical_df.to_csv(canonical_path, index=False)
    print(f"wrote {canonical_path}")


if __name__ == "__main__":
    main()
