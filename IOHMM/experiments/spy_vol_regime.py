from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

from IOHMM.regimes.features import build_har_features, build_vol_iohmm_dataset  # noqa: F401
from IOHMM.regimes.iohmm import GaussianIOHMM
from data_preprocessing.data_adapter import YFinanceAdapter
from data_preprocessing.price_utils import extract_adjusted_close
from experiments._shared import get_canonical_obs, get_canonical_rv_gk
from HMM.features import walk_forward_har_rv
from utils.metrics import dm_stat, qlike


# Canonical evaluation protocol matched to the HMM track:
#   - Period: 2019-01-01 → 2024-12-31
#   - Rolling 252-day train window
#   - Refit every 21 days
#   - Target: log(rv_gk) (Garman-Klass annualized variance, log-transformed)
#   - K selection on the first 504 dates of the prepared dataset (one-shot)
TRAIN_WINDOW = 252
REFIT_FREQ = 21
INIT_TRAIN = 504
K_VALUES = (2, 3, 4)
SEED = 42


_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_EXPERIMENT_DIR))
_CANONICAL_OUT_DIR = os.path.join(_PROJECT_ROOT, "experiments")


def per_regime_metrics(y_true: np.ndarray, y_hat: np.ndarray, dom_state: np.ndarray, K: int) -> pd.DataFrame:
    rows = []
    for k in range(K):
        mask = dom_state == k
        n = int(mask.sum())
        if n < 2:
            rows.append({"state": k, "n": n, "mse": np.nan, "qlike": np.nan})
            continue
        rv_t = np.exp(y_true[mask])
        rv_h = np.exp(y_hat[mask])
        rows.append(
            {
                "state": k,
                "n": n,
                "mse": float(np.mean((y_true[mask] - y_hat[mask]) ** 2)),
                "qlike": qlike(rv_t, rv_h),
            }
        )
    return pd.DataFrame(rows)


def main(
    train_window: int = TRAIN_WINDOW,
    refit_freq: int = REFIT_FREQ,
    init_train: int = INIT_TRAIN,
    k_values=K_VALUES,
    seed: int = SEED,
) -> None:
    print(f"seed: {seed}")
    print(f"train_window: {train_window}, refit_freq: {refit_freq}, init_train (K-sel): {init_train}")
    print()

    # ── Data ──────────────────────────────────────────────────────────────
    # Fetch with 6-month lead time so external-ticker rolling features are
    # warm by the time the canonical rv_gk series begins (2019-01-02).
    adapter = YFinanceAdapter()
    tickers = ["SPY", "TLT", "HYG", "UUP", "GLD"]
    raw = adapter.get_data(
        tickers=tickers,
        start_date="2018-06-01",
        end_date="2024-12-31",
        force_refresh=False,
    )

    rv_gk = get_canonical_rv_gk()  # canonical: 2019-01-02 → 2024-12-31, len 1510
    log_rv_gk = np.log(rv_gk).rename("log_rv_gk")

    prepared = build_vol_iohmm_dataset(
        raw,
        target_ticker="SPY",
        external_tickers=("TLT", "HYG", "UUP", "GLD"),
        rv_window_external=5,
        strictly_external_inputs=True,
        target=log_rv_gk,
    )

    X = prepared.X
    y = prepared.y
    dates = prepared.dates
    T = len(y)

    spy_close = pd.to_numeric(extract_adjusted_close(raw, "SPY"), errors="coerce").astype(float)
    r_raw = np.log(spy_close).diff().reindex(dates).to_numpy()

    print("Feature names:")
    for f in prepared.feature_names:
        print(f"  - {f}")
    print(f"\nTotal IOHMM observations: {T}")
    print(f"IOHMM date range: {dates[0].date()} → {dates[-1].date()}")
    print()

    # Sanity-check: the target column is exactly log(rv_gk) at the prepared dates.
    truth_check = log_rv_gk.reindex(dates).to_numpy()
    assert np.allclose(y, truth_check, equal_nan=False), "injected target mismatch"

    # ── HAR baseline (canonical, identical to Task A) ─────────────────────
    print("Running canonical walk_forward_har_rv on get_canonical_obs()...")
    obs = get_canonical_obs()
    har_pred_series = walk_forward_har_rv(obs, train_window=train_window, refit_every=refit_freq)
    print(f"HAR series: len={len(har_pred_series)}, "
          f"first valid at {har_pred_series.dropna().index[0].date()}")
    print()

    # ── K selection (one-shot on first init_train dates) ──────────────────
    bic_icl: dict = {}
    init_end = min(init_train, T)
    X_tr0 = X[:init_end]
    y_tr0 = y[:init_end]
    print(f"K selection on first {init_end} dates "
          f"({dates[0].date()} → {dates[init_end - 1].date()}):")

    for K in k_values:
        m = GaussianIOHMM(n_states=K, max_iter=100, n_init=10, random_state=seed)
        try:
            m.fit(X_tr0, y_tr0)
            bic_icl[K] = {"bic": m.bic_, "icl": m.icl_, "ll": m.best_loglik_}
            print(f"  K={K}: ll={m.best_loglik_:.2f}  BIC={m.bic_:.2f}  ICL={m.icl_:.2f}")
        except Exception as e:
            warnings.warn(f"K={K} fit failed: {e}")
            bic_icl[K] = {"bic": np.inf, "icl": np.inf, "ll": -np.inf}

    best_K = min(bic_icl, key=lambda k: bic_icl[k]["icl"])
    print(f"Selected K={best_K} by ICL")
    print()

    # ── Walk-forward: rolling 252 / refit every 21 ────────────────────────
    y_true_acc: list = []
    y_hat_iohmm: list = []
    y_hat_har: list = []
    y_hat_garch: list = []
    state_probs_acc: list = []
    dates_acc: list = []

    iohmm = GaussianIOHMM(n_states=best_K, max_iter=100, n_init=10, random_state=seed)

    print(f"Walk-forward: t in [{train_window}, {T}) stepping by {refit_freq}...")
    n_refits = 0
    for t in range(train_window, T, refit_freq):
        X_tr, y_tr = X[t - train_window:t], y[t - train_window:t]
        end = min(t + refit_freq, T)
        X_te, y_te = X[t:end], y[t:end]
        dates_te = dates[t:end]

        if len(X_te) == 0:
            break

        try:
            iohmm.fit(X_tr, y_tr)
        except Exception as e:
            warnings.warn(f"IOHMM fit failed at t={t}: {e}")
            continue

        # GARCH refit on rolling 252-day return window. Returns input is in
        # percent (×100); arch_model's forecast.variance is therefore in
        # percent². Convert to annualized fractional variance (rv_gk units):
        #     daily fractional variance      = pct² / 1e4
        #     annualized fractional variance = daily * 252
        r_tr = r_raw[t - train_window:t]
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
            # Rolling history for the IOHMM forward filter so context length
            # matches the training window length.
            hist_lo = t + i - train_window
            hist_hi = t + i
            X_hist = X[hist_lo:hist_hi]
            y_hist = y[hist_lo:hist_hi]
            try:
                y_h, _, p_next = iohmm.forecast(X_hist, y_hist, xi)
            except Exception as e:
                warnings.warn(f"IOHMM forecast failed at t={t}+{i}: {e}")
                continue

            date_i = dates_te[i]
            y_h_har_val = float(har_pred_series.get(date_i, np.nan))

            if garch_res is None:
                y_h_garch_log = np.nan
            else:
                garch_var_pct2 = float(
                    garch_res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
                )
                garch_var_annualized = garch_var_pct2 / 1e4 * 252.0
                y_h_garch_log = float(np.log(garch_var_annualized))

            y_true_acc.append(float(yi_true))
            y_hat_iohmm.append(float(y_h))
            y_hat_har.append(y_h_har_val)
            y_hat_garch.append(y_h_garch_log)
            state_probs_acc.append(p_next)
            dates_acc.append(date_i)

    print(f"Completed {n_refits} refits, {len(y_true_acc)} predictions.")
    print()

    # ── Aggregate / metrics ───────────────────────────────────────────────
    y_true_log = np.asarray(y_true_acc, dtype=float)
    y_iohmm_log = np.asarray(y_hat_iohmm, dtype=float)
    y_har_rv = np.asarray(y_hat_har, dtype=float)  # already on rv_gk scale
    y_garch_log = np.asarray(y_hat_garch, dtype=float)
    state_probs_arr = np.asarray(state_probs_acc, dtype=float)

    # Convert log predictions to rv_gk scale for cross-track comparison.
    y_true_rv = np.exp(y_true_log)
    y_iohmm_rv = np.exp(y_iohmm_log)
    y_garch_rv = np.exp(y_garch_log)
    # y_har_rv is already on rv_gk scale (walk_forward_har_rv exps internally).

    # Metric mask: drop rows with any NaN among the four series.
    valid = ~(
        np.isnan(y_true_rv)
        | np.isnan(y_iohmm_rv)
        | np.isnan(y_har_rv)
        | np.isnan(y_garch_rv)
    )
    n_dropped = int((~valid).sum())
    print(f"Predictions: {len(valid)} total, {int(valid.sum())} kept, {n_dropped} dropped (NaN).")

    yt_rv = y_true_rv[valid]
    yi_rv = y_iohmm_rv[valid]
    yh_rv = y_har_rv[valid]
    yg_rv = y_garch_rv[valid]
    sp = state_probs_arr[valid]
    dates_v = pd.DatetimeIndex([d for d, v in zip(dates_acc, valid) if v])

    # Volatility-percent scale metrics (canonical comparison frame).
    def vol_pct_metrics(name: str, y_pred_rv: np.ndarray) -> dict:
        y_true_pct = np.sqrt(yt_rv) * 100.0
        y_pred_pct = np.sqrt(y_pred_rv) * 100.0
        rmse = float(np.sqrt(np.mean((y_true_pct - y_pred_pct) ** 2)))
        mae = float(np.mean(np.abs(y_true_pct - y_pred_pct)))
        ql = float(np.mean(yt_rv / y_pred_rv - np.log(yt_rv / y_pred_rv) - 1.0))
        d_actual = np.sign(y_true_pct[1:] - y_true_pct[:-1])
        d_pred = np.sign(y_pred_pct[1:] - y_true_pct[:-1])
        diracc = float(np.mean(d_actual == d_pred))
        return {"model": name, "n": len(yt_rv), "rmse": rmse, "mae": mae, "qlike": ql, "diracc": diracc}

    metrics_df = pd.DataFrame(
        [
            vol_pct_metrics("IOHMM", yi_rv),
            vol_pct_metrics("HAR-RV", yh_rv),
            vol_pct_metrics("GARCH(1,1)", yg_rv),
        ]
    )
    print("\n=== OOS metrics (volatility-percentage scale, sqrt(rv)*100) ===")
    print(metrics_df.to_string(index=False))

    # Diebold-Mariano on log-MSE for back-compat with the prior file.
    e_iohmm = (y_true_log[valid] - y_iohmm_log[valid]) ** 2
    e_har = (y_true_log[valid] - np.log(yh_rv)) ** 2
    e_garch = (y_true_log[valid] - y_garch_log[valid]) ** 2
    print(f"\nDM(IOHMM vs HAR):   {dm_stat(e_iohmm, e_har):.4f}")
    print(f"DM(IOHMM vs GARCH): {dm_stat(e_iohmm, e_garch):.4f}")

    dom_state = np.argmax(sp, axis=1)
    pr = per_regime_metrics(y_true_log[valid], y_iohmm_log[valid], dom_state, best_K)
    print("\n=== Per-regime IOHMM metrics ===")
    print(pr.to_string(index=False))

    bic_icl_df = pd.DataFrame(bic_icl).T.reset_index().rename(columns={"index": "K"})
    print("\n=== K sweep ===")
    print(bic_icl_df.to_string(index=False))

    # ── CSV outputs ───────────────────────────────────────────────────────
    # Canonical cross-track CSV consumed by experiments/unified_table.py.
    # Pull y_true_rv_gk directly from the canonical source (not exp(log(.)))
    # so it is bit-equal to the truth column in Task A's HMM/HAR CSVs.
    yt_rv_canonical = rv_gk.reindex(dates_v).to_numpy(dtype=np.float64)
    canonical_df = pd.DataFrame(
        {
            "date": dates_v,
            "y_true_rv_gk": yt_rv_canonical,
            "y_iohmm_pred": yi_rv,
            "y_har_pred": yh_rv,
            "y_garch_pred": yg_rv,
        }
    )
    os.makedirs(_CANONICAL_OUT_DIR, exist_ok=True)
    canonical_path = os.path.join(_CANONICAL_OUT_DIR, "iohmm_canonical_predictions.csv")
    canonical_df.to_csv(canonical_path, index=False)

    # IOHMM-internal artifacts (per-regime, K-sweep, log-scale results).
    out = pd.DataFrame(
        {
            "date": dates_v,
            "y_true_log": y_true_log[valid],
            "y_iohmm_log": y_iohmm_log[valid],
            "y_har_log": np.log(yh_rv),
            "y_garch_log": y_garch_log[valid],
            "dom_state": dom_state,
        }
    )
    out.to_csv(os.path.join(_EXPERIMENT_DIR, "spy_vol_iohmm_results.csv"), index=False)
    metrics_df.to_csv(os.path.join(_EXPERIMENT_DIR, "spy_vol_iohmm_metrics.csv"), index=False)
    pr.to_csv(os.path.join(_EXPERIMENT_DIR, "spy_vol_iohmm_per_regime.csv"), index=False)
    bic_icl_df.to_csv(os.path.join(_EXPERIMENT_DIR, "spy_vol_iohmm_kselect.csv"), index=False)

    print()
    print(f"wrote {canonical_path}")
    print(f"wrote {os.path.join(_EXPERIMENT_DIR, 'spy_vol_iohmm_results.csv')}")
    print(f"wrote {os.path.join(_EXPERIMENT_DIR, 'spy_vol_iohmm_metrics.csv')}")
    print(f"wrote {os.path.join(_EXPERIMENT_DIR, 'spy_vol_iohmm_per_regime.csv')}")
    print(f"wrote {os.path.join(_EXPERIMENT_DIR, 'spy_vol_iohmm_kselect.csv')}")


if __name__ == "__main__":
    main()
