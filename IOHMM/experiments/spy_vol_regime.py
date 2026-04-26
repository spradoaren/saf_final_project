from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from arch import arch_model

from IOHMM.regimes.features import build_har_features, build_vol_iohmm_dataset
from IOHMM.regimes.iohmm import GaussianIOHMM
from data_preprocessing.data_adapter import YFinanceAdapter
from data_preprocessing.price_utils import extract_adjusted_close
from utils.metrics import dm_stat, qlike


# Defaults preserved from the original implementation. They are exposed
# via ``main`` arguments so callers can override at the call site
# without editing this file.
REFIT_FREQ = 21
MIN_TRAIN = 504
K_VALUES = (2, 3, 4)


# Absolute path to this experiment directory, used to anchor CSV outputs
# regardless of the current working directory at run time.
_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    min_train: int = MIN_TRAIN,
    refit_freq: int = REFIT_FREQ,
    k_values=K_VALUES,
) -> None:
    adapter = YFinanceAdapter()
    tickers = ["SPY", "TLT", "HYG", "UUP", "GLD"]

    raw = adapter.get_data(
        tickers=tickers,
        start_date="2017-06-01",
        end_date="2025-01-31",
        force_refresh=False,
    )

    prepared = build_vol_iohmm_dataset(
        raw,
        target_ticker="SPY",
        external_tickers=("TLT", "HYG", "UUP", "GLD"),
        rv_window_external=5,
        strictly_external_inputs=True,
    )

    X = prepared.X
    y = prepared.y
    dates = prepared.dates
    T = len(y)

    spy_close = pd.to_numeric(extract_adjusted_close(raw, "SPY"), errors="coerce").astype(float)
    har_X = build_har_features(spy_close, dates)
    r_raw = np.log(spy_close).diff().reindex(dates).to_numpy()

    print("Feature names:")
    for f in prepared.feature_names:
        print(f"  - {f}")

    print(f"\nTotal observations: {T}")
    print(f"Min train: {min_train}, refit freq: {refit_freq}")

    bic_icl: dict = {}
    init_train_end = min_train
    X_tr0 = X[:init_train_end]
    y_tr0 = y[:init_train_end]

    for K in k_values:
        m = GaussianIOHMM(n_states=K, max_iter=100, n_init=10, random_state=42)
        try:
            m.fit(X_tr0, y_tr0)
            bic_icl[K] = {"bic": m.bic_, "icl": m.icl_, "ll": m.best_loglik_}
            print(f"K={K}: ll={m.best_loglik_:.2f}, BIC={m.bic_:.2f}, ICL={m.icl_:.2f}")
        except Exception as e:
            warnings.warn(f"K={K} fit failed: {e}")
            bic_icl[K] = {"bic": np.inf, "icl": np.inf, "ll": -np.inf}

    best_K = min(bic_icl, key=lambda k: bic_icl[k]["icl"])
    print(f"\nSelected K={best_K} by ICL")

    y_true_acc: list = []
    y_hat_iohmm: list = []
    y_hat_har: list = []
    y_hat_garch: list = []
    state_probs_acc: list = []
    dates_acc: list = []

    iohmm = GaussianIOHMM(n_states=best_K, max_iter=100, n_init=10, random_state=42)

    for t in range(min_train, T - 1, refit_freq):
        X_tr, y_tr = X[:t], y[:t]
        end = min(t + refit_freq, T - 1)
        X_te, y_te = X[t:end], y[t:end]
        dates_te = dates[t:end]

        if len(X_te) == 0:
            break

        try:
            iohmm.fit(X_tr, y_tr)
        except Exception as e:
            warnings.warn(f"IOHMM fit failed at t={t}: {e}")
            continue

        har_tr = har_X[:t]
        har_y = y[:t]
        valid_har = ~np.isnan(har_tr).any(axis=1)
        if valid_har.sum() < 50:
            warnings.warn(f"Too few valid HAR rows at t={t}; skipping HAR.")
            har = None
        else:
            n_har = int(valid_har.sum())
            har_cv = TimeSeriesSplit(n_splits=min(5, max(2, n_har - 1)))
            har = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=har_cv)
            har.fit(har_tr[valid_har], har_y[valid_har])

        for i, (xi, yi_true) in enumerate(zip(X_te, y_te)):
            X_hist = X[: t + i]
            y_hist = y[: t + i]
            try:
                y_h, _, p_next = iohmm.forecast(X_hist, y_hist, xi)
            except Exception as e:
                warnings.warn(f"IOHMM forecast failed at t={t}+{i}: {e}")
                continue

            har_xi = har_X[t + i: t + i + 1]
            if har is None or np.isnan(har_xi).any():
                y_h_har = np.nan
            else:
                y_h_har = float(har.predict(har_xi)[0])

            r_tr_i = r_raw[:t + i]
            r_tr_i = r_tr_i[~np.isnan(r_tr_i)]
            try:
                garch_res_i = arch_model(
                    r_tr_i * 100, vol="Garch", p=1, q=1, dist="normal"
                ).fit(disp="off", show_warning=False)
                garch_var = garch_res_i.forecast(horizon=1, reindex=False).variance.values[-1, 0]
                y_h_garch = np.log(garch_var / 1e4 + 1e-8)
            except Exception as e:
                warnings.warn(f"GARCH fit failed at t={t}+{i}: {e}")
                y_h_garch = np.nan

            y_true_acc.append(float(yi_true))
            y_hat_iohmm.append(float(y_h))
            y_hat_har.append(y_h_har)
            y_hat_garch.append(float(y_h_garch) if y_h_garch == y_h_garch else np.nan)
            state_probs_acc.append(p_next)
            dates_acc.append(dates_te[i])

    y_true_arr = np.asarray(y_true_acc, dtype=float)
    y_iohmm_arr = np.asarray(y_hat_iohmm, dtype=float)
    y_har_arr = np.asarray(y_hat_har, dtype=float)
    y_garch_arr = np.asarray(y_hat_garch, dtype=float)
    state_probs_arr = np.asarray(state_probs_acc, dtype=float)

    valid = ~(
        np.isnan(y_true_arr)
        | np.isnan(y_iohmm_arr)
        | np.isnan(y_har_arr)
        | np.isnan(y_garch_arr)
    )

    yt = y_true_arr[valid]
    yi = y_iohmm_arr[valid]
    yh = y_har_arr[valid]
    yg = y_garch_arr[valid]
    sp = state_probs_arr[valid]
    dates_v = pd.DatetimeIndex([d for d, v in zip(dates_acc, valid) if v])

    def metrics_row(name: str, yhat: np.ndarray) -> dict:
        return {
            "model": name,
            "mse": mean_squared_error(yt, yhat),
            "mae": mean_absolute_error(yt, yhat),
            "qlike": qlike(np.exp(yt), np.exp(yhat)),
        }

    metrics_df = pd.DataFrame(
        [
            metrics_row("IOHMM", yi),
            metrics_row("HAR-RV", yh),
            metrics_row("GARCH(1,1)", yg),
        ]
    )

    print("\n=== OOS forecast metrics ===")
    print(metrics_df.to_string(index=False))

    e_iohmm = (yt - yi) ** 2
    e_har = (yt - yh) ** 2
    e_garch = (yt - yg) ** 2
    print(f"\nDM(IOHMM vs HAR):   {dm_stat(e_iohmm, e_har):.4f}")
    print(f"DM(IOHMM vs GARCH): {dm_stat(e_iohmm, e_garch):.4f}")

    dom_state = np.argmax(sp, axis=1)
    pr = per_regime_metrics(yt, yi, dom_state, best_K)
    print("\n=== Per-regime IOHMM metrics ===")
    print(pr.to_string(index=False))

    bic_icl_df = pd.DataFrame(bic_icl).T.reset_index().rename(columns={"index": "K"})
    print("\n=== K sweep ===")
    print(bic_icl_df.to_string(index=False))

    out = pd.DataFrame(
        {
            "date": dates_v,
            "y_true": yt,
            "y_iohmm": yi,
            "y_har": yh,
            "y_garch": yg,
            "dom_state": dom_state,
        }
    )
    out.to_csv(os.path.join(_EXPERIMENT_DIR, "spy_vol_iohmm_results.csv"), index=False)
    metrics_df.to_csv(os.path.join(_EXPERIMENT_DIR, "spy_vol_iohmm_metrics.csv"), index=False)
    pr.to_csv(os.path.join(_EXPERIMENT_DIR, "spy_vol_iohmm_per_regime.csv"), index=False)
    bic_icl_df.to_csv(os.path.join(_EXPERIMENT_DIR, "spy_vol_iohmm_kselect.csv"), index=False)


if __name__ == "__main__":
    main()
