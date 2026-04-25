from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from arch import arch_model

from IOHMM.regimes.features import build_vol_iohmm_dataset
from IOHMM.regimes.iohmm import GaussianIOHMM
from data_preprocessing.data_adapter import YFinanceAdapter


REFIT_FREQ = 21
MIN_TRAIN = 504
K_VALUES = (2, 3, 4)


def build_har_features(close: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    r = np.log(close).diff()
    rv_d = r ** 2
    rv_w = rv_d.rolling(5).mean()
    rv_m = rv_d.rolling(22).mean()
    feat = pd.DataFrame(
        {
            "log_rv_d_lag1": np.log(rv_d + 1e-8).shift(1),
            "log_rv_w_lag1": np.log(rv_w + 1e-8).shift(1),
            "log_rv_m_lag1": np.log(rv_m + 1e-8).shift(1),
        }
    )
    return feat.reindex(dates).to_numpy()


def qlike(rv_true: np.ndarray, rv_hat: np.ndarray) -> float:
    rv_hat = np.maximum(rv_hat, 1e-12)
    rv_true = np.maximum(rv_true, 1e-12)
    return float(np.mean(rv_true / rv_hat - np.log(rv_true / rv_hat) - 1.0))


def dm_stat(e1: np.ndarray, e2: np.ndarray) -> float:
    d = e1 - e2
    return float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))


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


def main() -> None:
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

    spy_close = pd.to_numeric(raw[("Close", "SPY")], errors="coerce").astype(float)
    har_X = build_har_features(spy_close, dates)
    r_pct = (np.log(spy_close).diff() * 100.0).reindex(dates).to_numpy()

    print("Feature names:")
    for f in prepared.feature_names:
        print(f"  - {f}")

    print(f"\nTotal observations: {T}")
    print(f"Min train: {MIN_TRAIN}, refit freq: {REFIT_FREQ}")

    bic_icl: dict = {}
    init_train_end = MIN_TRAIN
    X_tr0 = X[:init_train_end]
    y_tr0 = y[:init_train_end]

    for K in K_VALUES:
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

    for t in range(MIN_TRAIN, T - 1, REFIT_FREQ):
        X_tr, y_tr = X[:t], y[:t]
        end = min(t + REFIT_FREQ, T - 1)
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
            ridge = None
        else:
            ridge = Ridge(alpha=1.0)
            ridge.fit(har_tr[valid_har], har_y[valid_har])

        r_train = r_pct[:t]
        r_train = r_train[~np.isnan(r_train)]
        try:
            garch = arch_model(r_train, vol="Garch", p=1, q=1, dist="normal", rescale=False)
            res = garch.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=len(X_te), reindex=False)
            garch_var_pct2 = fc.variance.values[-1, :]
        except Exception as e:
            warnings.warn(f"GARCH fit failed at t={t}: {e}")
            garch_var_pct2 = np.full(len(X_te), np.nan)

        for i, (xi, yi_true) in enumerate(zip(X_te, y_te)):
            try:
                y_h, _, p_next = iohmm.forecast(X_tr, y_tr, xi)
            except Exception as e:
                warnings.warn(f"IOHMM forecast failed at t={t}+{i}: {e}")
                continue

            har_xi = har_X[t + i: t + i + 1]
            if ridge is None or np.isnan(har_xi).any():
                y_h_har = np.nan
            else:
                y_h_har = float(ridge.predict(har_xi)[0])

            v_pct2 = garch_var_pct2[i]
            if np.isnan(v_pct2) or v_pct2 <= 0:
                y_h_garch = np.nan
            else:
                y_h_garch = float(np.log(v_pct2 / 1e4 + 1e-8))

            y_true_acc.append(float(yi_true))
            y_hat_iohmm.append(float(y_h))
            y_hat_har.append(y_h_har)
            y_hat_garch.append(y_h_garch)
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
    out.to_csv("spy_vol_iohmm_results.csv", index=False)
    metrics_df.to_csv("spy_vol_iohmm_metrics.csv", index=False)
    pr.to_csv("spy_vol_iohmm_per_regime.csv", index=False)
    bic_icl_df.to_csv("spy_vol_iohmm_kselect.csv", index=False)


if __name__ == "__main__":
    main()
