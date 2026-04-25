import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from adapter import YFinanceAdapter


random_seed = 42


def _to_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    return np.log(df).diff().dropna()


def build_hmm_and_score(X, n_states=4, covariance_type="full", n_iter=100, tol=1e-4):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        min_covar=1e-3,
        init_params="stmc",
        transmat_prior=1.0,
        startprob_prior=1.0,
        random_state=random_seed,
    )
    model.fit(X)
    log_likelihood = model.score(X)
    return model, log_likelihood


def likelihood_distance(L1, L2, normalize=True, T=None):
    if normalize and T is not None:
        return abs(L1 / T - L2 / T)
    return abs(L1 - L2)


def num_hmm_params(n_states, n_features):
    trans = n_states * (n_states - 1)
    start = n_states - 1
    means = n_states * n_features
    cov = n_states * (n_features * (n_features + 1) / 2)
    return int(trans + start + means + cov)


def compute_information_criteria(logL, T, k):
    AIC = -2 * logL + 2 * k
    BIC = -2 * logL + k * np.log(T)
    HQC = -2 * logL + 2 * k * np.log(np.log(T))
    CAIC = -2 * logL + k * (np.log(T) + 1)
    return {"AIC": AIC, "BIC": BIC, "HQC": HQC, "CAIC": CAIC}


def rolling_ic_train_window(X, window=120, n_states=4):
    """In-sample information criteria computed on a rolling training window.

    This is NOT a backtest: each window is fit and scored on the same data.
    """
    results = []
    for t in range(window, len(X) - window + 1):
        X_win = X[t - window: t]
        _, logL = build_hmm_and_score(X_win, n_states=n_states)
        k = num_hmm_params(n_states, X.shape[1])
        ic = compute_information_criteria(logL, window, k)
        ic["t"] = t
        results.append(ic)
    return pd.DataFrame(results).set_index("t")


def get_data(start_date, test_date, end_date, freq):
    adapter = YFinanceAdapter()
    data = adapter.get_data(tickers="^GSPC", start_date=start_date, end_date=end_date)
    # auto_adjust=True ⇒ Close is the adjusted price
    df = data[["Open", "High", "Low", "Close"]].dropna()
    df = df.resample(freq).last()
    train = df.loc[start_date:test_date].copy()
    test = df.loc[test_date:end_date].copy()
    return train, test


def hmm_pipeline(n_states, window, freq,
                 start_date="2019-01-01", test_date="2024-01-01", end_date="2026-01-01"):
    train, _ = get_data(start_date, test_date, end_date, freq)
    train_r = _to_log_returns(train)
    return rolling_ic_train_window(train_r.values, window, n_states)


def backtest(n_states, freq,
             start_date="2019-01-01", test_date="2024-01-01", end_date="2026-01-01"):
    train, test = get_data(start_date, test_date, end_date, freq)
    train_r = _to_log_returns(train)
    test_r = _to_log_returns(test)

    if len(test_r) < 2:
        raise ValueError("Test set too short for one-step-ahead backtest.")

    model, _ = build_hmm_and_score(train_r.values, n_states=n_states)
    means = model.means_
    transmat = model.transmat_

    n_steps = len(test_r) - 1
    n_features = train_r.shape[1]
    preds = np.zeros((n_steps, n_features))
    actuals = test_r.values[1:]

    for i in range(n_steps):
        history = np.vstack([train_r.values, test_r.values[: i + 1]])
        state_now = int(model.predict(history)[-1])
        next_state_dist = transmat[state_now]
        preds[i] = next_state_dist @ means

    mse_per_feature = np.mean((actuals - preds) ** 2, axis=0)
    return {
        "n_steps": int(n_steps),
        "mse_per_feature": mse_per_feature.tolist(),
        "mse_total": float(np.mean(mse_per_feature)),
        "feature_names": list(train_r.columns),
    }


if __name__ == "__main__":
    ic_df = hmm_pipeline(n_states=4, window=120, freq="W")
    print(ic_df)
    bt = backtest(n_states=4, freq="W")
    print(bt)
