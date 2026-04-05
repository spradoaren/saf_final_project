import numpy as np
import pandas as pd

import os
os.environ["OMP_NUM_THREADS"] = "1"

from hmmlearn.hmm import GaussianHMM
import yfinance as yf
from adapter import YFinanceAdapter

random_seed = 42
def build_hmm_and_score(X, n_states=4, covariance_type="full", n_iter=100, tol=1e-4):
    """
    Input:
        X: np.ndarray, shape (T, n_features)
        n_states: regime number
    Output:
        model: trained HMM
        log_likelihood: log P(O | λ)
    """

    model = GaussianHMM(n_states,covariance_type,n_iter,
        init_params='mc',random_state=random_seed ) # only initialize means and covariances

    # avoid all 0
    model.startprob_ = np.full(n_states, 1.0 / n_states)
    model.transmat_ = np.full((n_states, n_states), 1.0 / n_states)
    # Baum-Welch (EM) training
    model.fit(X)

    # Forward algorithm -> likelihood
    log_likelihood = model.score(X)

    return model, log_likelihood

def likelihood_distance(L1, L2, normalize=True, T=None):
    #|L1 - L2|
    if normalize and T is not None:
        return abs(L1 / T - L2 / T)
    return abs(L1 - L2)

def num_hmm_params(n_states, n_features):
    # transition matrix
    trans = n_states * (n_states - 1)

    # initial probs
    start = n_states - 1
    # means
    means = n_states * n_features
    # covariance (full)
    cov = n_states * (n_features * (n_features + 1) / 2)

    return int(trans + start + means + cov)


def compute_information_criteria(logL, T, k):

    #T: sample length
    #k: parameter number
    AIC = -2 * logL + 2 * k
    BIC = -2 * logL + k * np.log(T)
    HQC = -2 * logL + 2 * k * np.log(np.log(T))
    CAIC = -2 * logL + k * (np.log(T) + 1)

    return {"AIC": AIC,"BIC": BIC,"HQC": HQC,"CAIC": CAIC}

def rolling_ic(X,window=120, n_states=4):
    #rolling AIC/BIC/HQC/CAIC
    results = []

    for t in range(window, len(X)-window+1):
        X_win = X[t-window:t]

        model, logL = build_hmm_and_score(X_win, n_states=n_states)
        k = num_hmm_params(n_states, X.shape[1])
        ic = compute_information_criteria(logL, window, k)
        ic["t"] = t
        results.append(ic)

    return pd.DataFrame(results).set_index("t")

def get_data(start_date,test_date,end_date,freq):
    adapter = YFinanceAdapter()
    data= adapter.get_data(tickers="^GSPC", start_date=start_date,end_date=end_date)
    df=data[['Open','High','Low','Adj Close']].dropna()

    #get weekly data
    df=df.resample(freq).last()
    train = df.loc[start_date:test_date].copy()
    test  = df.loc[test_date:end_date].copy()

    return train,test

def hmm_pipeline(n_states,window,freq,
                 start_date="2019-01-01",test_date="2024-01-01",end_date="2026-01-01"):

    train,test=get_data(start_date,test_date,end_date,freq)
    ic_df=rolling_ic(train,window, n_states)
    return ic_df


#backtest
def block_forecast(X):
    return True
def backtest(n_states,window,freq,
             start_date="2019-01-01",test_date="2024-01-01",end_date="2026-01-01"):
    train,test=get_data(start_date,test_date,end_date,freq)
    model,log_likelihood=build_hmm_and_score(train,n_states)
    return True
if __name__ == "__main__":
    ic_df=hmm_pipeline(n_states=4,window=120,freq="W")
    print(ic_df)
