"""Regime diagnostics for paper Section 5.3 (R1).

For each of the three latent-state models (HMM K=3, IOHMM K=2, MS-AR K=2),
fit ONCE on the full canonical sample (2019-01-02 to 2024-12-31) and
extract:

    1. Smoothed regime posteriors  gamma_t(k)   shape (T, K)
    2. Estimated transition matrix              shape (K, K)

The state index is then permuted so that state 0 corresponds to the
lowest within-state weighted mean of log(rv_gk_t) (using gamma_t(k) as
weights) and state K-1 corresponds to the highest.

NOTE: This is a retrospective full-sample fit for regime IDENTIFICATION
ONLY. It does NOT replace the walk-forward forecast evaluation. The
smoothed posteriors here come from the full-sample forward-backward
recursion and therefore differ from the filtered (forward-only)
posteriors used inside the walk-forward forecasting scripts.

Configurations match the canonical walk-forward scripts:

    HMM    : K=3, HAR triple features (GKVolFeatures), GaussianHMM with
             full covariance, n_iter=100, random_state=42 (mirrors
             SecondOrderHMM as used in HMM.features.RVForecastWF).

    IOHMM  : K=2 (the ICL choice on h=1; ICL is NOT re-run here).
             12-feature macro vector from
             IOHMM.regimes.features.build_vol_iohmm_dataset with
             external_tickers=("TLT","HYG","UUP","GLD"),
             rv_window_external=5, strictly_external_inputs=True;
             emission_ridge=1e-4 (quantile-based emission init);
             n_init=10, max_iter=100, random_state=42.
             The class internally permutes states by ascending sigma2 at
             fit time; this script then RE-permutes by within-state
             weighted log(rv_gk) means as required by the paper.

    MS-AR  : k_regimes=2, order=1, switching_ar=False,
             switching_variance=True, trend="c", target=log(rv_gk).
             statsmodels MarkovAutoregression MLE is deterministic; the
             seed is recorded for documentation parity.

Outputs (per model in {hmm, iohmm, msar}, in the volatility-sorted
state order, state 0 = lowest log(rv_gk) mean):

    experiments/cache/regime_diagnostics_{model}.parquet
        DataFrame indexed by canonical rv_gk dates (full T = 1510),
        columns ['gamma_0', ..., 'gamma_{K-1}']. Rows that fall in the
        per-model warm-up (HMM: monthly rolling, MS-AR: AR-lag) are
        preserved as NaN.

    experiments/cache/regime_diagnostics_{model}_transmat.parquet
        DataFrame, shape (K, K). Index = from-state, columns = to-state.

Pure consumer: this script does NOT modify HMM/features.py,
IOHMM/regimes/iohmm.py, or Markov_Switching_AR/model_utils.py.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

from data_preprocessing.data_adapter import YFinanceAdapter
from experiments._shared import get_canonical_rv_gk
from HMM.features import GKVolFeatures
from IOHMM.regimes.features import build_vol_iohmm_dataset
from IOHMM.regimes.iohmm import GaussianIOHMM
from Markov_Switching_AR.model_utils import fit_msar_model


CACHE_DIR = Path(__file__).resolve().parent / "cache"
SEED = 42
EPS_STD = 1e-12


# ──────────────────────────────────────────────────────────────────────
# Permutation utilities
# ──────────────────────────────────────────────────────────────────────

def vol_weighted_means(
    gamma_full: np.ndarray, log_rv: np.ndarray
) -> np.ndarray:
    """Within-state weighted mean of log_rv using gamma weights.

    Rows where log_rv or gamma is NaN are excluded.
    Returns shape (K,) in the OLD state ordering of `gamma_full`.
    """
    K = gamma_full.shape[1]
    means = np.full(K, np.nan, dtype=np.float64)
    valid = (~np.isnan(log_rv)) & np.all(~np.isnan(gamma_full), axis=1)
    g = gamma_full[valid]
    lr = log_rv[valid]
    for k in range(K):
        w_sum = float(g[:, k].sum())
        if w_sum > 0.0:
            means[k] = float((g[:, k] * lr).sum() / w_sum)
    return means


def apply_permutation(
    gamma_full: np.ndarray,
    transmat: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply state permutation sigma (new -> old) to posteriors and transmat.

    gamma_new[:, i]   = gamma_old[:, sigma[i]]
    trans_new[i, j]   = trans_old[sigma[i], sigma[j]]
    """
    gamma_perm = gamma_full[:, sigma]
    trans_perm = transmat[np.ix_(sigma, sigma)]
    return gamma_perm, trans_perm


def reindex_to_canonical(
    gamma_local: np.ndarray,
    local_index: pd.DatetimeIndex,
    canonical_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Embed gamma_local (length T_local) into a (T_canonical, K) array,
    placing rows at the matching canonical positions and leaving the rest
    as NaN. Asserts local_index ⊆ canonical_index.
    """
    K = gamma_local.shape[1]
    out = np.full((len(canonical_index), K), np.nan, dtype=np.float64)
    pos = canonical_index.get_indexer(pd.DatetimeIndex(local_index))
    if (pos < 0).any():
        missing = pd.DatetimeIndex(local_index)[pos < 0]
        raise ValueError(
            f"local_index contains {len(missing)} dates not in canonical index "
            f"(first missing: {missing[0]})"
        )
    out[pos] = gamma_local.astype(np.float64, copy=False)
    return out


# ──────────────────────────────────────────────────────────────────────
# Per-model fits
# ──────────────────────────────────────────────────────────────────────

def fit_hmm_full_sample(
    rv_gk: pd.Series,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Full-sample HMM fit. Returns (obs_index, gamma_smoothed, transmat).

    Mirrors HMM.features.SecondOrderHMM (StandardScaler + GaussianHMM,
    full covariance, n_iter=100, random_state=42).
    """
    obs_df = GKVolFeatures().fit_transform(rv_gk)
    X_raw = obs_df.to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw).astype(np.float64)

    hmm = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=100,
        random_state=SEED,
    )
    hmm.fit(X_sc)

    # hmmlearn's predict_proba returns smoothed posteriors (full-sample)
    gamma = hmm.predict_proba(X_sc).astype(np.float64)
    transmat = np.asarray(hmm.transmat_, dtype=np.float64)

    return pd.DatetimeIndex(obs_df.index), gamma, transmat


def fit_iohmm_full_sample(
    rv_gk: pd.Series,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, int, int]:
    """Full-sample IOHMM fit. K=2.

    Returns
    -------
    dates : DatetimeIndex of length T (matches canonical rv_gk dates).
    gamma : (T, 2) smoothed posteriors in the model's internal
        sigma2-sorted order (re-permuted later by this script).
    transmat_avg : (2, 2) time-averaged input-dependent transition
        matrix, P_bar[i, j] = (1/T_valid) * sum_t softmax(W_i @ x_tilde_t)[j].
    n_skipped : count of t-rows skipped in the time-average due to NaN
        in the standardized feature row x_tilde_t.
    n_features : number of macro features used (sanity check).
    """
    log_rv = np.log(rv_gk).rename("log_rv_gk").astype(np.float64)

    adapter = YFinanceAdapter()
    raw = adapter.get_data(
        tickers=["SPY", "TLT", "HYG", "UUP", "GLD"],
        start_date="2018-06-01",
        end_date="2024-12-31",
        force_refresh=False,
    )

    prepared = build_vol_iohmm_dataset(
        raw,
        target_ticker="SPY",
        external_tickers=("TLT", "HYG", "UUP", "GLD"),
        rv_window_external=5,
        strictly_external_inputs=True,
        target=log_rv,
    )
    X = prepared.X.astype(np.float64)
    y = prepared.y.astype(np.float64)
    dates = pd.DatetimeIndex(prepared.dates)

    iohmm = GaussianIOHMM(
        n_states=2,
        emission_ridge=1e-4,
        max_iter=100,
        n_init=10,
        random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iohmm.fit(X, y)

    # smoothed posteriors via the class's forward-backward, evaluated on
    # the FITTED (already-sigma2-permuted) emission and transition models.
    gamma = iohmm.predict_state_proba(X, y, smoothed=True).astype(np.float64)

    # Time-averaged input-dependent transition matrix.
    # Standardize X with the fitted standardization (same convention as
    # IOHMM._standardize_apply: zero-std features get std=1).
    std = np.where(iohmm.feature_std_ < EPS_STD, 1.0, iohmm.feature_std_)
    Xs = (X - iohmm.feature_mean_) / std

    valid_mask = ~np.any(np.isnan(Xs), axis=1)
    n_skipped = int((~valid_mask).sum())
    Xs_v = Xs[valid_mask]
    X1 = np.column_stack([np.ones(len(Xs_v), dtype=np.float64), Xs_v])

    K = iohmm.n_states
    P_bar = np.zeros((K, K), dtype=np.float64)
    W = iohmm.transition_model.W.astype(np.float64)  # (K, K, n_features+1)
    for i in range(K):
        # logits_t for source i: shape (T_valid, K)
        logits = X1 @ W[i].T
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=1, keepdims=True)
        P_bar[i] = probs.mean(axis=0)

    return dates, gamma, P_bar, n_skipped, X.shape[1]


def fit_msar_full_sample(
    rv_gk: pd.Series,
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Full-sample MS-AR fit. Target = log(rv_gk).

    Returns
    -------
    smp_index : DatetimeIndex of smoothed_marginal_probabilities (length
        T - order, omitting the AR-lag warm-up rows).
    gamma : (T - order, 2) smoothed posteriors.
    transmat : (2, 2) time-homogeneous transition matrix in the model's
        native ordering, with M[i, j] = P(next=j | now=i).
    """
    log_rv = np.log(rv_gk).rename("log_rv_gk").astype(np.float64)

    res = fit_msar_model(
        returns=log_rv,
        k_regimes=2,
        order=1,
        switching_ar=False,
        switching_variance=True,
        trend="c",
    )

    smp = res.smoothed_marginal_probabilities
    if isinstance(smp, pd.DataFrame):
        smp_np = smp.to_numpy(dtype=np.float64)
        smp_index = pd.DatetimeIndex(smp.index)
    else:
        smp_np = np.asarray(smp, dtype=np.float64)
        smp_index = log_rv.index[-smp_np.shape[0]:]

    # regime_transition[to, from, t] for time-homogeneous chains has t-axis 1.
    rt = np.asarray(res.regime_transition, dtype=np.float64)
    K = 2
    transmat = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            transmat[i, j] = float(rt[j, i, 0])

    return smp_index, smp_np, transmat


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def _save_one(
    name: str,
    K: int,
    canonical_index: pd.DatetimeIndex,
    gamma_canonical: np.ndarray,
    transmat: np.ndarray,
) -> Tuple[Path, Path]:
    df_g = pd.DataFrame(
        gamma_canonical.astype(np.float64),
        index=canonical_index,
        columns=[f"gamma_{k}" for k in range(K)],
    )
    df_g.index.name = "date"
    for c in df_g.columns:
        df_g[c] = df_g[c].astype(np.float64)

    df_t = pd.DataFrame(
        transmat.astype(np.float64),
        index=[f"from_{k}" for k in range(K)],
        columns=[f"to_{k}" for k in range(K)],
    )
    for c in df_t.columns:
        df_t[c] = df_t[c].astype(np.float64)

    print(f"  (f) gamma dtypes:    {sorted(set(str(t) for t in df_g.dtypes))}")
    print(f"  (f) transmat dtypes: {sorted(set(str(t) for t in df_t.dtypes))}")

    g_path = CACHE_DIR / f"regime_diagnostics_{name}.parquet"
    t_path = CACHE_DIR / f"regime_diagnostics_{name}_transmat.parquet"
    df_g.to_parquet(g_path)
    df_t.to_parquet(t_path)
    print(f"  → wrote {g_path}")
    print(f"  → wrote {t_path}")
    return g_path, t_path


def _validate(
    name: str,
    canonical_index: pd.DatetimeIndex,
    rv_gk_canonical: pd.Series,
    log_rv_canonical: np.ndarray,
    gamma_canonical: np.ndarray,
    transmat: np.ndarray,
    means_old: np.ndarray,
    sigma: np.ndarray,
) -> None:
    K = gamma_canonical.shape[1]

    # (b) Posterior row sums = 1 within tolerance (skip NaN rows)
    valid_rows = ~np.any(np.isnan(gamma_canonical), axis=1)
    rs = gamma_canonical[valid_rows].sum(axis=1)
    max_dev_rs = float(np.max(np.abs(rs - 1.0))) if len(rs) > 0 else 0.0
    assert max_dev_rs < 1e-10, f"{name}: posterior row sums deviate by {max_dev_rs}"
    print(f"  (b) posterior row-sum max |dev|: {max_dev_rs:.3e}  (tol < 1e-10)")

    # (c) Transition matrix row sums = 1
    trs = transmat.sum(axis=1)
    max_dev_trs = float(np.max(np.abs(trs - 1.0)))
    assert max_dev_trs < 1e-10, f"{name}: transmat row sums deviate by {max_dev_trs}"
    print(f"  (c) transmat row sums:           {trs.tolist()}  (max |dev| = {max_dev_trs:.3e})")

    # (d) Within-state weighted means monotonic increasing post-permutation
    means_new = means_old[sigma]
    print(f"  (d) within-state means (old order):    {means_old.tolist()}")
    print(f"  (d) within-state means (sorted, new):  {means_new.tolist()}")
    assert np.all(np.diff(means_new) > 0), (
        f"{name}: within-state means not strictly increasing after permutation"
    )

    # (e) NaN handling
    rv_nan_idx = set(np.where(np.isnan(rv_gk_canonical.to_numpy(dtype=np.float64)))[0].tolist())
    gamma_nan_idx = set(np.where(np.isnan(gamma_canonical[:, 0]))[0].tolist())
    extra_nan = sorted(gamma_nan_idx - rv_nan_idx)
    print(
        f"  (e) NaN rows in posteriors: {len(gamma_nan_idx)} total"
        f" | rv_gk NaN rows: {len(rv_nan_idx)}"
        f" | posterior-only NaN (warm-up): {len(extra_nan)}"
    )

    # Internal consistency: sigma should be a valid permutation of [0..K-1]
    assert sorted(sigma.tolist()) == list(range(K)), f"{name}: sigma is not a permutation"


def main() -> None:
    print(f"R1: regime diagnostics, seed = {SEED}")
    print(f"sample period: 2019-01-02 → 2024-12-31 (per get_canonical_rv_gk)")
    print()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rv_gk_canonical = get_canonical_rv_gk().astype(np.float64)
    canonical_index = pd.DatetimeIndex(rv_gk_canonical.index)
    log_rv_canonical = np.log(rv_gk_canonical).to_numpy(dtype=np.float64)
    print(
        f"canonical rv_gk: T={len(canonical_index)}, "
        f"{canonical_index[0].date()} → {canonical_index[-1].date()}, "
        f"NaN={int(rv_gk_canonical.isna().sum())} (expected 0)"
    )
    print()

    # ── HMM ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("HMM  (K=3, HAR triple, GaussianHMM full covariance)")
    print("=" * 70)
    hmm_idx, gamma_hmm, transmat_hmm = fit_hmm_full_sample(rv_gk_canonical)

    # (a) bit-equality: HMM consumes get_canonical_rv_gk() directly inside
    # GKVolFeatures().fit_transform(rv_gk). The internal rv_gk reference
    # IS the canonical Series object.
    diff_hmm = float(
        np.max(
            np.abs(
                rv_gk_canonical.to_numpy(dtype=np.float64)
                - get_canonical_rv_gk().to_numpy(dtype=np.float64)
            )
        )
    )
    print(f"(a) HMM   internal rv_gk vs get_canonical_rv_gk(): max abs diff = {diff_hmm}")

    gamma_hmm_full = reindex_to_canonical(gamma_hmm, hmm_idx, canonical_index)
    means_hmm = vol_weighted_means(gamma_hmm_full, log_rv_canonical)
    sigma_hmm = np.argsort(means_hmm)
    print(f"  permutation sigma (new -> old): {sigma_hmm.tolist()}")
    gamma_hmm_p, transmat_hmm_p = apply_permutation(
        gamma_hmm_full, transmat_hmm, sigma_hmm
    )

    _validate(
        "HMM",
        canonical_index,
        rv_gk_canonical,
        log_rv_canonical,
        gamma_hmm_p,
        transmat_hmm_p,
        means_hmm,
        sigma_hmm,
    )
    _save_one("hmm", 3, canonical_index, gamma_hmm_p, transmat_hmm_p)
    print()

    # ── IOHMM ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("IOHMM (K=2, 12-feature macro vector, ridge=1e-4, n_init=10)")
    print("=" * 70)
    iohmm_dates, gamma_iohmm, transmat_iohmm_avg, n_skipped, n_feats = (
        fit_iohmm_full_sample(rv_gk_canonical)
    )
    print(f"  IOHMM macro feature count: {n_feats} (expected 12 = 4 tickers × 3 features)")
    print(f"  IOHMM time-average: skipped {n_skipped} rows with NaN in standardized x_t")

    diff_iohmm = float(
        np.max(
            np.abs(
                rv_gk_canonical.to_numpy(dtype=np.float64)
                - get_canonical_rv_gk().to_numpy(dtype=np.float64)
            )
        )
    )
    print(f"(a) IOHMM internal rv_gk vs get_canonical_rv_gk(): max abs diff = {diff_iohmm}")

    gamma_iohmm_full = reindex_to_canonical(gamma_iohmm, iohmm_dates, canonical_index)
    means_iohmm = vol_weighted_means(gamma_iohmm_full, log_rv_canonical)
    sigma_iohmm = np.argsort(means_iohmm)
    print(f"  permutation sigma (new -> old): {sigma_iohmm.tolist()}")
    gamma_iohmm_p, transmat_iohmm_p = apply_permutation(
        gamma_iohmm_full, transmat_iohmm_avg, sigma_iohmm
    )

    _validate(
        "IOHMM",
        canonical_index,
        rv_gk_canonical,
        log_rv_canonical,
        gamma_iohmm_p,
        transmat_iohmm_p,
        means_iohmm,
        sigma_iohmm,
    )
    _save_one("iohmm", 2, canonical_index, gamma_iohmm_p, transmat_iohmm_p)
    print()

    # ── MS-AR ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("MS-AR (K=2, order=1, sw_var=True, trend='c', target=log(rv_gk))")
    print("=" * 70)
    msar_idx, gamma_msar, transmat_msar = fit_msar_full_sample(rv_gk_canonical)

    diff_msar = float(
        np.max(
            np.abs(
                rv_gk_canonical.to_numpy(dtype=np.float64)
                - get_canonical_rv_gk().to_numpy(dtype=np.float64)
            )
        )
    )
    print(f"(a) MS-AR internal rv_gk vs get_canonical_rv_gk(): max abs diff = {diff_msar}")

    gamma_msar_full = reindex_to_canonical(gamma_msar, msar_idx, canonical_index)
    means_msar = vol_weighted_means(gamma_msar_full, log_rv_canonical)
    sigma_msar = np.argsort(means_msar)
    print(f"  permutation sigma (new -> old): {sigma_msar.tolist()}")
    gamma_msar_p, transmat_msar_p = apply_permutation(
        gamma_msar_full, transmat_msar, sigma_msar
    )

    _validate(
        "MS-AR",
        canonical_index,
        rv_gk_canonical,
        log_rv_canonical,
        gamma_msar_p,
        transmat_msar_p,
        means_msar,
        sigma_msar,
    )
    _save_one("msar", 2, canonical_index, gamma_msar_p, transmat_msar_p)
    print()

    print("R1 complete: posteriors and transition matrices saved.")


if __name__ == "__main__":
    main()
