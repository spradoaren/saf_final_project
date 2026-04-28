"""Regime quality metrics for paper Section 5.3 (R2).

Consumes the parquet outputs of R1 (experiments/regime_diagnostics.py)
and the canonical realized-variance series, and emits one row per
(model, regime) into results/regime_metrics.csv.

Per-regime fields
    duration_days   = 1 / (1 - P[k, k])      [trading days]
    mu_log_rv       = sum_t gamma_t(k) * log(rv_gk_t) / sum_t gamma_t(k)
    sigma_log_rv    = sqrt( sum_t gamma_t(k) * (log(rv_gk_t)-mu_k)^2
                            / sum_t gamma_t(k) )
    mu_log_rv and sigma_log_rv are restricted to the TEST WINDOW
    (2020-01-31 → 2024-12-31), which matches the forecasting OOS window
    used in paper Sections 5.1 and 5.2.

Model-level fields (repeated across that model's regime rows)
    n_test_days                = #{t in test window with valid gamma & rv_gk}
    stress_auc                 = roc_auc_score(is_stress_t, p_high_t)
                                 over the test window, where
                                 is_stress_t = (sqrt(rv_gk_t)*100 >=
                                 95th-pctile-of-test-window-vol_pct)
                                 and p_high_t = gamma_t(K-1).
    stress_threshold_vol_pct   = the 95th-percentile threshold itself
    n_stress_days              = sum(is_stress_t)
    covid_frac_high_regime     = mean( gamma_t(K-1) > 0.5 ) over the
                                 COVID window 2020-02-24 → 2020-04-30
                                 (full sample).
    n_covid_days               = #{t in COVID window with valid gamma}

This script is a pure consumer of R1's outputs. It does NOT modify any
model code or R1 parquet files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments._shared import get_canonical_rv_gk


CACHE_DIR = Path(__file__).resolve().parent / "cache"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODELS = ("hmm", "iohmm", "msar")
# Test window matches the forecasting OOS window in paper Sections 5.1/5.2.
TEST_START = pd.Timestamp("2020-01-31")
TEST_END = pd.Timestamp("2024-12-31")
COVID_START = pd.Timestamp("2020-02-24")
COVID_END = pd.Timestamp("2020-04-30")
STRESS_PCTILE = 95.0

# Static reference values from the previous R2 run (test window
# 2024-01-02 → 2024-12-31), embedded here ONLY for cross-check printing
# and AUC-drift sanity flagging. Not consumed by any computation.
R2_OLD_MU_TEST = {
    "hmm":   (-5.1835, -4.6396, -4.5788),
    "iohmm": (-5.2277, -4.8559),
    "msar":  (-5.3618, -4.6129),
}
R2_OLD_AUC = {
    "hmm":   0.8085,
    "iohmm": 0.9575,
    "msar":  0.8249,
}
AUC_DRIFT_FLAG = 0.10


# ──────────────────────────────────────────────────────────────────────
# Loaders / helpers
# ──────────────────────────────────────────────────────────────────────

def _load_posteriors(name: str, canonical_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load R1 posteriors parquet, reindex to canonical_index if shorter."""
    path = CACHE_DIR / f"regime_diagnostics_{name}.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.astype(np.float64)

    n_orig = len(df)
    range_str = f"[{df.index.min().date()}, {df.index.max().date()}]"
    test_mask_orig = (df.index >= TEST_START) & (df.index <= TEST_END)
    n_test_orig = int(test_mask_orig.sum())
    n_nan_test_orig = int(df.loc[test_mask_orig].iloc[:, 0].isna().sum())

    print(
        f"  {name.upper():<5} parquet audit: "
        f"rows={n_orig}, range={range_str}, "
        f"test_rows={n_test_orig}, NaN-in-test={n_nan_test_orig}"
    )

    if n_orig < len(canonical_index):
        missing_n = len(canonical_index) - n_orig
        df = df.reindex(canonical_index)
        print(
            f"  {name.upper():<5} reindexed to canonical "
            f"({n_orig} → {len(canonical_index)}, "
            f"{missing_n} dates filled with NaN)"
        )
        # Recount NaN in test window after reindex
        n_nan_test_after = int(
            df.loc[(df.index >= TEST_START) & (df.index <= TEST_END)].iloc[:, 0].isna().sum()
        )
        print(f"  {name.upper():<5} test-window NaN after reindex: {n_nan_test_after}")
    return df


def _load_transmat(name: str) -> np.ndarray:
    path = CACHE_DIR / f"regime_diagnostics_{name}_transmat.parquet"
    df = pd.read_parquet(path)
    return df.to_numpy(dtype=np.float64)


def _weighted_mean_std(
    g: np.ndarray, log_rv: np.ndarray, valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Within-regime weighted mean and std of log_rv, masked by `valid`.

    g   : (T, K)
    log_rv : (T,)
    valid  : (T,) bool — rows usable
    Returns (mu_k, sigma_k) of shape (K,).
    """
    g = g[valid]
    lr = log_rv[valid]
    K = g.shape[1]
    mu = np.full(K, np.nan, dtype=np.float64)
    sd = np.full(K, np.nan, dtype=np.float64)
    for k in range(K):
        w = g[:, k]
        w_sum = float(w.sum())
        if w_sum > 0.0:
            mk = float((w * lr).sum() / w_sum)
            mu[k] = mk
            sd[k] = float(np.sqrt((w * (lr - mk) ** 2).sum() / w_sum))
    return mu, sd


def _full_sample_means(g: np.ndarray, log_rv: np.ndarray) -> np.ndarray:
    """For the R1 cross-check print only."""
    K = g.shape[1]
    mu = np.full(K, np.nan, dtype=np.float64)
    valid = (~np.isnan(log_rv)) & np.all(~np.isnan(g), axis=1)
    g = g[valid]
    lr = log_rv[valid]
    for k in range(K):
        w = g[:, k]
        w_sum = float(w.sum())
        if w_sum > 0.0:
            mu[k] = float((w * lr).sum() / w_sum)
    return mu


# ──────────────────────────────────────────────────────────────────────
# Per-model metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics_for_model(
    name: str,
    canonical_index: pd.DatetimeIndex,
    rv_gk: pd.Series,
    log_rv: np.ndarray,
    vol_pct: np.ndarray,
) -> Tuple[List[Dict], Dict]:
    """Return (per-regime row dicts, model-level diagnostics dict)."""
    df_g = _load_posteriors(name, canonical_index)
    P = _load_transmat(name)
    K = P.shape[0]

    # Align gamma to canonical_index (already aligned; defensive)
    df_g = df_g.reindex(canonical_index)
    g = df_g.to_numpy(dtype=np.float64)

    # ── (1) durations ─────────────────────────────────────────────────
    durations = 1.0 / (1.0 - np.diag(P))
    durations = durations.astype(np.float64)

    # ── (2) within-regime moments on the TEST WINDOW ──────────────────
    test_mask = (canonical_index >= TEST_START) & (canonical_index <= TEST_END)
    valid_test = (
        test_mask
        & (~np.isnan(log_rv))
        & np.all(~np.isnan(g), axis=1)
    )
    n_test_days = int(valid_test.sum())
    mu_test, sd_test = _weighted_mean_std(g, log_rv, valid_test)

    # Cross-check: R1's full-sample means vs R2's test-window means
    mu_full = _full_sample_means(g, log_rv)

    # ── (3) stress alignment AUC on the TEST WINDOW ───────────────────
    vol_pct_test = vol_pct[valid_test]
    p_high_test = g[valid_test, K - 1]
    threshold = float(np.percentile(vol_pct_test, STRESS_PCTILE))
    is_stress = (vol_pct_test >= threshold).astype(int)
    n_stress_days = int(is_stress.sum())
    auc = float(roc_auc_score(is_stress, p_high_test))

    # ── (4) COVID period fraction (full sample) ───────────────────────
    covid_mask = (canonical_index >= COVID_START) & (canonical_index <= COVID_END)
    covid_valid = covid_mask & np.all(~np.isnan(g), axis=1)
    n_covid_days = int(covid_valid.sum())
    p_high_covid = g[covid_valid, K - 1]
    covid_frac_high = float(np.mean(p_high_covid > 0.5)) if n_covid_days > 0 else np.nan

    rows: List[Dict] = []
    for k in range(K):
        rows.append(
            {
                "model": name,
                "regime": int(k),
                "duration_days": float(durations[k]),
                "mu_log_rv": float(mu_test[k]),
                "sigma_log_rv": float(sd_test[k]),
                "n_test_days": n_test_days,
                "stress_auc": auc,
                "stress_threshold_vol_pct": threshold,
                "n_stress_days": n_stress_days,
                "covid_frac_high_regime": covid_frac_high,
                "n_covid_days": n_covid_days,
            }
        )

    diag = {
        "K": K,
        "mu_full": mu_full,
        "mu_test": mu_test,
        "sd_test": sd_test,
        "durations": durations,
        "auc": auc,
        "threshold": threshold,
        "n_stress_days": n_stress_days,
        "n_test_days": n_test_days,
        "n_covid_days": n_covid_days,
        "covid_frac_high": covid_frac_high,
    }
    return rows, diag


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("R2-revised: regime quality metrics")
    print(f"test window:  {TEST_START.date()} → {TEST_END.date()}")
    print(f"COVID window: {COVID_START.date()} → {COVID_END.date()}")
    print(f"stress percentile: {STRESS_PCTILE}")
    print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rv_gk = get_canonical_rv_gk().astype(np.float64)
    canonical_index = pd.DatetimeIndex(rv_gk.index)
    log_rv = np.log(rv_gk).to_numpy(dtype=np.float64)
    vol_pct = (np.sqrt(rv_gk.to_numpy(dtype=np.float64)) * 100.0).astype(np.float64)
    print(f"canonical rv_gk: T={len(canonical_index)}, "
          f"{canonical_index[0].date()} → {canonical_index[-1].date()}")
    print()

    print("── R1 parquet coverage audit ─────────────────────────────────")
    all_rows: List[Dict] = []
    diags: Dict[str, Dict] = {}
    for name in MODELS:
        rows, diag = compute_metrics_for_model(
            name, canonical_index, rv_gk, log_rv, vol_pct
        )
        all_rows.extend(rows)
        diags[name] = diag
    print()

    # ── Three-way means cross-check ───────────────────────────────────
    # R1 full-sample (2019-01-02 to 2024-12-31), R2-old (2024-only),
    # R2-new (2020-01-31 to 2024-12-31, matches forecasting OOS).
    print("── Within-regime μ_log_rv: R1 full / R2-old (2024) / R2-new (2020-2024) ──")
    for name in MODELS:
        d = diags[name]
        old = R2_OLD_MU_TEST[name]
        print(
            f"  {name.upper():<5}  R1 full-sample      : "
            f"{[round(float(x), 4) for x in d['mu_full']]}"
        )
        print(
            f"         R2-old (2024-only)  : "
            f"{[round(float(x), 4) for x in old]}"
        )
        print(
            f"         R2-new (2020-2024)  : "
            f"{[round(float(x), 4) for x in d['mu_test']]}"
        )
    print()

    # ── Build CSV (sorted by model then regime) ───────────────────────
    df = pd.DataFrame(all_rows)
    float_cols = [
        "duration_days",
        "mu_log_rv",
        "sigma_log_rv",
        "stress_auc",
        "stress_threshold_vol_pct",
        "covid_frac_high_regime",
    ]
    for c in float_cols:
        df[c] = df[c].astype(np.float64)
    int_cols = ["regime", "n_test_days", "n_stress_days", "n_covid_days"]
    for c in int_cols:
        df[c] = df[c].astype(np.int64)
    df = df.sort_values(["model", "regime"]).reset_index(drop=True)

    # ── Validation ────────────────────────────────────────────────────
    print("── Validation ────────────────────────────────────────────────")

    # (a) n_test_days in expected range (~1,230-1,250 for 2020-01-31 to 2024-12-31)
    n_test = {name: diags[name]["n_test_days"] for name in MODELS}
    print(f"  (a) n_test_days per model: {n_test}  (expected ~1,230-1,250)")
    for name, v in n_test.items():
        assert 1200 <= v <= 1260, f"{name}: n_test_days={v} outside ~1200-1260"

    # (b) n_stress_days in expected range (~60-65, 5% of ~1,238)
    n_stress = {name: diags[name]["n_stress_days"] for name in MODELS}
    print(f"  (b) n_stress_days per model: {n_stress}  (expected ≈ 60-65)")
    for name, v in n_stress.items():
        assert 50 <= v <= 80, f"{name}: n_stress_days={v} far from expected ≈60-65"

    # (c) AUC in [0, 1] AND >= 0.5 (high-regime ordering must not invert)
    aucs = {name: diags[name]["auc"] for name in MODELS}
    print(f"  (c) stress_auc per model: "
          f"{ {k: round(v, 4) for k, v in aucs.items()} }  "
          f"(must be in [0,1] and ≥ 0.5)")
    for name, v in aucs.items():
        assert 0.0 <= v <= 1.0, f"AUC out of [0,1] for {name}: {v}"
        assert v >= 0.5, (
            f"{name}: AUC={v:.4f} < 0.5 — the high-regime ordering from "
            f"R1's permutation has inverted as a stress classifier"
        )

    # AUC drift vs. the previous R2 (2024-only) run
    print("       AUC drift vs. R2-old (2024-only) — flag if |drift| > "
          f"{AUC_DRIFT_FLAG}:")
    for name in MODELS:
        old = R2_OLD_AUC[name]
        new = aucs[name]
        drift = new - old
        flag = "  ⚠ FLAG" if abs(drift) > AUC_DRIFT_FLAG else ""
        print(
            f"        {name.upper():<5} old={old:.4f}  new={new:.4f}  "
            f"Δ={drift:+.4f}{flag}"
        )

    # (d) within-regime means strictly increasing in regime index
    print("  (d) test-window μ triples (must increase strictly):")
    for name in MODELS:
        mu = diags[name]["mu_test"]
        print(f"        {name.upper():<5} {[round(float(x), 4) for x in mu]}")
        assert np.all(np.diff(mu) > 0), (
            f"{name}: test-window means NOT strictly increasing: {mu.tolist()}"
        )

    # Stress threshold and COVID-frac diagnostics (informational, not asserted
    # beyond covid_frac ∈ [0,1])
    thresholds = {name: diags[name]["threshold"] for name in MODELS}
    print(f"  stress_threshold_vol_pct per model: "
          f"{ {k: round(v, 4) for k, v in thresholds.items()} }  "
          "(expected ~18-24 vs. the 16.68 from 2024-only)")

    covid_frac = {name: diags[name]["covid_frac_high"] for name in MODELS}
    print(f"  covid_frac_high_regime per model: "
          f"{ {k: round(v, 4) for k, v in covid_frac.items()} }")
    for name, v in covid_frac.items():
        assert 0.0 <= v <= 1.0, f"{name}: covid_frac out of [0,1]: {v}"

    print()

    # ── Write CSV ─────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "regime_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print()

    # (e) print full CSV
    print("── results/regime_metrics.csv ────────────────────────────────")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
    print()
    print("R2-revised complete: regime metrics written "
          f"(test window {TEST_START.date()} → {TEST_END.date()}).")


if __name__ == "__main__":
    main()
