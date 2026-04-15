from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass
class RegimeForecastResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    model: GaussianHMM


def load_return_data(
    csv_path: str,
    date_col: str = "Date",
    price_col: str = "price",
    return_col: str = "log_return",
) -> pd.DataFrame:
    """
    Load the user's return CSV and standardize key columns.

    Expected columns from your screenshot:
    Date, price, log_return, simple_return, abs_return, squ_return, rv_21d_annualised
    """
    df = pd.read_csv(csv_path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    if return_col not in df.columns:
        raise ValueError(f"Missing return column: {return_col}")

    if price_col in df.columns:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    for col in [return_col, "simple_return", "abs_return", "squ_return", "rv_21d_annualised"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df



def build_features(
    df: pd.DataFrame,
    return_col: str = "log_return",
    use_columns: Optional[List[str]] = None,
    vol_window: int = 21,
) -> pd.DataFrame:
    """
    Build a return-centered feature matrix for HMM.

    Default idea:
    - return itself
    - absolute return
    - realized / rolling volatility
    """
    out = df.copy()

    if "abs_return" not in out.columns:
        out["abs_return"] = out[return_col].abs()

    if "rv_21d_annualised" not in out.columns:
        out["rv_21d_annualised"] = out[return_col].rolling(vol_window).std() * np.sqrt(252)

    if use_columns is None:
        use_columns = [return_col, "abs_return", "rv_21d_annualised"]

    feats = out[use_columns].copy()
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats



def train_test_split_by_date(
    feature_df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = feature_df.loc[:train_end].copy()
    if test_end is None:
        test = feature_df.loc[test_start:].copy()
    else:
        test = feature_df.loc[test_start:test_end].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty dataset.")

    return train, test



def fit_hmm(
    X: np.ndarray,
    n_states: int = 3,
    covariance_type: str = "full",
    n_iter: int = 500,
    random_state: int = 42,
) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        min_covar=1e-6,
    )
    model.fit(X)
    return model



def filtered_state_probabilities(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Real-time filtered probabilities using only information up to time t.
    """
    logprob, posteriors = model.score_samples(X)
    _ = logprob
    return posteriors



def summarise_regimes(
    model: GaussianHMM,
    X: np.ndarray,
    index: pd.Index,
    return_series: pd.Series,
) -> pd.DataFrame:
    """
    Summarize each inferred regime using return-based statistics.
    """

    probs = filtered_state_probabilities(model, X)
    states = probs.argmax(axis=1)

    df = pd.DataFrame(index=index)
    df["state"] = states
    df["return"] = return_series.reindex(index)

    rows = []
    total_n = len(df)

    for s in range(model.n_components):
        mask = df["state"] == s
        sub = df.loc[mask, "return"].dropna()

        mean_ret = float(sub.mean()) if len(sub) else np.nan
        vol = float(sub.std()) if len(sub) else np.nan
        sharpe_like = np.nan if vol in [0, np.nan] else mean_ret / vol

        rows.append(
            {
                "state": s,
                "count": int(mask.sum()),
                "fraction": float(mask.mean()),
                "avg_return": mean_ret,
                "volatility": vol,
                "annualised_return_approx": float(mean_ret * 252) if len(sub) else np.nan,
                "annualised_vol_approx": float(vol * np.sqrt(252)) if len(sub) else np.nan,
                "return_to_vol_ratio": sharpe_like,
            }
        )

    return pd.DataFrame(rows).sort_values("avg_return").reset_index(drop=True)



def _fit_ar1(y: pd.Series) -> Tuple[float, float]:
    y = y.dropna()
    if len(y) < 5:
        return float(y.mean()), 0.0

    y_lag = y.shift(1).dropna()
    y_now = y.loc[y_lag.index]

    X = np.column_stack([np.ones(len(y_lag)), y_lag.values])
    beta = np.linalg.lstsq(X, y_now.values, rcond=None)[0]
    alpha, phi = float(beta[0]), float(beta[1])
    return alpha, phi



def rolling_regime_forecast(
    feature_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    return_col: str = "log_return",
    train_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
    n_states: int = 3,
    expanding: bool = True,
    min_train_size: int = 252,
) -> RegimeForecastResult:
    aligned_returns = raw_df[return_col].reindex(feature_df.index).dropna()
    feature_df = feature_df.reindex(aligned_returns.index).dropna()
    aligned_returns = aligned_returns.reindex(feature_df.index)

    train_feat = feature_df.loc[:train_end].copy()
    test_feat = feature_df.loc[test_start:].copy()

    if len(train_feat) < min_train_size:
        raise ValueError(f"Need at least {min_train_size} training rows, got {len(train_feat)}")

    full_feat = pd.concat([train_feat, test_feat], axis=0)
    test_dates = test_feat.index

    preds = []

    for current_date in test_dates[:-1]:
        pos = full_feat.index.get_loc(current_date)
        if isinstance(pos, slice):
            raise ValueError("Duplicate dates found in feature dataframe.")

        train_slice = full_feat.iloc[: pos + 1] if expanding else full_feat.iloc[max(0, pos + 1 - len(train_feat)) : pos + 1]
        if len(train_slice) < min_train_size:
            continue

        model = fit_hmm(train_slice.values, n_states=n_states)
        probs = filtered_state_probabilities(model, train_slice.values)
        current_probs = probs[-1]
        next_state_probs = current_probs @ model.transmat_
        state_return_means = model.means_[:, 0]
        regime_pred = float(next_state_probs @ state_return_means)

        next_date = test_dates[test_dates.get_loc(current_date) + 1]
        actual = float(aligned_returns.loc[next_date])

        preds.append(
            {
                "forecast_date": next_date,
                "regime_pred": regime_pred,
                "actual_return": actual,
            }
        )

    pred_df = pd.DataFrame(preds)
    if pred_df.empty:
        raise ValueError("No forecasts were generated. Check your date split and training size.")

    pred_df["forecast_date"] = pd.to_datetime(pred_df["forecast_date"])
    pred_df = pred_df.set_index("forecast_date")

    metrics = evaluate_forecasts(pred_df)
    final_model = fit_hmm(train_feat.values, n_states=n_states)
    summary = summarise_regimes(final_model, train_feat.values, train_feat.index, aligned_returns)

    return RegimeForecastResult(summary=summary, predictions=pred_df, metrics=metrics, model=final_model)


def evaluate_forecasts(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate return forecasts using multiple forecasting metrics.

    Metrics included:
    - RMSE
    - MAE
    - directional accuracy
    - out-of-sample R^2 (relative to zero forecast)
    - IC (correlation between forecast and realized return)
    """

    rows = []

    for col in ["regime_pred"]:
        y_true = pred_df["actual_return"].astype(float)
        y_pred = pred_df[col].astype(float)

        err = y_true - y_pred

        mae = float(np.abs(err).mean())
        mse = float((err ** 2).mean())
        rmse = float(np.sqrt(mse))

        directional_accuracy = float((np.sign(y_true) == np.sign(y_pred)).mean())

        # Out-of-sample R^2 relative to a zero-return benchmark
        denom = float((y_true ** 2).sum())
        num = float((err ** 2).sum())
        r2_oos = np.nan if denom == 0 else 1.0 - num / denom

        # Information Coefficient: correlation between predicted and realized returns
        if y_pred.std() == 0 or y_true.std() == 0:
            ic = np.nan
        else:
            ic = float(np.corrcoef(y_pred, y_true)[0, 1])

        rows.append(
            {
                "model": col,
                "rmse": rmse,
                "mae": mae,
                "mse": mse,
                "directional_accuracy": directional_accuracy,
                "r2_oos_vs_zero": r2_oos,
                "ic": ic,
            }
        )

    return pd.DataFrame(rows).sort_values("rmse")



def regime_persistence_from_transition_matrix(model):
    """
    Compute regime persistence metrics implied by the HMM transition matrix.

    Metrics:
    - self-transition probability
    - expected duration
    """

    rows = []
    transmat = model.transmat_

    for s in range(model.n_components):
        p_ss = float(transmat[s, s])
        expected_duration = np.inf if p_ss >= 0.999999 else 1.0 / (1.0 - p_ss)

        rows.append(
            {
                "state": s,
                "self_transition_prob": p_ss,
                "expected_duration_periods": expected_duration,
            }
        )

    return pd.DataFrame(rows)

def realised_regime_persistence(
    model: GaussianHMM,
    X: np.ndarray,
    index: pd.Index,
) -> pd.DataFrame:
    """
    Compute realized consecutive run lengths for each inferred state.
    """

    probs = filtered_state_probabilities(model, X)
    states = probs.argmax(axis=1)

    state_series = pd.Series(states, index=index, name="state")

    runs = []
    start_idx = 0
    values = state_series.values

    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            runs.append(
                {
                    "state": int(values[i - 1]),
                    "start_date": state_series.index[start_idx],
                    "end_date": state_series.index[i - 1],
                    "run_length": i - start_idx,
                }
            )
            start_idx = i

    runs.append(
        {
            "state": int(values[-1]),
            "start_date": state_series.index[start_idx],
            "end_date": state_series.index[-1],
            "run_length": len(values) - start_idx,
        }
    )

    run_df = pd.DataFrame(runs)

    summary = (
        run_df.groupby("state")["run_length"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n_runs",
                "mean": "avg_run_length",
                "median": "median_run_length",
                "max": "max_run_length",
            }
        )
    )

    return summary

def build_market_stress_indicators(
    raw_df: pd.DataFrame,
    return_col: str = "log_return",
    price_col: str = "price",
    vol_col: str = "rv_21d_annualised",
    vol_quantile: float = 0.9,
    drawdown_window: int = 252,
) -> pd.DataFrame:
    """
    Build simple market stress indicators:
    - high volatility episode
    - negative return day
    - large drawdown episode
    """

    df = raw_df.copy()

    # High-volatility indicator
    if vol_col not in df.columns:
        raise ValueError(f"Missing volatility column: {vol_col}")
    vol_threshold = df[vol_col].quantile(vol_quantile)
    df["is_high_vol"] = (df[vol_col] >= vol_threshold).astype(int)

    # Negative return indicator
    df["is_negative_return"] = (df[return_col] < 0).astype(int)

    # Drawdown indicator based on rolling peak
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col}")
    rolling_peak = df[price_col].rolling(drawdown_window, min_periods=1).max()
    df["drawdown"] = df[price_col] / rolling_peak - 1.0

    # Mark severe drawdown days, for example worse than -10%
    df["is_drawdown_stress"] = (df["drawdown"] <= -0.10).astype(int)

    return df

def evaluate_regime_alignment(
    model: GaussianHMM,
    X: np.ndarray,
    index: pd.Index,
    stress_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate how well each regime aligns with stress-related market episodes.

    Metrics reported by state:
    - fraction of days in high-vol periods
    - fraction of days in negative-return periods
    - fraction of days in drawdown stress periods
    """

    probs = filtered_state_probabilities(model, X)
    states = probs.argmax(axis=1)

    df = pd.DataFrame(index=index)
    df["state"] = states

    aligned = stress_df.reindex(index)
    df["is_high_vol"] = aligned["is_high_vol"]
    df["is_negative_return"] = aligned["is_negative_return"]
    df["is_drawdown_stress"] = aligned["is_drawdown_stress"]

    rows = []
    for s in range(model.n_components):
        sub = df[df["state"] == s]

        rows.append(
            {
                "state": s,
                "count": len(sub),
                "pct_high_vol_days": float(sub["is_high_vol"].mean()) if len(sub) else np.nan,
                "pct_negative_return_days": float(sub["is_negative_return"].mean()) if len(sub) else np.nan,
                "pct_drawdown_stress_days": float(sub["is_drawdown_stress"].mean()) if len(sub) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def evaluate_regime_interpretability(regime_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple interpretability diagnostics based on separation in
    average return and volatility across inferred regimes.

    The goal is not to define a universal interpretability score, but to
    quantify whether the estimated states are economically distinguishable.
    """

    required_cols = {"state", "avg_return", "volatility", "fraction"}
    missing = required_cols - set(regime_summary.columns)
    if missing:
        raise ValueError(
            "regime_summary is missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = regime_summary.copy().sort_values("state").reset_index(drop=True)

    return_spread = float(out["avg_return"].max() - out["avg_return"].min())
    vol_spread = float(out["volatility"].max() - out["volatility"].min())

    # Weighted averages help benchmark how large the spreads are relative to
    # the overall regime distribution in the sample.
    weights = out["fraction"].astype(float)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("regime fractions must sum to a positive value.")
    weights = weights / weight_sum

    weighted_avg_abs_return = float((weights * out["avg_return"].abs()).sum())
    weighted_avg_volatility = float((weights * out["volatility"]).sum())

    return_spread_ratio = (
        np.nan
        if weighted_avg_abs_return == 0
        else float(return_spread / weighted_avg_abs_return)
    )
    vol_spread_ratio = (
        np.nan
        if weighted_avg_volatility == 0
        else float(vol_spread / weighted_avg_volatility)
    )

    # A simple ranking check: if the ordering of states by return is not the
    # same as the ordering by volatility, the regimes may reflect richer market
    # structure than a single monotonic risk scale.
    return_rank_order = out.sort_values("avg_return")["state"].tolist()
    vol_rank_order = out.sort_values("volatility")["state"].tolist()
    different_rank_order = return_rank_order != vol_rank_order

    diagnostics = pd.DataFrame(
        [
            {
                "n_states": int(len(out)),
                "return_spread_across_states": return_spread,
                "vol_spread_across_states": vol_spread,
                "weighted_avg_abs_return": weighted_avg_abs_return,
                "weighted_avg_volatility": weighted_avg_volatility,
                "return_spread_ratio": return_spread_ratio,
                "vol_spread_ratio": vol_spread_ratio,
                "different_return_vs_vol_rank_order": different_rank_order,
            }
        ]
    )

    return diagnostics