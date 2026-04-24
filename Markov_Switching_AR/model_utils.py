import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from adapter import YFinanceAdapter

warnings.filterwarnings("ignore")


@dataclass
class MSARConfig:
    ticker: str = "SPY"
    start_date: str = "2019-01-01"
    end_date: str = "2025-01-01"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    k_regimes: int = 2
    order: int = 1
    switching_ar: bool = False
    switching_variance: bool = True
    trend: str = "c"
    rv_window: int = 5


def extract_price_series(
    df: pd.DataFrame,
    ticker: str,
    price_col: str = "Adj Close",
) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        if price_col in level0:
            if ticker in df[price_col].columns:
                series = df[price_col][ticker].copy()
            else:
                raise ValueError(f"{ticker} not found under '{price_col}'.")
        elif "Close" in level0:
            if ticker in df["Close"].columns:
                series = df["Close"][ticker].copy()
            else:
                raise ValueError(f"{ticker} not found under 'Close'.")
        else:
            raise ValueError(f"Cannot find '{price_col}' or 'Close' in dataframe columns.")
    else:
        if price_col in df.columns:
            series = df[price_col].copy()
        elif "Close" in df.columns:
            series = df["Close"].copy()
        else:
            raise ValueError("Cannot find price column in dataframe.")

    series = pd.to_numeric(series, errors="coerce").dropna()
    series.name = "price"
    return series


def prepare_return_data(
    ticker: str,
    start_date: str,
    end_date: str,
    adapter: Optional[YFinanceAdapter] = None,
    rv_window: int = 5,
) -> pd.DataFrame:
    """
    Prepare SPY price, daily log return, realized volatility proxy, and log realized volatility.

    return: daily log return, r_t = log(P_t) - log(P_{t-1})
    rv: rolling realized volatility proxy, std(r_{t-rv_window+1}, ..., r_t)
    log_rv: log(rv), used as the RV prediction target
    """
    if adapter is None:
        adapter = YFinanceAdapter()

    raw = adapter.get_data(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
    )

    price = extract_price_series(raw, ticker=ticker, price_col="Adj Close")

    df = pd.DataFrame(index=price.index.copy())
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df["price"] = price.astype(float)
    df["log_price"] = np.log(df["price"])
    df["return"] = df["log_price"].diff()

    # Realized volatility proxy. This keeps the project aligned with RV forecasting
    # while using only daily data already available from Yahoo Finance.
    df["rv"] = df["return"].rolling(rv_window).std()
    df["log_rv"] = np.log(df["rv"])

    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()

    if df.empty:
        raise ValueError("Dataframe is empty after preprocessing.")

    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    n = len(df)
    if n < 30:
        raise ValueError("Dataset is too small to split meaningfully.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(f"Bad split: train={len(train)}, val={len(val)}, test={len(test)}")

    return train, val, test


def fit_msar_model(
    series: pd.Series,
    k_regimes: int = 2,
    order: int = 1,
    switching_ar: bool = False,
    switching_variance: bool = True,
    trend: str = "c",
):
    y = pd.Series(series).dropna().astype(float)

    if len(y) <= order + 10:
        raise ValueError(
            f"Not enough observations to fit MS-AR(order={order}). "
            f"Need more than {order + 10}, got {len(y)}."
        )

    model = MarkovAutoregression(
        endog=y,
        k_regimes=k_regimes,
        order=order,
        trend=trend,
        switching_ar=switching_ar,
        switching_variance=switching_variance,
    )

    result = model.fit(disp=False)
    return result


def grid_search_msar(
    train_series: pd.Series,
    regime_list: List[int] = None,
    order_list: List[int] = None,
    switching_variance_list: List[bool] = None,
    trend: str = "c",
) -> Tuple[Dict, pd.DataFrame]:
    if regime_list is None:
        regime_list = [2]
    if order_list is None:
        order_list = [1, 2]
    if switching_variance_list is None:
        switching_variance_list = [True, False]

    rows = []
    y = pd.Series(train_series).dropna().astype(float)

    for k in regime_list:
        for order in order_list:
            for sv in switching_variance_list:
                try:
                    res = fit_msar_model(
                        series=y,
                        k_regimes=k,
                        order=order,
                        switching_ar=False,
                        switching_variance=sv,
                        trend=trend,
                    )

                    rows.append({
                        "k_regimes": k,
                        "order": order,
                        "switching_variance": sv,
                        "aic": float(res.aic),
                        "bic": float(res.bic),
                        "llf": float(res.llf),
                        "error": None,
                    })

                except Exception as e:
                    rows.append({
                        "k_regimes": k,
                        "order": order,
                        "switching_variance": sv,
                        "aic": np.nan,
                        "bic": np.nan,
                        "llf": np.nan,
                        "error": str(e),
                    })

    result_df = pd.DataFrame(rows)
    valid_df = result_df.dropna(subset=["bic"]).copy()

    if valid_df.empty:
        raise ValueError("All grid search model fits failed.")

    result_df = result_df.sort_values(by=["bic", "aic"], ascending=True).reset_index(drop=True)
    best_row = valid_df.sort_values(by=["bic", "aic"], ascending=True).iloc[0].to_dict()

    best_config = {
        "k_regimes": int(best_row["k_regimes"]),
        "order": int(best_row["order"]),
        "switching_variance": bool(best_row["switching_variance"]),
        "trend": trend,
        "switching_ar": False,
    }

    return best_config, result_df


def _extract_transition_matrix(res, k_regimes: int = 2) -> np.ndarray:
    if k_regimes != 2:
        raise NotImplementedError("This helper currently supports k_regimes=2 only.")

    params = pd.Series(res.params, index=res.model.param_names)

    p00_name = [x for x in params.index if "p[0->0]" in x]
    p10_name = [x for x in params.index if "p[1->0]" in x]

    if len(p00_name) == 0 or len(p10_name) == 0:
        raise ValueError("Could not find transition probabilities in params.")

    p00 = float(params[p00_name[0]])
    p10 = float(params[p10_name[0]])

    P = np.array([
        [p00, 1.0 - p00],
        [p10, 1.0 - p10],
    ])

    return P


def _extract_intercepts_and_ar(
    res,
    k_regimes: int = 2,
    order: int = 1,
):
    params = pd.Series(res.params, index=res.model.param_names)

    intercepts = []
    for r in range(k_regimes):
        name_candidates = [x for x in params.index if (f"const[{r}]" in x or f"intercept[{r}]" in x)]
        if len(name_candidates) == 0:
            raise ValueError(f"Could not find intercept for regime {r}.")
        intercepts.append(float(params[name_candidates[0]]))

    ar_coefs = []
    for lag in range(1, order + 1):
        name_candidates = [x for x in params.index if f"ar.L{lag}" in x]
        if len(name_candidates) == 0:
            raise ValueError(f"Could not find AR coefficient for lag {lag}.")
        ar_coefs.append(float(params[name_candidates[0]]))

    return np.array(intercepts), np.array(ar_coefs)


def one_step_forecast_from_result(
    res,
    history: pd.Series,
    k_regimes: int = 2,
    order: int = 1,
) -> float:
    if k_regimes != 2:
        raise NotImplementedError("Current forecast helper supports k_regimes=2 only.")

    history = pd.Series(history).dropna().astype(float)

    filtered_probs = np.asarray(res.filtered_marginal_probabilities.iloc[-1]).reshape(-1)
    P = _extract_transition_matrix(res, k_regimes=k_regimes)
    intercepts, ar_coefs = _extract_intercepts_and_ar(res, k_regimes=k_regimes, order=order)

    next_regime_probs = filtered_probs @ P

    last_vals = history.iloc[-order:].values[::-1]
    ar_part = float(np.dot(ar_coefs, last_vals))

    regime_means = intercepts + ar_part
    forecast = float(np.dot(next_regime_probs, regime_means))

    return forecast


def rolling_forecast_msar(
    full_series: pd.Series,
    train_size: int,
    k_regimes: int = 2,
    order: int = 1,
    switching_variance: bool = True,
    trend: str = "c",
) -> pd.DataFrame:
    y = pd.Series(full_series).dropna().astype(float)

    preds = []
    actuals = []
    dates = []

    for i in range(train_size, len(y)):
        history = y.iloc[:i].copy()
        y_true = float(y.iloc[i])

        try:
            res = fit_msar_model(
                series=history,
                k_regimes=k_regimes,
                order=order,
                switching_ar=False,
                switching_variance=switching_variance,
                trend=trend,
            )

            y_pred = one_step_forecast_from_result(
                res=res,
                history=history,
                k_regimes=k_regimes,
                order=order,
            )

        except Exception:
            y_pred = np.nan

        preds.append(y_pred)
        actuals.append(y_true)
        dates.append(y.index[i])

    out = pd.DataFrame({
        "actual": actuals,
        "pred": preds,
    }, index=pd.to_datetime(dates))

    out["error"] = out["actual"] - out["pred"]
    out = out.dropna().copy()

    if out.empty:
        raise ValueError("All rolling forecasts failed; forecast output is empty.")

    return out


def evaluate_forecasts(df_forecast: pd.DataFrame, target_col: str = "return") -> Dict[str, float]:
    if df_forecast.empty:
        raise ValueError("Forecast dataframe is empty.")

    actual = pd.to_numeric(df_forecast["actual"], errors="coerce")
    pred = pd.to_numeric(df_forecast["pred"], errors="coerce")

    mask = actual.notna() & pred.notna()
    actual = actual[mask]
    pred = pred[mask]

    mse = np.mean((actual - pred) ** 2)
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(mse)

    out = {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
    }

    if target_col == "return":
        hit_ratio = np.mean(np.sign(actual) == np.sign(pred))
        out["Directional_Accuracy"] = float(hit_ratio)

    return out


def summarize_regimes(res, series: pd.Series) -> pd.DataFrame:
    series = pd.Series(series).dropna().astype(float)

    probs = np.asarray(res.smoothed_marginal_probabilities)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)

    common_len = min(len(series), probs.shape[0])

    aligned_series = series.iloc[-common_len:].copy()
    aligned_probs = probs[-common_len:, :]

    out = pd.DataFrame(index=aligned_series.index)
    out["target"] = aligned_series.values

    for i in range(aligned_probs.shape[1]):
        out[f"Regime_{i}"] = aligned_probs[:, i]

    regime_cols = [f"Regime_{i}" for i in range(aligned_probs.shape[1])]
    out["most_likely_regime"] = out[regime_cols].idxmax(axis=1)

    return out


def regime_persistence_metrics(res) -> pd.DataFrame:
    P = _extract_transition_matrix(res, k_regimes=2)

    durations = []
    for i in range(P.shape[0]):
        pii = P[i, i]
        expected_duration = np.inf if np.isclose(1 - pii, 0) else 1.0 / (1.0 - pii)
        durations.append({
            "regime": i,
            "p_ii": float(pii),
            "expected_duration": float(expected_duration),
        })

    return pd.DataFrame(durations)


def regime_interpretability_metrics(res, target_series: pd.Series, target_col: str = "return") -> pd.DataFrame:
    regime_df = summarize_regimes(res, target_series)

    rows = []
    regime_cols = [col for col in regime_df.columns if col.startswith("Regime_")]
    for i, regime_col in enumerate(regime_cols):
        mask = regime_df["most_likely_regime"] == regime_col
        subset = regime_df.loc[mask, "target"]

        rows.append({
            "regime": i,
            "target": target_col,
            "n_obs": int(mask.sum()),
            "mean_target": float(subset.mean()) if len(subset) > 0 else np.nan,
            "std_target": float(subset.std()) if len(subset) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def regime_stress_alignment_metrics(
    res,
    target_series: pd.Series,
    return_series: pd.Series,
    target_col: str = "return",
    vol_window: int = 20,
) -> Dict[str, float]:
    """
    Measure whether the inferred high-volatility regime aligns with market stress.
    Stress is always defined from original returns, even when the MS-AR target is log_rv.
    """
    regime_df = summarize_regimes(res, target_series).copy()

    returns = pd.Series(return_series).dropna().astype(float)
    common_index = regime_df.index.intersection(returns.index)
    regime_df = regime_df.loc[common_index].copy()
    returns = returns.loc[common_index].copy()

    regime_df["original_return"] = returns.values
    regime_df["rolling_vol"] = regime_df["original_return"].rolling(vol_window).std()
    regime_df["drawdown_flag"] = regime_df["original_return"] < regime_df["original_return"].quantile(0.1)
    regime_df["high_vol_flag"] = regime_df["rolling_vol"] > regime_df["rolling_vol"].quantile(0.9)

    regime_cols = [col for col in regime_df.columns if col.startswith("Regime_")]

    if target_col in ["rv", "log_rv"]:
        target_means = {}
        for col in regime_cols:
            target_means[col] = regime_df.loc[regime_df["most_likely_regime"] == col, "target"].mean()
        high_vol_regime = max(target_means, key=target_means.get)
    else:
        vol_means = {}
        for col in regime_cols:
            vol_means[col] = regime_df[col].mul(regime_df["rolling_vol"]).mean()
        high_vol_regime = max(vol_means, key=vol_means.get)

    drawdown_alignment = regime_df.loc[regime_df["drawdown_flag"], high_vol_regime].mean()
    high_vol_alignment = regime_df.loc[regime_df["high_vol_flag"], high_vol_regime].mean()

    return {
        "high_vol_regime": high_vol_regime,
        "avg_prob_high_vol_regime_on_drawdown_days": float(drawdown_alignment) if pd.notna(drawdown_alignment) else np.nan,
        "avg_prob_high_vol_regime_on_high_vol_days": float(high_vol_alignment) if pd.notna(high_vol_alignment) else np.nan,
    }


def plot_regime_probabilities(
    res,
    series: pd.Series,
    title: str = "Smoothed Regime Probabilities",
    target_label: str = "Target Series",
):
    regime_df = summarize_regimes(res, series)
    regime_cols = [col for col in regime_df.columns if col.startswith("Regime_")]

    fig, axes = plt.subplots(
        len(regime_cols) + 1,
        1,
        figsize=(12, 3 * (len(regime_cols) + 1)),
        sharex=True,
    )

    axes[0].plot(regime_df.index, regime_df["target"].values)
    axes[0].set_title(target_label)
    axes[0].grid(True, alpha=0.3)

    for i, col in enumerate(regime_cols, start=1):
        axes[i].plot(regime_df.index, regime_df[col].values)
        axes[i].set_title(f"Probability of {col}")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def run_full_msar_pipeline(config: MSARConfig, target_col: str = "return") -> Dict:
    df = prepare_return_data(
        ticker=config.ticker,
        start_date=config.start_date,
        end_date=config.end_date,
        rv_window=config.rv_window,
    )

    if target_col not in df.columns:
        raise ValueError(f"{target_col} is not in dataframe columns. Available columns: {list(df.columns)}")

    train_df, val_df, test_df = split_data(
        df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    best_config, tuning_table = grid_search_msar(
        train_series=train_df[target_col],
        regime_list=[config.k_regimes],
        order_list=[1, 2],
        switching_variance_list=[True, False],
        trend=config.trend,
    )

    best_res = fit_msar_model(
        series=train_df[target_col],
        k_regimes=best_config["k_regimes"],
        order=best_config["order"],
        switching_ar=best_config["switching_ar"],
        switching_variance=best_config["switching_variance"],
        trend=best_config["trend"],
    )

    val_input = pd.concat([train_df[target_col], val_df[target_col]])
    val_forecasts = rolling_forecast_msar(
        full_series=val_input,
        train_size=len(train_df),
        k_regimes=best_config["k_regimes"],
        order=best_config["order"],
        switching_variance=best_config["switching_variance"],
        trend=best_config["trend"],
    )

    val_metrics = evaluate_forecasts(val_forecasts, target_col=target_col)

    persistence_metrics = regime_persistence_metrics(best_res)
    interpretability_metrics = regime_interpretability_metrics(
        best_res,
        train_df[target_col],
        target_col=target_col,
    )
    stress_alignment_metrics = regime_stress_alignment_metrics(
        best_res,
        target_series=train_df[target_col],
        return_series=train_df["return"],
        target_col=target_col,
    )

    return {
        "raw_data": df,
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "target_col": target_col,
        "best_config": best_config,
        "tuning_table": tuning_table,
        "best_result": best_res,
        "validation_forecasts": val_forecasts,
        "validation_metrics": val_metrics,
        "regime_persistence_metrics": persistence_metrics,
        "regime_interpretability_metrics": interpretability_metrics,
        "regime_stress_alignment_metrics": stress_alignment_metrics,
    }
