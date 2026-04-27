import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from data_preprocessing.data_adapter import YFinanceAdapter
from data_preprocessing.price_utils import extract_adjusted_close


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


def extract_price_series(
    df: pd.DataFrame,
    ticker: str,
    price_col: str = "Close"
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
) -> pd.DataFrame:
    if adapter is None:
        adapter = YFinanceAdapter()

    raw = adapter.get_data(
        tickers=[ticker],
        start_date=start_date,
        end_date=end_date,
    )

    # adjusted prices via auto_adjust=True
    price = extract_adjusted_close(raw, ticker)
    price = pd.to_numeric(price, errors="coerce").dropna().astype(float)
    price.name = "price"

    df = pd.DataFrame(index=price.index.copy())
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df["price"] = price.astype(float)
    df["log_price"] = np.log(df["price"])
    df["return"] = df["log_price"].diff()
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()

    if df.empty:
        raise ValueError("Return dataframe is empty after preprocessing.")

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
    returns: pd.Series,
    k_regimes: int = 2,
    order: int = 1,
    switching_ar: bool = False,
    switching_variance: bool = True,
    trend: str = "c",
):
    # switching_ar=False: AR coefficients common across regimes (Hamilton 1989 simplification).
    if k_regimes != 2 and not switching_ar:
        raise NotImplementedError(
            "switching_ar=False with k_regimes>2 is untested; set switching_ar=True or use k_regimes=2."
        )

    y = pd.Series(returns).dropna().astype(float)

    if len(y) <= order + 10:
        raise ValueError(
            f"Not enough observations to fit MS-AR(order={order}). Need more than {order + 10}, got {len(y)}."
        )

    model = MarkovAutoregression(
        endog=y,
        k_regimes=k_regimes,
        order=order,
        trend=trend,
        switching_ar=switching_ar,
        switching_variance=switching_variance,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        result = model.fit(disp=False)
    return result


def grid_search_msar(
    train_returns: pd.Series,
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
    y = pd.Series(train_returns).dropna().astype(float)

    for k in regime_list:
        if k != 2:
            raise NotImplementedError(
                "switching_ar=False with k_regimes>2 is untested; set switching_ar=True or use k_regimes=2."
            )
        for order in order_list:
            for sv in switching_variance_list:
                try:
                    res = fit_msar_model(
                        returns=y,
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


def _extract_transition_matrix(result, k_regimes):
    return np.array([[result.transition[j, i] for j in range(k_regimes)]
                     for i in range(k_regimes)])


def _extract_intercepts_and_ar(
    res,
    k_regimes: int = 2,
    order: int = 1
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
    k_regimes: int,
    order: int = 1,
) -> float:
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
    full_returns: pd.Series,
    train_size: int,
    k_regimes: int = 2,
    order: int = 1,
    switching_variance: bool = True,
    trend: str = "c",
) -> pd.DataFrame:
    y = pd.Series(full_returns).dropna().astype(float)

    preds = []
    actuals = []
    dates = []

    for i in range(train_size, len(y)):
        history = y.iloc[:i].copy()
        y_true = float(y.iloc[i])

        try:
            res = fit_msar_model(
                returns=history,
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


def evaluate_forecasts(df_forecast: pd.DataFrame) -> Dict[str, float]:
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
    hit_ratio = np.mean(np.sign(actual) == np.sign(pred))

    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Directional_Accuracy": float(hit_ratio),
    }


def plot_regime_probabilities(
    res,
    returns: pd.Series,
    title: str = "Smoothed Regime Probabilities"
):
    y = pd.Series(returns).dropna().astype(float)
    probs = res.smoothed_marginal_probabilities.copy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(y.index, y.values)
    axes[0].set_title("Log Returns")
    axes[0].grid(True, alpha=0.3)

    for i in range(probs.shape[1]):
        axes[i + 1].plot(probs.index, probs.iloc[:, i])
        axes[i + 1].set_title(f"Probability of Regime {i}")
        axes[i + 1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def run_full_msar_pipeline(config: MSARConfig) -> Dict:
    df = prepare_return_data(
        ticker=config.ticker,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    train_df, val_df, test_df = split_data(
        df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    best_config, tuning_table = grid_search_msar(
        train_returns=train_df["return"],
        regime_list=[config.k_regimes],
        order_list=[1, 2],
        switching_variance_list=[True, False],
        trend=config.trend,
    )

    best_res = fit_msar_model(
        returns=train_df["return"],
        k_regimes=best_config["k_regimes"],
        order=best_config["order"],
        switching_ar=best_config["switching_ar"],
        switching_variance=best_config["switching_variance"],
        trend=best_config["trend"],
    )

    val_input = pd.concat([train_df["return"], val_df["return"]])
    val_forecasts = rolling_forecast_msar(
        full_returns=val_input,
        train_size=len(train_df),
        k_regimes=best_config["k_regimes"],
        order=best_config["order"],
        switching_variance=best_config["switching_variance"],
        trend=best_config["trend"],
    )

    val_metrics = evaluate_forecasts(val_forecasts)

    return {
        "raw_data": df,
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "best_config": best_config,
        "tuning_table": tuning_table,
        "best_result": best_res,
        "validation_forecasts": val_forecasts,
        "validation_metrics": val_metrics,
    }
