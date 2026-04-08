from __future__ import annotations

import pandas as pd

from IOHMM.regimes.features import build_vol_iohmm_dataset
from IOHMM.regimes.iohmm import GaussianIOHMM
from IOHMM.regimes.diagnostics import summarize_regimes
from data_preprocessing.data_adapter import YFinanceAdapter


TRAIN_START = "2019-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"


def main() -> None:
    adapter = YFinanceAdapter()

    tickers = ["SPY", "TLT", "HYG", "UUP", "GLD"]

    # Fetch full range covering both train and test periods
    raw = adapter.get_data(
        tickers=tickers,
        start_date="2018-06-01",  # extra lead for rolling windows
        end_date="2025-01-31",
        force_refresh=False,
    )

    prepared = build_vol_iohmm_dataset(
        raw,
        target_ticker="SPY",
        external_tickers=("TLT", "HYG", "UUP", "GLD"),
        rv_window_target=10,
        rv_window_external=5,
        strictly_external_inputs=True,
    )

    # ----- train / test split -----
    dates = prepared.dates
    train_mask = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    test_mask = (dates >= TEST_START) & (dates <= TEST_END)

    X_train, y_train = prepared.X[train_mask], prepared.y[train_mask]
    X_test, y_test = prepared.X[test_mask], prepared.y[test_mask]
    dates_train, dates_test = dates[train_mask], dates[test_mask]

    # ----- fit on training period -----
    model = GaussianIOHMM(
        n_states=3,
        emission_ridge=1e-4,
        transition_l2=1e-3,
        max_iter=100,
        tol=1e-4,
    )

    fit_result = model.fit(X_train, y_train)

    # ----- in-sample results -----
    train_results = model.make_results_frame(
        dates=dates_train,
        X=X_train,
        y=y_train,
        use_viterbi=True,
    )
    train_summary = summarize_regimes(train_results)

    # ----- out-of-sample results -----
    test_results = model.make_results_frame(
        dates=dates_test,
        X=X_test,
        y=y_test,
        use_viterbi=True,
    )
    test_summary = summarize_regimes(test_results)

    # ----- print report -----
    print("\nFeature names:")
    for f in prepared.feature_names:
        print(f"  - {f}")

    print("\nFit result:")
    print(f"  converged: {fit_result.converged}")
    print(f"  n_iter:    {fit_result.n_iter}")
    print(f"  final ll:  {fit_result.log_likelihoods[-1]:.4f}")

    print(f"\nTrain score (log-lik): {model.score(X_train, y_train):.4f}")
    print(f"Test  score (log-lik): {model.score(X_test, y_test):.4f}")

    print("\n=== Train regime summary ===")
    print(train_summary)

    print("\n=== Test regime summary ===")
    print(test_summary)

    print("\nTest results tail:")
    print(test_results.tail())

    # ----- persist -----
    train_results.to_csv("spy_vol_iohmm_train_results.csv")
    test_results.to_csv("spy_vol_iohmm_test_results.csv")
    train_summary.to_csv("spy_vol_iohmm_train_summary.csv", index=False)
    test_summary.to_csv("spy_vol_iohmm_test_summary.csv", index=False)


if __name__ == "__main__":
    main()
