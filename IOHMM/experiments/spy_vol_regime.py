from __future__ import annotations

import pandas as pd

from IOHMM.regimes.features import build_vol_iohmm_dataset
from IOHMM.regimes.iohmm import GaussianIOHMM
from IOHMM.regimes.diagnostics import summarize_regimes
from data_preprocessing.data_adapter import YFinanceAdapter


def main() -> None:
    adapter = YFinanceAdapter()

    tickers = ["SPY", "TLT", "HYG", "UUP", "GLD"]

    raw = adapter.get_data(
        tickers=tickers,
        start_date="2012-01-01",
        end_date="2025-12-31",
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

    model = GaussianIOHMM(
        n_states=3,
        emission_ridge=1e-4,
        transition_l2=1e-3,
        max_iter=100,
        tol=1e-4,
    )

    fit_result = model.fit(prepared.X, prepared.y)

    results = model.make_results_frame(
        dates=prepared.dates,
        X=prepared.X,
        y=prepared.y,
        use_viterbi=False,
    )

    summary = summarize_regimes(results)

    print("\nFeature names:")
    for f in prepared.feature_names:
        print(f"  - {f}")

    print("\nFit result:")
    print(f"  converged: {fit_result.converged}")
    print(f"  n_iter:    {fit_result.n_iter}")
    print(f"  final ll:  {fit_result.log_likelihoods[-1]:.4f}")

    print("\nRegime summary:")
    print(summary)

    print("\nTail of results:")
    print(results.tail())

    results.to_csv("spy_vol_iohmm_results.csv")
    summary.to_csv("spy_vol_iohmm_summary.csv", index=False)


if __name__ == "__main__":
    main()