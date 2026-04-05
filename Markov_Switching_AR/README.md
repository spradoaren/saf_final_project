# Markov Switching AR

This folder contains the implementation of the Markov-Switching Autoregressive model for regime detection and short-term return forecasting on broad-based market ETFs.

## Files

- `__init__.py`: package initializer
- `adapter.py`: Yahoo Finance data adapter with cache support
- `model_utils.py`: model fitting, forecasting, evaluation, and plotting
- `test_basic.ipynb`: basic data preparation test
- `test_model.ipynb`: model training and validation test
- `result.ipynb`: final result presentation

## Main Idea

We model ETF log returns using a Markov-Switching AR model, where the hidden state follows a Markov chain and return dynamics may differ across regimes.

## How to Run

Open the notebooks and run cells in order:

1. `test_basic.ipynb`
2. `test_model.ipynb`
3. `result.ipynb`
