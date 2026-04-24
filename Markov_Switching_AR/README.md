# Markov Switching AR

This folder contains the implementation of the Markov-Switching Autoregressive (MS-AR) model for regime detection and forecasting on broad-based market ETFs.

## Files

- `__init__.py`: package initializer  
- `adapter.py`: Yahoo Finance data adapter with caching support  
- `model_utils.py`: model fitting, forecasting, evaluation, and plotting utilities  
- `test_basic.ipynb`: basic data preparation and preprocessing checks  
- `test_model.ipynb`: model training, validation, and diagnostics  
- `result.ipynb`: final results and comparison between models  

## Main Idea

We implement the Markov-Switching AR (MS-AR) model in two settings:

1. **Return Prediction**  
   We model daily log returns and evaluate whether regime-switching improves short-term return predictability.

2. **Realized Volatility Prediction**  
   We model log realized volatility (constructed from rolling return volatility) to better capture latent market regimes such as low-volatility and high-volatility states.

The hidden state follows a Markov chain, allowing the model to capture structural changes in market dynamics over time.

## Key Insights

- Return predictability is limited, consistent with the Efficient Market Hypothesis.
- The MS-AR model provides more meaningful regime identification when applied to realized volatility.
- The model identifies persistent regimes corresponding to calm and turbulent market periods.

## How to Run

Run the notebooks in the following order:

1. `test_basic.ipynb`  
   - Prepares data and verifies the construction of:
     - returns  
     - realized volatility (RV)  
     - log RV  

2. `test_model.ipynb`  
   - Trains MS-AR models for:
     - return prediction  
     - realized volatility prediction  
   - Outputs validation metrics and regime diagnostics  

3. `result.ipynb`  
   - Presents final results, including:
     - model configurations  
     - forecasting performance  
     - regime persistence and interpretability  
     - comparison between return and RV models  