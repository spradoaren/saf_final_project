import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression


class MSARModel:
    """
    Markov Switching AR Model
    """

    def __init__(self, n_regimes=2, order=1, switching_variance=True):
        self.n_regimes = n_regimes
        self.order = order
        self.switching_variance = switching_variance

        self.model = None
        self.result = None

    def preprocess(self, df: pd.DataFrame, price_col="Adj Close"):
        """
        Convert OHLC → log returns
        Compatible with adapter output
        """

        if isinstance(df.columns, pd.MultiIndex):
            df = df[price_col]

        if price_col in df.columns:
            series = df[price_col]
        else:
            series = df.iloc[:, 0]

        returns = np.log(series).diff().dropna()

        return returns

    def fit(self, returns):
        self.model = MarkovAutoregression(
            returns,
            k_regimes=self.n_regimes,
            order=self.order,
            switching_variance=self.switching_variance
        )

        self.result = self.model.fit(disp=False)

        return self.result

    def get_regime_probabilities(self):
        return self.result.smoothed_marginal_probabilities

    def forecast(self, steps=1):
        return self.result.forecast(steps=steps)

    def evaluate(self, test_returns):
        forecast = self.forecast(len(test_returns))
        forecast.index = test_returns.index

        mse = np.mean((forecast - test_returns) ** 2)
        return mse
