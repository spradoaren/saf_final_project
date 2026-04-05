import numpy as np
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression


class MSARModel:
    """
    Markov-Switching Autoregressive Model
    """

    def __init__(self, k_regimes=2, order=1):
        self.k_regimes = k_regimes
        self.order = order
        self.model = None
        self.result = None

    def fit(self, y):
        """
        Fit MS-AR model
        y: pandas Series
        """
        self.model = MarkovAutoregression(
            y,
            k_regimes=self.k_regimes,
            order=self.order,
            switching_variance=True
        )

        self.result = self.model.fit(disp=False)

        return self.result

    def predict(self, start, end):
        """
        In-sample prediction
        """
        return self.result.predict(start=start, end=end)

    def forecast(self, steps=1):
        """
        Out-of-sample forecast
        """
        return self.result.forecast(steps=steps)

    def get_regime_probs(self):
        """
        Smoothed regime probabilities
        """
        return self.result.smoothed_marginal_probabilities

    def summary(self):
        return self.result.summary()
