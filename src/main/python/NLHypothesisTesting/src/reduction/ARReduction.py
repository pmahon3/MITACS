import numpy as np

from main.python.NLHypothesisTesting.src.reduction.Reduction import Reduction
from statsmodels.tsa.arima.model import ARIMA


class ARReduction(Reduction):
    def __init__(self, time_series: np.array):
        super().__init__(time_series=time_series)

        self.model = None
        self.fitted = None
        self.normality = None

    def fit_model(self, p: int = None, d: int = 0, q: int = 0):
        if p is None:
            p = self.ts_statistic.embedding_dimension
        self.model = ARIMA(endog=self.ts_statistic.hypothesis.time_series, order=(p, d, q))
        self.fitted = self.model.fit()
        return self.fitted

    def predict_time_series(self, split=None):
        self.model_array[:, 1] = self.fitted.predict()
        self.model_array[:, 2] = self.fitted.resid
        self.rs_statistic.hypothesis.time_series = self.fitted.resid
        self.normality = self.fitted.test_normality(method='jarquebera')
        return self.model_array
