import numpy as np
import pandas as pd
import pyEDM

from main.python.NLHypothesisTesting.src.statistics.SMapPrediction import SMapPrediction
from main.python.NLHypothesisTesting.src.hypotheses.NonlinearHypothesis import NonlinearHypothesis
from statsmodels.tsa.arima.model import ARIMA


class ARReduction:
    def __init__(self, time_series_statistic: SMapPrediction):
        self.statistic = time_series_statistic
        self.residual_statistic = None
        self.model = None
        self.fitted = None
        self.ar_predictions = None
        self.ar_residuals = None
        self.normality = None

        self.residual_predictions = np.empty(shape=len(self.statistic.hypothesis.time_series))
        self.smap_residuals = None

    def compute_parameter_performance(self, max_dim: int, max_theta: int, tau: int, performance_measure: str):
        self.statistic.compute_parameter_performance(max_dim=max_dim, max_theta=max_theta, tau=tau,
                                                     performance_measure=performance_measure)
        return self.statistic.set_optimal_parameters()

    def fit_and_predict_arima(self, p: int = None, d: int = 0, q: int = 0):
        if p is None:
            p = self.statistic.embedding_dimension
        self.model = ARIMA(endog=self.statistic.hypothesis.time_series, order=(p, d, q))
        self.fitted = self.model.fit()
        self.ar_predictions = self.fitted.predict()
        self.ar_residuals = self.fitted.resid
        self.normality = self.fitted.test_normality(method='jarquebera')

        return self.ar_predictions, self.ar_residuals, self.normality

    def optimize_ar_residuals_smap(self, tau: int = -1, time_to_prediction: int = 1, performance_measure: str = 'rho'):
        self.residual_statistic = SMapPrediction(
            hypothesis=NonlinearHypothesis(self.ar_residuals),
            embedding_dimension=None,
            library='1 ' + str(len(self.ar_residuals)),
            prediction='1 ' + str(len(self.ar_residuals)),
            theta=None,
            time_to_prediction=time_to_prediction,
            tau=tau
        )
        self.residual_statistic.compute_parameter_performance(max_dim=10, max_theta=10, tau=tau,
                                                              performance_measure=performance_measure)
        self.residual_statistic.set_optimal_parameters()

    def predict_ar_residuals(self, split: float):
        ts = pd.DataFrame(self.ar_residuals)
        ts.insert(0, 1, np.linspace(1, len(ts), len(ts)))
        ts.columns = ['Time', 'Series']
        columns = ts.columns[1]
        target = columns

        start = int(split * len(self.statistic.hypothesis.time_series))
        library = '1 ' + str(start)
        prediction = str(start + 1) + ' ' + str(len(self.statistic.hypothesis.time_series))

        residual_predictions = pyEDM.SMap(
            dataFrame=ts,
            E=self.residual_statistic.embedding_dimension,
            theta=self.residual_statistic.theta,
            tau=self.residual_statistic.tau,
            Tp=self.residual_statistic.time_to_prediction,
            lib=library,
            pred=prediction,
            target=target,
            columns=target,
            showPlot=False
        )
        self.residual_predictions[start:] = np.array(residual_predictions['predictions']['Predictions'])[:-1]
        print()
        print(pyEDM.ComputeError(residual_predictions['predictions']['Observations'], residual_predictions['predictions']['Predictions']))
        print(pyEDM.ComputeError(self.ar_residuals, self.residual_predictions))
