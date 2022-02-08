import numpy as np
import pandas as pd
import pyEDM

from main.python.NLHypothesisTesting.src.statistics.SMapPrediction import SMapPrediction
from main.python.NLHypothesisTesting.src.hypotheses.NonlinearHypothesis import NonlinearHypothesis


class Reduction:
    def __init__(self, time_series: np.array, predicted_series: np.array = None, residuals: np.array = None):
        """ model_array columns:
            0 - Raw time series
            1 - AR predictions of 0
            2 - AR residuals (0-1)
            3 - SMap prediction of residuals (SMap of 2)
            4 - SMap prediction of residuals residuals (2 - 3)
            5 - SMap + AR predictions (1 + 3)
            6 - SMap + AR predictions residuals (0 - 5)
            """
        self.model_array = np.empty(shape=(len(time_series), 7))
        self.model_array.fill(np.nan)
        self.model_array[:, 0] = time_series
        self.ts_statistic = SMapPrediction(
            hypothesis=NonlinearHypothesis(self.model_array[:, 0]),
            library='1 ' + str(len(self.model_array[:, 0])),
            prediction='1 ' + str(len(self.model_array[:, 0]))
        )
        self.rs_statistic = SMapPrediction(
            hypothesis=NonlinearHypothesis(self.model_array[:, 2]),
            library='1 ' + str(len(self.model_array[:, 2])),
            prediction='1 ' + str(len(self.model_array[:, 2]))
        )
        if predicted_series is not None:
            self.model_array[:, 1] = predicted_series
        if residuals is not None:
            self.model_array[:, 2] = residuals

    def optimize_time_series_smap(self, max_dim: int = 5, max_theta: int = 5, tau: int = -1,
                                  performance_measure: str = 'rho'):
        return self.ts_statistic.optimize_smap(max_dim=max_dim, max_theta=max_theta, tau=tau,
                                               performance_measure=performance_measure)

    def optimize_residual_smap(self, max_dim: int = 5, max_theta: int = 5, tau: int = -1,
                               performance_measure: str = 'rho'):
        return self.rs_statistic.optimize_smap(max_dim=max_dim, max_theta=max_theta, tau=tau,
                                               performance_measure=performance_measure)

    def predict_residuals(self, split: float):
        ts = pd.DataFrame(self.model_array[:, 2])
        ts.insert(0, 1, np.linspace(1, len(ts), len(ts)))
        ts.columns = ['Time', 'Series']
        columns = ts.columns[1]
        target = columns

        str_start = int(split * len(ts))
        library = '1 ' + str(str_start)
        prediction = str(str_start + 1) + ' ' + str(len(ts))
        idx_start = str_start - 1

        residual_predictions = pyEDM.SMap(
            dataFrame=ts,
            E=self.rs_statistic.embedding_dimension,
            theta=self.rs_statistic.theta,
            tau=self.rs_statistic.tau,
            Tp=self.rs_statistic.time_to_prediction,
            lib=library,
            pred=prediction,
            target=target,
            columns=target,
            showPlot=False
        )

        self.model_array[str_start:, 3] = residual_predictions['predictions']['Predictions'][:-1]
        self.model_array[:, 4] = self.model_array[:, 2] - self.model_array[:, 3]
        self.model_array[:, 5] = self.model_array[:, 1] + self.model_array[:, 3]
        self.model_array[:, 6] = self.model_array[:, 0] - self.model_array[:, 5]
        return self.model_array

    def fit_model(self):
        pass

    def predict_time_series(self, split: int = 0.8):
        pass
