import numpy as np

from NonlinearHypothesis import *
from NonlinearStatistic import *


class FixedDensity(NonlinearHypothesis):
    """ Class FixedDensity is a nonlinear hypothesis test (abstract class NonlinearHypothesis) with null hypothesis
        that the data are independently drawn from a fixed probability density. For details see:
            Nonlinear Time Series Analysis. Holger Kantz and Thomas Schreiber. Cambridge University Press, 2003. Sec 7.1


    """

    def __init__(self, time_series: np.ndarray, modulo: int, statistic: NonlinearStatistic,
                 num_surrogates=100, alpha=0.05):
        super().__init__(time_series=time_series, statistic=statistic, num_surrogates=num_surrogates, alpha=alpha)
        self.modulo = modulo

    """ Surrogate data is generated via shuffling the data wherein every entry at time_series.index modulo mod = a 
        are randomly shuffled with another entry at time_series.index modulo mod = a"""
    def generate_null_series(self):
        self.null_series = np.empty(shape=(len(self.time_series), self.num_surrogates))
        for i in range(self.num_surrogates):
            self.null_series[:, i] = self.time_series
            for j in range(len(self.time_series)):
                idx_mod = np.modulo(j, self.modulo)
                idx = np.random.randint(0, len(self.time_series), 1)
                while np.mod(idx, self.modulo) != idx_mod:
                    idx = np.random.randint(0, len(self.time_series), 1)
                self.null_series[i, idx] = self.null_series[i, j]
                self.null_series[i, j] = self.null_series[i, idx]

    def compute_lambda_null_series(self):
        pass

    def compute_lambda_time_series(self):
        pass
