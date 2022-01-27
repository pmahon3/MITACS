from abc import (
    ABC,
    abstractmethod
)

import numpy
import numpy as np

import NonlinearStatistic


class NonlinearHypothesis(ABC):
    """Class Nonlinear Hypothesis defines null hypothesis testing for nonlinear structure of a given
    time series

    Attributes
    ----------
    ts : ndarray numpy array.
        an nx1 of a time series of length n.
    statistic : NonlinearStatistic object that defines the statistic against which to perform the hypothesis test
    alpha : float
        significance level for hypothesis test
    """

    def __init__(self, time_series: np.ndarray, statistic: NonlinearStatistic, num_surrogates=100, alpha=0.05):
        self.time_series = time_series
        self.statistic = statistic
        self.num_surrogates = num_surrogates
        self.alpha = alpha

    """Method lambda_ts computes 

        ...

        Attributes
        ----------
        ts : ndarray numpy array.
            an nx1 of a time series of length n.
        statistic : NonlinearStatistic object that defines the statistic against which to perform the hypothesis test
        alpha : float
            significance level for hypothesis test
        """
    @property
    def lambda_ts(self):
        return self.statistic.compute_lmbda(self, null_series=False)

    @property
    def lambda_ns(self):
        return self.statistic.compute_lmbda(self, null_series=True)

    @property
    def null_series(self):
        ...

    @abstractmethod
    def compute_lambda_time_series(self):
        self.statistic.compute_lmbda(self, self.time_series)

    @abstractmethod
    def compute_lambda_null_series(self):
        self.statistic.compute_lmbda(self, self.null_series)

    @abstractmethod
    def generate_null_series(self):
        ...
        return self.null_series

    def get_time_series(self):
        return self.time_series

    def get_null_series(self):
        return self.null_series

    def get_lambda_nl(self):
        return self.lambda_nl
