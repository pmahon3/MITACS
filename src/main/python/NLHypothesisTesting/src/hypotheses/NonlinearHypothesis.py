import numpy as np


class NonlinearHypothesis:
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

    def __init__(self, time_series: np.ndarray, num_surrogates=100, alpha=0.05):
        self.time_series = time_series
        self.lambda_ts = None
        self.null_series = None
        self.lambda_ns = None
        self.num_surrogates = num_surrogates
        self.alpha = alpha

    def generate_null_series(self) -> np.ndarray:
        pass

    def get_time_series(self):
        return self.time_series

    def get_null_series(self):
        return self.null_series
