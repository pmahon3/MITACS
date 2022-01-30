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

    def make_block(self, embedding_dimension, tau):
        series = []
        spread = embedding_dimension * tau
        for i in reversed(range(len(self.time_series)-1)):
            if (i + spread) < 0:
                break
            vector = np.empty(shape=(1, embedding_dimension))
            for dim in range(embedding_dimension):
                vector[0][dim] = self.time_series[i + dim * tau]
            series.append(vector)
        return series

    def get_time_series(self):
        return self.time_series

    def get_null_series(self):
        return self.null_series
