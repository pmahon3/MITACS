from NonlinearHypothesis import *
import numpy as np


class LinearStochastic(NonlinearHypothesis):
    """ Class LinearStochastic is a nonlinear hypothesis test (abstract class NonlinearHypothesis) with null hypothesis
    that the data is generated from a linear stochastic function with gaussian inputs. For details see:
        Nonlinear Time Series Analysis. Holger Kantz and Thomas Schreiber. Cambridge University Press, 2003. Sec 7.1"""

    def __init__(self, time_series: np.ndarray, statistic: NonlinearStatistic, num_surrogates=100, alpha=0.05):
        super.__init__(time_series=time_series, statistic=statistic, num_surrogates=num_surrogates, alpha=alpha)
        self.n = num_surrogates

    """ Surrogate data is generated via phase randomization. See: 
            Nonlinear Time Series Analysis. Holger Kantz and Thomas Schreiber. Cambridge University Press, 2003. 
            Sec 7.1.2"""
    def generate_null_series(self):
        self.null_series = np.empty(shape=(len(self.time_series), self.num_surrogates))
        for i in range(self.num_surrogates):
            ft = np.fft.fft(self.time_series)
            s = np.empty(shape=len(self.time_series), dtype=complex)
            for k in range(int(len(self.time_series) / 2 - 1)):
                phase = np.random.rand(1) * 2 * np.pi
                s[k] = (ft[k] * np.exp(1j * phase))[0]
                s[len(self.time_series) - k - 1] = ft[len(self.time_series) - k - 1] * np.exp(1j * -phase)[0]
            self.null_series[:, i] = np.fft.ifft(s)

    def compute_lambda_time_series(self):
        pass

    def compute_lambda_null_series(self):
        pass