from main.python.NLHypothesisTesting.src.hypotheses.NonlinearHypothesis import *
from tqdm import tqdm
import numpy as np


class LinearStochasticEmpirical(NonlinearHypothesis):
    """ Class LinearStochasticGaussian is a nonlinear hypothesis test (abstract class NonlinearHypothesis) with
    null hypothesis that the data is generated from a linear stochastic function with non-gaussian inputs. For details see:
        Nonlinear Time Series Analysis. Holger Kantz and Thomas Schreiber. Cambridge University Press, 2003. Sec 7.1"""
    def __init__(self, time_series: np.ndarray, num_surrogates=100, alpha=0.05):
        super().__init__(time_series=time_series, num_surrogates=num_surrogates, alpha=alpha)

    """ Surrogate data is generated via phase randomization. See: 
            Nonlinear Time Series Analysis. Holger Kantz and Thomas Schreiber. Cambridge University Press, 2003. 
            Sec 7.1.2"""
    def generate_null_series(self) -> np.ndarray:
        print('Generating Surrogates:')
        self.null_series = np.empty(shape=(len(self.time_series), self.num_surrogates))
        for i in tqdm(range(self.num_surrogates), leave=True):
            ft = np.fft.fft(self.invert_distribution())
            s = np.empty(shape=len(self.time_series), dtype=complex)
            for k in range(int(len(self.time_series) / 2 - 1)):
                phase = np.random.rand(1) * 2 * np.pi
                s[k] = (ft[k] * np.exp(1j * phase))[0]
                s[len(self.time_series) - k - 1] = ft[len(self.time_series) - k - 1] * np.exp(1j * -phase)[0]
            self.null_series[:, i] = self.apply_distribution(np.fft.ifft(s))
        return self.null_series

    def compute_empirical_distribution(self):
        return

    def invert_distribution(self):
        return

    def apply_distribution(self):
        return
