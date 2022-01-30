from main.python.NLHypothesisTesting.src.hypotheses.NonlinearHypothesis import *
from tqdm import tqdm


class FixedDensity(NonlinearHypothesis):
    """ Class FixedDensity is a nonlinear hypothesis test (abstract class NonlinearHypothesis) with null hypothesis
        that the data are independently drawn from a fixed probability density. For details see:
            Nonlinear Time Series Analysis. Holger Kantz and Thomas Schreiber. Cambridge University Press, 2003. Sec 7.1

        Attributes
        ----------
        modulo: int
            parameter controlling how the data are shuffled. See generate_null_series()
    """

    def __init__(self, time_series: np.ndarray, modulo: int, num_surrogates=100,
                 alpha=0.05):
        super().__init__(time_series=time_series, num_surrogates=num_surrogates, alpha=alpha)
        self.modulo = modulo

    """ Surrogate data is generated via shuffling the data wherein every entry at time_series.index modulo mod = a 
        are randomly shuffled with another entry at time_series.index modulo mod = a"""
    def generate_null_series(self) -> np.ndarray:
        self.null_series = np.empty(shape=(len(self.time_series), self.num_surrogates))
        for i in range(self.num_surrogates):
            self.null_series[:, i] = self.time_series
            subsamples = []
            for n in range(self.modulo):
                np.random.shuffle(self.null_series[n:len(self.null_series):self.modulo])
        return self.null_series
