from abc import (
    ABC,
    abstractmethod
)

from NonlinearHypothesis import NonlinearHypothesis


class NonlinearStatistic(ABC):

    def __init__(self, hypothesis: NonlinearHypothesis):
        """NonlinearStatistic is an object that defines the type of statistic used in a NonlinearHypothesis test
        ...

        Attributes
        ----------
        hypothesis : NonlinearHypothesis
            The hypothesis that this statistic will be used to test
        """
        self.hypothesis = hypothesis

    """"""
    @abstractmethod
    def compute_lmbda(self):
        ...
