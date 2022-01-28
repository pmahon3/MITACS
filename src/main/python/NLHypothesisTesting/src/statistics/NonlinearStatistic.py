from main.python.NLHypothesisTesting.src.hypotheses.NonlinearHypothesis import NonlinearHypothesis


class NonlinearStatistic:

    def __init__(self, hypothesis: NonlinearHypothesis):
        """NonlinearStatistic is an object that defines the type of statistic used in a NonlinearHypothesis test
        """
        self.hypothesis = hypothesis

    def compute_lambda(self, null_series=False):
        pass
