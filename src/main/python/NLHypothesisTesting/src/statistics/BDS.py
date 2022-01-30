from main.python.NLHypothesisTesting.src.hypotheses.NonlinearHypothesis import NonlinearHypothesis
import numpy as np


class Bds(NonlinearHypothesis):

    def __init__(self, hypothesis, epsilon, embedding_dimension, tau, norm=2):
        super().__init__(hypothesis=hypothesis)
        self.epsilon = epsilon
        self.pairwise_norms = np.empty(shape=(len(self.hypothesis.time_series), len(self.hypothesis.time_series)))
        self.norm = norm
        self.embedding_dimension = embedding_dimension
        self.tau = tau
        self.block = self.make_block()

    def compute_lambda(self, null_series=False):
        for i in range(len(self.hypothesis.time_series)):
            for j in range(len(self.hypothesis.time_series)):
                self.pairwise_norms[i][j] = np.linalg.norm()
