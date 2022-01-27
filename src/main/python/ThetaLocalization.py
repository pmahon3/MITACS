import numpy as np
import pandas as pd
from NonlinearStatistic import *
from NonlinearHypothesis import *
import pyEDM


class ThetaLocalization(NonlinearStatistic):
    """ThetaLocalization is an object of type NonlinearStatistic that uses pyEDM's PredictNonlinear, which describes
    the degree of nonlinearity for a delay embedded time series, to test a NonlinearHypothesis.
            ...

            Attributes
            ----------
            hypothesis : NonlinearHypothesis
                The hypothesis that this statistic will be used to test
            embedding_dimension: int
                The dimension in which the time series is delay embedded.
            library: str
                The data used from which make predictions in pyEDM's PredictNonlinear. Two integer indices separated by
                a space: e.g. "1 500"
            prediction: str
                The data to predict using the library data in pyEDM's PredictNonlinear. Two integer indices separated by
                a space: e.g. "501 1000"
            time_to_prediction: int
                How far ahead to make predictions
            tau: int
                The time lag used to perform the time series embedding
            """

    def __init__(self, hypothesis: NonlinearHypothesis, embedding_dimension: int, library: str, prediction: str,
                 time_to_prediction=1, tau=-1):
        self.hypothesis = hypothesis
        self.embedding_dimension = embedding_dimension
        self.library = library
        self.prediction = prediction
        self.time_to_prediction = time_to_prediction
        self.tau = tau

    def compute_lmbda(self, null_series=False):
        if null_series:
            out = []
            for i in range(self.hypothesis.num_surrogates):
                columns = self.null_series.dtype.names[0]
                target = columns
                out.appned(
                    pyEDM.PredictNonlinear(
                        dataFrame=self.hypothesis.null_series[:, i],
                        lib=self.library,
                        pred=self.predicition,
                        E=self.embedding_dimension,
                        Tp=self.time_to_prediction,
                        tau=self.tau,
                        columns=columns,
                        target=target
                    )
                )
            self.hypothesis.lambda_ns = out
        else:
            columns = self.hypothesis.time_series.dtype.names[0]
            target = columns
            self.hypothesis.lambda_ts = pyEDM.PredictNonlinear(
                dataFrame=self.hypothesis.time_series,
                lib=self.library,
                pred=self.predicition,
                E=self.embedding_dimension,
                Tp=self.time_to_prediction,
                tau=self.tau,
                columns=columns,
                target=target
            )
