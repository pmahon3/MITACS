from main.python.NLHypothesisTesting.src.statistics.NonlinearStatistic import *
from main.python.NLHypothesisTesting.src.hypotheses.NonlinearHypothesis import *
from tqdm import tqdm
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import pandas as pd
import pyEDM


class SMapPrediction(NonlinearStatistic):
    """SMapPrediction is an object of type NonlinearStatistic that uses pyEDM's SMap prediction skill to test a
    NonlinearHypothesis.

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
                 theta: str, time_to_prediction=1, tau=-1):
        super().__init__(hypothesis=hypothesis)
        self.embedding_dimension = embedding_dimension
        self.library = library
        self.prediction = prediction
        self.time_to_prediction = time_to_prediction
        self.tau = tau
        self.theta = theta

    def compute_lambda(self, null_series=False, performance_measure=None):
        if null_series:
            print('Computing lambda:')
            out = []
            for i in tqdm(range(self.hypothesis.num_surrogates), leave=True):
                ts = pd.DataFrame(self.hypothesis.null_series[:, i])
                ts.insert(0, 1, np.linspace(1, len(ts), len(ts)))
                ts.columns = ['Time', 'Series']
                columns = 'Series'
                target = columns
                out.append(
                    pyEDM.SMap(
                        dataFrame=ts,
                        lib=self.library,
                        pred=self.prediction,
                        E=self.embedding_dimension,
                        Tp=self.time_to_prediction,
                        tau=self.tau,
                        columns=columns,
                        target=target,
                        theta=self.theta
                    )
                )
                if performance_measure is None:
                    out[i] = pyEDM.ComputeError(out[i]['predictions']['Observations'], out[i]['predictions']['Predictions'])
                else:
                    out[i] = pyEDM.ComputeError(out[i]['predictions']['Observations'], out[i]['predictions']['Predictions'])[performance_measure]
            self.hypothesis.lambda_ns = out

        else:
            ts = pd.DataFrame(self.hypothesis.time_series)
            ts.insert(0, 1, np.linspace(1, len(ts), len(ts)))
            ts.columns = ['Time', 'Series']
            columns = 'Series'
            target = columns
            out = pyEDM.SMap(
                dataFrame=ts,
                lib=self.library,
                pred=self.prediction,
                E=self.embedding_dimension,
                Tp=self.time_to_prediction,
                tau=self.tau,
                columns=columns,
                target=target
            )

            if performance_measure is None:
                self.hypothesis.lambda_ts = pyEDM.ComputeError(out['predictions']['Observations'],
                                                           out['predictions']['Predictions'])
            else:
                self.hypothesis.lambda_ts = pyEDM.ComputeError(out['predictions']['Observations'],
                                                               out['predictions']['Predictions'])[performance_measure]
