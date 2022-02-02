from main.python.NLHypothesisTesting.src import hypotheses
from main.python.NLHypothesisTesting.src import statistics
from main.python.NLHypothesisTesting.src import example_data
import pickle
import numpy as np
import warnings
import tqdm

warnings.filterwarnings('ignore')

# params
max_dim = 10
max_tau = 10
fixed_tau = -1
max_theta = 10
fixed_theta = 9
performance_measure = 'rho'

in_file = "../../../resources/data/processed_data_sets/temp/forecast_demand.csv"

surface_out_files = [
    'pickles/test/predict_nonlinear/delta_demand_E_theta_rho.pickle',
    'pickles/test/predict_nonlinear/demand_E_theta_rho.pickle',
    'pickles/test/smap/delta_demand_E_tau_rho.pickle',
    'pickles/test/smap/demand_E_tau_rho.pickle'
]

data = np.genfromtxt(in_file, delimiter=",", skip_header=1)
data = data[:, 3]

for out_file in surface_out_files:
    print(out_file)
    ts = data
    if 'delta' in out_file:
        ts = np.diff(ts, axis=0)

    lib_length = 24 * 90
    idx = len(ts) - lib_length
    lib = str(idx) + " " + str(len(ts))
    pred = lib

    dims, params, skill = [], [], []

    hypothesis = hypotheses.NonlinearHypothesis(time_series=ts)
    statistic = statistics.NonlinearStatistic(hypothesis=hypothesis)

    for dim in tqdm.tqdm(range(1, max_dim + 1)):
        if 'tau' in out_file:
            for tau in range(1, max_tau + 1):
                statistic = statistics.SMapPrediction(
                    hypothesis=hypothesis,
                    embedding_dimension=dim,
                    tau=-int(tau),
                    theta=fixed_theta,
                    time_to_prediction=1,
                    library=lib,
                    prediction=pred
                )
                statistic.compute_lambda(performance_measure=performance_measure)
                dims.append(dim)
                params.append(tau)
                skill.append(statistic.hypothesis.lambda_ts)

        elif 'theta' in out_file:
            for theta in range(1, max_theta + 1):
                statistic = statistics.ThetaLocalization(
                    hypothesis=hypothesis,
                    embedding_dimension=dim,
                    tau=fixed_tau,
                    theta=str(theta),
                    time_to_prediction=1,
                    library=lib,
                    prediction=pred
                )
                statistic.compute_lambda()
                dims.append(dim)
                params.append(theta)
                skill.append(statistic.hypothesis.lambda_ts)

    with open(out_file, 'wb') as file:
        pickle.dump([dims, params, skill], file, protocol=pickle.HIGHEST_PROTOCOL)
