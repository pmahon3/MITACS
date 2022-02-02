from main.python.NLHypothesisTesting.src import hypotheses
from main.python.NLHypothesisTesting.src import statistics
from main.python.NLHypothesisTesting.src import example_data
import pickle
import matplotlib.pyplot as plt
import numpy as np
import warnings
import tqdm

warnings.filterwarnings('ignore')

# params
num_surrogates = 100
bins = 20
lib_length = 24 * 90
modulo = 23
theta = 10
tau = -1
tp = 1
E = 4
mod = 23
stat = 'SMap'
performance_measure = 'rho'

in_file = "../../../resources/data/processed_data_sets/temp/forecast_demand.csv"

lsg_dir = 'pickles/linear_stochastic_gaussian/LSG_'
fd_dir = 'pickles/fixed_density/FD_'
lsg_file = stat + '_n' + str(num_surrogates) + '_E' + str(4) + '_tau' + str(-tau) + '_theta' + str(theta) + '_' + \
                  'tp' + str(tp) + '.pickle'
fd_file = stat + '_n' + str(num_surrogates) + '_E' + str(4) + '_tau' + str(-tau) + '_theta' + str(theta) + '_tp' + str(tp)\
                 + '_mod' + str(mod) + '.pickle'

data = np.genfromtxt(in_file, delimiter=",", skip_header=1)
data = data[(len(data) - lib_length):len(data), 3]

lib = "1 " + str(len(data))
pred = lib

# lsg
LSG_hypothesis = hypotheses.LinearStochasticGaussian(data)
statistic = statistics.SMapPrediction(
    hypothesis=LSG_hypothesis,
    embedding_dimension=E,
    tau=tau,
    theta=theta,
    time_to_prediction=tp,
    library=lib,
    prediction=pred
)
LSG_hypothesis.generate_null_series()
statistic.compute_lambda(performance_measure=performance_measure)
statistic.compute_lambda(null_series=True, performance_measure=performance_measure)

with open(lsg_dir + lsg_file, 'wb') as file:
    pickle.dump(statistic, file, protocol=pickle.HIGHEST_PROTOCOL)

# fd
FD_hypothesis = hypotheses.FixedDensity(data, modulo=modulo)
statistic.hypothesis = FD_hypothesis
FD_hypothesis.generate_null_series()
statistic.compute_lambda(performance_measure=performance_measure)
statistic.compute_lambda(null_series=True, performance_measure=performance_measure)

with open(fd_dir + fd_file, 'wb') as file:
    pickle.dump(statistic, file, protocol=pickle.HIGHEST_PROTOCOL)

# delta lsg
data = np.diff(data, axis=0)
statistic.library = "1 " + str(len(data))
statistic.prediction = statistic.library

LSG_hypothesis.time_series = data
statistic.hypothesis = LSG_hypothesis
LSG_hypothesis.generate_null_series()
statistic.compute_lambda(performance_measure=performance_measure)
statistic.compute_lambda(null_series=True, performance_measure=performance_measure)

with open(lsg_dir + 'delta_' + lsg_file, 'wb') as file:
    pickle.dump(statistic, file, protocol=pickle.HIGHEST_PROTOCOL)

# delta fd
FD_hypothesis.time_series = data
statistic.hypothesis = FD_hypothesis
LSG_hypothesis.generate_null_series()
statistic.compute_lambda(performance_measure=performance_measure)
statistic.compute_lambda(null_series=True, performance_measure=performance_measure)

with open(fd_dir + 'delta_' + fd_file, 'wb') as file:
    pickle.dump(statistic, file, protocol=pickle.HIGHEST_PROTOCOL)