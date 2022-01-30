from main.python.NLHypothesisTesting.src import hypotheses
from main.python.NLHypothesisTesting.src import statistics
from main.python.NLHypothesisTesting.src import visualizations
# from main.python.NLHypothesisTesting.src import example_data
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# params
num_surrogates = 100
bins = 20
lib_length = 24 * 90
modulo = 23
error = 'rho'
max_dim = 10
max_tau = 10
performance_measure = 'rho'

in_file = "../../../resources/data/processed_data_sets/temp/forecast_demand.csv"
data = np.genfromtxt(in_file, delimiter=",", skip_header=1)
# data = np.diff(data, axis=0)
# idx = np.max(np.argwhere(np.isnan(data[:, 3])))
idx = len(data) - lib_length
data = data[idx:, 3]
lib = str(1) + " " + str(len(data))
pred = lib

hypothesis = hypotheses.FixedDensity(
    time_series=data,
    modulo=7)

statistic = statistics.SMapPrediction(
    hypothesis=hypothesis,
    embedding_dimension=3,
    tau=-1,
    theta=10,
    time_to_prediction=1,
    library=lib,
    prediction=pred
)
hypothesis.generate_null_series()
statistic.compute_lambda(null_series=True, error='rho')
statistic.compute_lambda(error='rho')

plt.hist(hypothesis.lambda_ns, bins=20)
plt.vlines(hypothesis.lambda_ts, 0, 1, color='red')
plt.show(block=True)

