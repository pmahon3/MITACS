from main.python.NLHypothesisTesting.src.hypotheses.LinearStochastic import *
from main.python.NLHypothesisTesting.src.hypotheses.FixedDensity import *
from main.python.NLHypothesisTesting.src.statistics.ThetaLocalization import *
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

num_surrogates = 100
bins = 20
lib_length = 24 * 30
modulo = 23
error = 'rho'

in_file = "../../../resources/data/processed_data_sets/temp/forecast_demand.csv"
data = np.genfromtxt(in_file, delimiter=",", skip_header=1)
# data = np.diff(data, axis=0)
idx = np.max(np.argwhere(np.isnan(data[:, 3])))
lib = str(len(data) - lib_length) + " " + str(len(data))
pred = lib

hypothesis = LinearStochastic(data[:, 2], num_surrogates=num_surrogates, alpha=0.05)
# hypothesis = FixedDensity(data[:, 2], modulo=modulo, num_surrogates=num_surrogates, alpha=0.05)
hypothesis.generate_null_series()

statistic = ThetaLocalization(
    hypothesis=hypothesis,
    embedding_dimension=6,
    library=lib,
    prediction=pred,
    time_to_prediction=1,
    tau=-1
)
statistic.compute_lambda(error=error)
statistic.compute_lambda(error=error, null_series=True)

plt.hist(hypothesis.lambda_ns, bins=20)
plt.show()
print(hypothesis.lambda_ts)
print(hypothesis.lambda_ns)
