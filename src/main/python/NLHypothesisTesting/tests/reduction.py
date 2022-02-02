from statsmodels.tsa.arima.model import ARIMA
import pickle
import numpy as np
import warnings
from main.python.NLHypothesisTesting.src import hypotheses
from main.python.NLHypothesisTesting.src import statistics

warnings.filterwarnings('ignore')

# params
theta = 10
tau = -1
tp = 1
E = 4
d = 0
q = 0
performance_measure = 'rho'

in_file = "../../../resources/data/processed_data_sets/temp/forecast_demand.csv"
data = np.genfromtxt(in_file, delimiter=",", skip_header=1)
start_idx = max(np.argwhere(np.isnan(data[:, 3])))[0]
data_full = data[(start_idx+1):len(data), :]
data = data_full[:, 2]
print('Length: ' + str(len(data)))

model = ARIMA(endog=data, order=(E, d, q))
fitted = model.fit()
predictions = fitted.predict()
residuals = predictions - data

library = '1 ' + str(len(residuals))
prediction = library
hypothesis = hypotheses.NonlinearHypothesis(residuals)
statistic = statistics.SMapPrediction(
                    hypothesis=hypothesis,
                    embedding_dimension=E,
                    tau=tau,
                    theta=theta,
                    time_to_prediction=tp,
                    library=library,
                    prediction=prediction
                )
statistic.compute_performance(10, 10, tau, performance_measure=performance_measure)

with open('reduction.pickle', 'wb') as file:
    pickle.dump(obj=[statistic, model, fitted, predictions, data, data_full, start_idx], file=file, protocol=pickle.HIGHEST_PROTOCOL)
