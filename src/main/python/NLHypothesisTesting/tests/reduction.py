import pyEDM
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import warnings
import tqdm
from main.python.NLHypothesisTesting.src import hypotheses
from main.python.NLHypothesisTesting.src import statistics
from main.python.NLHypothesisTesting.src import visualizations

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
idx = max(np.argwhere(np.isnan(data[:, 3])))[0]
data = data[(idx+1):len(data), 3]
print('Length: ' + str(len(data)))

model = ARIMA(endog=data, order=(E, d, q))
fitted = model.fit()
predictions = fitted.predict()
residuals = predictions[1:] - data[1:]
plt.plot(residuals[-72:])
plt.xlabel('Time (Hours)')
plt.ylabel('Residual (MW)')
plt.title('AR(' + str(E) + ',' + str(d) + ',' + str(q) + ')')
plt.show()

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
    pickle.dump([statistic, model, fitted, predictions], protocol=pickle.HIGHEST_PROTOCOL)
