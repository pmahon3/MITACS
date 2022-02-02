import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyEDM

from main.python.NLHypothesisTesting.src.visualizations.plot_3d_surface import plot_3d_surface
from statsmodels.tsa.stattools import acf
from sklearn import metrics
from math import sqrt

with open('reduction.pickle', 'rb') as file:
    statistic, model, fitted, predictions, data, data_full, start_idx = pickle.load(file)


residuals = data - predictions

plt.plot(residuals[-125:])
plt.xlabel('Time (Hours)')
plt.ylabel('Residual (MW)')
plt.title('AR Demand Residuals')
plt.show()

plt.plot(acf(residuals, nlags=200)[-125:])
plt.xlabel('Lag (Hours)')
plt.ylabel('ACF')
plt.title('Autocorrelation of AR Demand Residuals')
plt.show()

plot_3d_surface(statistic.performance[0], 'E', statistic.performance[1], r'$\theta$', statistic.performance[2],
                r'$\rho$')

optimal_idx = np.argmax(statistic.performance[2])
E = statistic.performance[0][optimal_idx]
theta = statistic.performance[1][optimal_idx]

print('Dimension: ' + str(E))
print('Theta    : ' + str(theta))

ts = pd.DataFrame(residuals)
ts.insert(0, 1, np.linspace(1, len(ts), len(ts)))
ts.columns = ['Time', 'Series']
columns = ts.columns[1]
target = columns

split_idx = int(len(data)*0.8)
library = '1 ' + str(split_idx)
prediction = str(split_idx + 1) + " " + str(len(data))

out = pyEDM.SMap(
    dataFrame=ts,
    E=E,
    theta=theta,
    tau=-1,
    Tp=1,
    lib=library,
    pred=prediction,
    target=target,
    columns=target,
    showPlot=False
)


idx = len(out['predictions']['Predictions'])-1
prediction_prime = out['predictions']['Predictions'][-idx:] + predictions[-idx:]
ar_prime_rmse = sqrt(metrics.mean_squared_error(data[-idx:], prediction_prime))
ar_rmse = sqrt(metrics.mean_squared_error(predictions[-idx:], data_full[-idx:, 3]))
print('RMSE of AR model with SMap residual predictions: ' + str(ar_prime_rmse))
print('RMSE of AR model w/o  SMap residual predictions: ' + str(ar_rmse))
print('Relative RMSE change: ' + str((ar_prime_rmse-ar_rmse)/ar_rmse))

prev = 24
plt.plot(range(prev), prediction_prime[-prev:], label='AR with SMap AR Residual Prediction')
plt.plot(range(prev), data[-prev:], label='Real Data')
plt.plot(range(prev), predictions[-prev:], label='AR')
plt.legend()
plt.xlabel('Time (Hours)')
plt.ylabel('Demand (MW)')
plt.title('Comparison of AR and AR with SMap AR Residual Prediction')
plt.show()
