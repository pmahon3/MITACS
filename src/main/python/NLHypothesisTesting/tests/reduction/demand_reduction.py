import numpy as np
import pyEDM
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf

from main.python.NLHypothesisTesting.src.reduction.ARReduction import ARReduction
from main.python.NLHypothesisTesting.src.visualizations.residual_delay_map import plot_residual_delay_map as rdm

in_file = "../../../../resources/data/processed_data_sets/temp/forecast_demand.csv"
data = np.genfromtxt(in_file, delimiter=",", skip_header=1)


# Parameters
split = 0.8
start_idx = max(np.argwhere(np.isnan(data[:, 3])))[0] + 1
data = data[start_idx:, 3]
start_idx = int(split*len(data) + 1)

# Computing

reduction = ARReduction(data)
reduction.optimize_time_series_smap(max_dim=5, max_theta=5, tau=-1, performance_measure='rho')
optimal_dimension = reduction.ts_statistic.embedding_dimension
reduction.fit_model()
reduction.predict_time_series()

reduction.optimize_residual_smap()
reduction.predict_residuals(split=split)

# Plotting
optimal_dimension = reduction.ts_statistic.embedding_dimension

plt.plot(reduction.ts_statistic.hypothesis.time_series)
plt.xlabel('Time')
plt.ylabel('Demand (MW)')
plt.title('Demand')
plt.show()

plt.plot(reduction.model_array[1:, 2])
plt.title('AR(' + str(optimal_dimension) + ') Residuals')
plt.ylabel(r'$x_t - \mathregular{\hat{x}_t}$')
plt.xlabel('Time (t)')
plt.show()

plt.plot(acf(reduction.model_array[1:, 2], nlags=100))
plt.title('ACF of AR(' + str(optimal_dimension) + ') Residuals')
plt.xlabel('Lag')
plt.ylabel('Acf')
plt.show()


rdm(reduction.model_array[:, [0, 2]],
    title='Residual Delay Map for AR(' + str(optimal_dimension) + ')',
    x_label='x_t',
    y_label='AR(' + str(optimal_dimension) + r'$\mathregular{)_{t+1}}$ Residual'
    )

plt.plot(reduction.model_array[:, 2])
plt.title('Residuals of SMap prediction of AR(' + str(optimal_dimension) + ') Residuals')
plt.xlabel('Time (t)')
plt.show()

rdm(reduction.model_array[:, 3:],
    title='Residual Delay Map for SMap Prediction of AR(' + str(optimal_dimension) + ')',
    x_label='Time (t)',
    y_label='AR(' + str(optimal_dimension) + ') Residual' + r'$\mathregular{_{t+1}}$ Residual'
    )

plt.plot(acf(reduction.model_array[start_idx:, 4], nlags=100))
plt.title('ACF of Residuals of SMap Predicted Residuals for AR(' + str(optimal_dimension) + ')')
plt.xlabel('Lag')
plt.ylabel('Acf')
plt.show()

plt.plot(reduction.model_array[start_idx:, 5], label='SMap + AR(' + str(optimal_dimension) + ')')
plt.plot(reduction.model_array[start_idx:, 0], label='Demand')
plt.plot(reduction.model_array[start_idx:, 1], label='AR(' + str(optimal_dimension) + ')')
plt.title('Model Comparison')
plt.ylabel(r'x')
plt.xlabel('Time (t)')
plt.legend()
plt.show()

plt.plot(reduction.model_array[start_idx:, 6])
plt.title('SMap + AR(' + str(optimal_dimension) + ') Residuals')
plt.ylabel(r'$\mathregular{x_{t} - \hat{x}_{t}}$')
plt.xlabel('Time (t)')
plt.show()

plt.plot(acf(reduction.model_array[start_idx:, 6], nlags=100))
plt.title('ACF of Residuals of SMap + AR(' + str(optimal_dimension) + ')')
plt.xlabel('Lag')
plt.ylabel('Acf')
plt.show()

rdm(reduction.model_array[:, [5, 6]],
    title='Residual Delay Map for SMap + AR(' + str(optimal_dimension) + ')',
    x_label='Time (t)',
    y_label='AR(' + str(optimal_dimension) + '$)_t$ Residual'
    )

# Error Comparison
print('AR(' + str(optimal_dimension) + ') Performance')
ar_perf = pyEDM.ComputeError(reduction.model_array[start_idx:, 0], reduction.model_array[start_idx:, 1])
print(ar_perf)
print('SMap Residual Prediction Performance')
ar_res_perf = pyEDM.ComputeError(reduction.model_array[start_idx:, 2], reduction.model_array[start_idx:, 3])
print(ar_res_perf)
print('SMap + AR(' + str(optimal_dimension) + ') Performance')
ar_smap_perf = pyEDM.ComputeError(reduction.model_array[start_idx:, 0], reduction.model_array[start_idx:, 5])
print(ar_smap_perf)
rho_diff = ar_smap_perf['rho'] - ar_perf['rho']
print('Rho Diff: ' + str(rho_diff))
