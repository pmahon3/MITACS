from main.python.NLHypothesisTesting.src.reduction.ARReduction import ARReduction
from main.python.NLHypothesisTesting.src.hypotheses.NonlinearHypothesis import NonlinearHypothesis
from main.python.NLHypothesisTesting.src.statistics.SMapPrediction import SMapPrediction
from main.python.NLHypothesisTesting.src.example_data.differentials import lorenz
from matplotlib import pyplot as plt

import statsmodels.api as sm
import pickle


lorenz_x, lorenz_y, lorenz_z = lorenz(steps=1000, noise_scale=0.01)
plt.plot(lorenz_x)
plt.cla()

hypothesis = NonlinearHypothesis(lorenz_x)
statistic = SMapPrediction(
    hypothesis=hypothesis,
    embedding_dimension=0,
    library='1 ' + str(len(lorenz_x)),
    prediction='1 ' + str(len(lorenz_x))
)
reduction = ARReduction(statistic)
reduction.compute_parameter_performance(max_dim=5, max_theta=5, tau=-1, performance_measure='rho')
reduction.fit_and_predict_arima()

plt.plot(reduction.statistic.hypothesis.time_series[-750:])
plt.show()
plt.plot(reduction.ar_residuals[-750:])
plt.show()
fig = sm.qqplot(reduction.ar_residuals[-750:])
plt.show()

reduction.optimize_ar_residuals_smap()
reduction.residual_statistic.set_optimal_parameters()
reduction.predict_ar_residuals(0.8)
