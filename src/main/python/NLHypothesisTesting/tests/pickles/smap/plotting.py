import pickle
from main.python.NLHypothesisTesting.src.visualizations.plot_3d_surface import plot_3d_surface

with open('demand_E_tau_rho.pickle', 'rb') as file:
    E, tau, rho = pickle.load(file)
fig = plot_3d_surface(E, "E", tau, "tau", rho, 'rho')
