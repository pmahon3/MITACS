import pickle
from main.python.NLHypothesisTesting.src.visualizations.plot_3d_surface import plot_3d_surface

with open('delta_demand_E_theta_rho.pickle', 'rb') as file:
    E, theta, rho = pickle.load(file)
fig = plot_3d_surface(E, "E", theta, "Theta", rho, 'rho')
