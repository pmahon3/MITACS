import ray
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ray.util.multiprocessing import Pool

from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.estimators import nonlinearity, dimensionality
from edynamics.modelling_tools.optimizers import gnn_optimizer
from edynamics.modelling_tools.projectors import smap
from edynamics.modelling_tools.observers import lag
from edynamics.modelling_tools.weighers import exponential

from warnings import simplefilter

from sklearn.preprocessing import MinMaxScaler

simplefilter(action="ignore", category=FutureWarning)

ray.init(log_to_driver=False)
compute_pool = Pool()

# manual params
steps = 48
step_size = 1
target = "Load (kW)"
variables = [
    "Load (kW)",
    "Pressure_kpa",
    "Cloud Cover (%)",
    "Humidity (%)",
    "Wind Speed (kmh)",
]
method = "smap"
improvement_threshold = 0.0005
dimensions = 10
thetas = np.linspace(0, 10, 11)

type_set = ["weekdays", "saturdays", "sundays"]

# In Directories
data_file = (
    r"/home/pmahon/repositories/MITACS/src/main/resources/data/IEEE/pickles/main.pickle"
)

# Out Directories
plot_folders = ["./plots/dimensionalities/", "./plots/nonlinearities/"]
pickle_folder = ["./pickles/"]

for folder in plot_folders + pickle_folder:
    if not os.path.exists(folder):
        os.mkdir(folder)

# Load data
with open(data_file, "rb") as file:
    data = pickle.load(file)
file.close()

data.index = list(map(pd.Timestamp, data.index.values))
data.index.freq = data.index.inferred_freq

library_times = data.loc[
    pd.Timestamp("2020-09-01") : pd.Timestamp("2020-12-07 07:00:00")
].index
training_times = data.loc[
    pd.Timestamp("2020-12-07 07:00:00") : pd.Timestamp("2020-12-21 07:00:00")
].index
prediction_times = data.loc[
    pd.Timestamp("2020-12-21 07:00:00") : pd.Timestamp("2021-01-04 07:00:00")
].index

# scaling
load_min = data[target].loc[library_times].min()
load_max = data[target].loc[library_times].max()
scaler = MinMaxScaler(feature_range=(load_min, load_max))

for variable in variables:
    scaler.fit(data[variable].loc[library_times].values.reshape(-1, 1))
    data[variable] = scaler.transform(data[variable].values.reshape(-1, 1))

# Initialize model and embedding
embedding_ = embedding(
    library_times=library_times, data=data, observers=[lag(variable_name=target, tau=0)]
)

dimensionalities = dict.fromkeys(variables)
optimal_dimensions = dict.fromkeys(variables)
# Dimensionality Estimate
print("Dimensionality:")
for variable in variables:
    print(variable + ":")
    dimensionalities[variable] = dimensionality(
        embedding=embedding_,
        projector_=smap(),
        target=variable,
        points=data.loc[training_times],
        dimensions=dimensions,
        steps=1,
        step_size=1,
        compute_pool=compute_pool,
    )

    plt.plot(dimensionalities[variable], "-*")
    plt.title("Dimensionality of " + variable)
    plt.xlabel("Dimensions")
    plt.xticks(np.arange(1, dimensions + 1, 1))
    plt.ylabel("Prediction Skill (" + "\u03C1" + ")")
    plt.savefig(plot_folders[0] + variable + ".png")
    plt.close()

    dimensionality_ = dimensionalities[variable]

    optimal_dimensions[variable] = dimensionality_[
        dimensionality_["rho"] == dimensionality_["rho"].max()
    ].index[0]
print()

# Nonlinearity Estimate
nonlinearities = dict.fromkeys(variables)
optimal_thetas = dict.fromkeys(variables)
print("Nonlinearities:")
for variable in variables:
    print(variable + ":")
    opt_E = optimal_dimensions[variable]
    embedding_ = embedding(
        library_times=library_times,
        data=data,
        observers=[lag(variable_name=variable, tau=-i) for i in range(opt_E)],
    )
    embedding.compile()

    nonlinearity_ = nonlinearity(
        embedding=embedding_,
        projector_=smap(),
        target=variable,
        points=data.loc[training_times],
        thetas=thetas,
        steps=1,
        step_size=1,
        compute_pool=compute_pool,
    )

    nonlinearities[variable] = nonlinearity_

    plt.plot(nonlinearity_)
    plt.title("Nonlinearity of " + variable)
    plt.xlabel("Theta (" + "\u03B8" + ")")
    plt.xticks(thetas)
    plt.ylabel("Prediction Skill (" + "\u03C1" + ")")
    plt.savefig(plot_folders[1] + variable + ".png")
    plt.close()

    opt_Theta = nonlinearity_[nonlinearity_["rho"] == nonlinearity_["rho"].max()].index[
        0
    ]

    optimal_thetas[variable] = opt_Theta
print()

# Observation function optimization
print("GNN Observer Optimization")
for variable in variables:
    print(variable + ":")
    observers = [lag(variable_name="Load (kW)", tau=-i) for i in range(1, 49)]
    opt_theta = optimal_thetas[variable]

    optimal_observers = gnn_optimizer(
        embedding=embedding_,
        target=variable,
        observers=observers,
        projector_=smap(weigher_=exponential(theta=opt_theta)),
        points=data.loc[training_times],
        steps=24,
        step_size=1,
        compute_pool=compute_pool,
        improvement_threshold=0.0,
        verbose=False,
    )

    print([observer_.observation_name for observer_ in optimal_observers])

    with open(pickle_folder[0] + variable + " gnn observers.pickle", "rb") as file:
        pickle.dump(optimal_observers, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
