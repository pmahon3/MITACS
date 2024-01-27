import os
import pickle
import uuid

import matplotlib.pyplot as plt
import pandas as pd
from edynamics.modelling_tools import Embedding
from edynamics.modelling_tools import Lag
from edynamics.modelling_tools.estimators import dimensionality, nonlinearity, greedy_nearest_neighbour
from edynamics.modelling_tools.projectors import Projector

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ray.util.multiprocessing import Pool

from typing import Type, TypeVar


def load_data(
        data_file: str,
        library_times: pd.DatetimeIndex,
        target: str
):
    """
    Loads and builds an embedding with an observer of the target variable

    :return Embedding: an embedding with library times defined, observers set to a 'target' variable to forecast.
    """
    with open(data_file, "rb") as file:
        data = pickle.load(file)
    file.close()

    data.index = list(map(pd.Timestamp, data.index.values))
    data = data[~data.index.duplicated(keep="first")]
    data = data.asfreq(library_times.freq)

    # scaling
    variable_min = data[target].loc[library_times].min()
    variable_max = data[target].loc[library_times].max()
    scaler = MinMaxScaler(feature_range=(variable_min, variable_max))

    for variable in data.columns:
        scaler.fit(data[variable].loc[library_times].values.reshape(-1, 1))
        data[variable] = scaler.transform(data[variable].values.reshape(-1, 1))

    # Initialize model and embedding
    embedding = Embedding(
        library_times=library_times,
        data=data,
        observers=[Lag(variable_name=target, tau=0)],
        compile_block=False
    )

    return embedding


# noinspection DuplicatedCode
def fit_dimensionality(fitting_set_index,
                       training_embedding,
                       variable,
                       types,
                       which_type,
                       max_dimensions,
                       steps,
                       step_size
                       ):
    figure_paths = []
    dimensions = dict.fromkeys(types.keys())

    for type_ in types.keys():
        # Set library times to just this day type
        times = fitting_set_index[fitting_set_index.map(which_type) == type_]

        training_embedding.set_library(
            library_times=times
        )

        # Fit max dimensions
        dimensionality_ = dimensionality(
            embedding=training_embedding,
            target=variable,
            times=times,
            max_dimensions=max_dimensions,
            steps=steps,
            step_size=step_size
        )

        # Get max skill
        dimensions[type_] = dimensionality_[dimensionality_['rho'] == dimensionality_['rho'].max()].index[0]

        # Plotting
        unique_id = uuid.uuid4()
        figure_paths.append(os.path.join(os.getenv('PYTEST_REPORT_IMAGES'),
                                         f'dimensionality_{types[type_]}_{unique_id}.png'))

        dimensionality_.plot()
        plt.title(f'Hour Ahead K-NN Projection - {variable} - {types[type_]}')
        plt.axvline(x=dimensions[type_], linestyle='--', color='r')
        plt.ylabel('Rho')
        plt.xlabel('Dimension')
        plt.savefig(figure_paths[-1])
        plt.close()

    return dimensions, figure_paths


# noinspection DuplicatedCode
def fit_nonlinearity(training_set_indices,
                     training_embedding,
                     variable,
                     types,
                     which_type,
                     dimensions,
                     steps,
                     step_size
                     ):
    figure_paths = []
    thetas = dict.fromkeys(types.keys())
    for type_ in types.keys():
        # Set library times to this day type and compile
        times = training_set_indices[training_set_indices.map(which_type) == type_]

        training_embedding.set_library(
            library_times=times
        )

        # Set observers to n lags of variable
        training_embedding.set_observers(
            observers=[Lag(variable_name=variable, tau=-i) for i in range(dimensions[type_])],
            compile_block=True
        )

        # Set initial times for multistep prediction
        times = times[times.hour == 7]

        # Fit weighting kernel

        nonlinearity_ = nonlinearity(
            embedding=training_embedding,
            target=variable,
            steps=steps,
            step_size=step_size,
            times=times[int(len(times) / 2):]
        )

        thetas[type_] = nonlinearity_[nonlinearity_['rho'] == nonlinearity_['rho'].max()].index[0]

        # Plotting
        unique_id = uuid.uuid4()
        figure_paths.append(os.path.join(os.getenv('PYTEST_REPORT_IMAGES'),
                                         f'nonlinearity_{types[type_]}_{unique_id}.png'))
        nonlinearity_.plot()
        plt.title(f'Day Ahead Weighted Least Squares Projection - {variable} - {types[type_]}')
        plt.xlabel('Kernel Locality Parameter (Theta)')
        plt.ylabel('Rho')
        plt.savefig(figure_paths[-1])

    return thetas, figure_paths


def fit_greedy_nearest_neighbours(training_embedding: Embedding,
                                  target: str,
                                  projector: Projector,
                                  times: pd.DatetimeIndex,
                                  max_dimensions: int,
                                  steps: int,
                                  step_size: int,
                                  improvement_threshold: float = -np.inf,
                                  compute_pool: Pool = None,
                                  verbose: bool = False
                                  ):
    """
    :param training_embedding: The embedding containing the training data.
    :type training_embedding: Embedding
    :param target: The target variable to predict.
    :type target: str
    :param projector: The projector used to project the embedding onto a lower-dimensional space.
    :type projector: Projector
    :param times: The timestamps for the training data.
    :type times: pd.DatetimeIndex
    :param max_dimensions: The maximum number of lag variables to consider.
    :type max_dimensions: int
    :param steps: The number of steps to perform in the greedy nearest neighbors algorithm.
    :type steps: int
    :param step_size: The step size used in the greedy nearest neighbors algorithm.
    :type step_size: int
    :param improvement_threshold: The threshold for improvement in the target variable to continue the algorithm. Defaults to -inf.
    :type improvement_threshold: float
    :param compute_pool: The multiprocessing pool to use for parallel computation. Defaults to None.
    :type compute_pool: Pool
    :return: The result of the greedy nearest neighbors algorithm.
    :rtype: Any
    """
    observers = [Lag(variable_name=target, tau=-i) for i in range(1, max_dimensions)]
    return greedy_nearest_neighbour(embedding=training_embedding,
                                    target=target,
                                    observers=observers,
                                    projector=projector,
                                    times=times,
                                    steps=steps,
                                    step_size=step_size,
                                    improvement_threshold=improvement_threshold,
                                    compute_pool=compute_pool,
                                    verbose=verbose)
