import numpy as np
import pickle
import matplotlib.pyplot as plt
import pyEDM

from datetime import timedelta
from tqdm import tqdm

from edynamics.modelling_tools.blocks.Block import Block
from edynamics.modelling_tools.data_types.Lag import Lag
from edynamics.modelling_tools.models.Model import Model

from Code.src.main.python.utils.IEEE_helpers import *
from Code.src.main.python.IEEE.data_loaders.CompetitionLoader import CompetitionLoader
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

# Load params
parameters = pd.read_csv('./parameters.csv', header=0, index_col='parameter_name')

data_folder = parameters.loc['data_folder'].parameter_value
library_start = pd.Timestamp(parameters.loc['library_start'].parameter_value)
library_end = pd.Timestamp(parameters.loc['library_end'].parameter_value)
prediction_start = pd.Timestamp(parameters.loc['prediction_start'].parameter_value)
prediction_end = pd.Timestamp(parameters.loc['prediction_end'].parameter_value)
target = parameters.loc['target'].parameter_value
error_measures = eval(parameters.loc['errors'].parameter_value)

# Load data
loader = CompetitionLoader(data_directory=data_folder)
data = loader.load_data(variable='Load_(kW)', frequency='H').to_frame()

# Define the masks
types_ = {weekdays.__name__: weekdays, saturdays.__name__: saturdays, sundays.__name__: sundays}

for type_ in types_.keys():
    # Load results
    with open('./' + type_ + '/simplex_' + type_ + '.pickle', 'rb') as file:
        _, output = pickle.load(file=file)
    file.close()

    # Process shortest lag TSP shortest path
    shortest = list(output.keys())[-1]
    # get best from each round
    errors = [_ for i in range(len(shortest))]
    for i, lag in enumerate(shortest):
        errors[i] = output[shortest[:i+2]]

    error_curves = dict.fromkeys(error_measures)
    for i, measure in enumerate(error_measures):
        error_curves[measure] = [x[i] for x in errors]
        if measure == 'rho' or measure == 'mae':
            plt.plot(error_curves[measure][1:])
            plt.xlabel('Added Lag')
            plt.ylabel(measure + '(kW)')
            plt.title('Lag Optimization using MAE Greedy Nearest Neighbour')
            plt.xticks([i for i in range(len(shortest)-1)], labels=shortest[1:])
            plt.xticks(rotation=45)
            figure = plt.gcf()
            figure.set_size_inches(16, 12)
            plt.savefig('./' + type_ + '/knn_gnn_' + measure + '_' + 'mae_' + type_ + '.png', dpi=200)
            plt.close()

    # Build Model
    block = Block(
        library_start=library_start,
        library_end=prediction_end,
        series=data,
        frequency=timedelta(hours=1),
        lags=[Lag(variable_name=target, tau=0)]
    )
    block.compile()
    model = Model(block=block, target=target, theta=0)

    # Subset times for given day type
    times = pd.date_range(start=library_start,
                          end=prediction_end,
                          freq='H').to_series().apply(lambda x: x if types_[type_](x) else None).dropna()

    # Set up pyEDM call
    frame = block.frame.loc[times]
    frame.insert(0, 'Time', frame.index)
    lib_start = 1
    lib_end = len(frame.loc[:library_end])
    pred_start = len(frame.loc[:prediction_start])
    pred_end = len(frame.loc[:prediction_end])
    dimensionality = pyEDM.EmbedDimension(
        dataFrame=frame,
        lib=str(lib_start) + ' ' + str(lib_end),
        pred=str(pred_start) + ' ' + str(pred_end),
        maxE=20,
        target=target,
        showPlot=False,
        columns=frame.columns[1:],
        verbose=True
    )

    # Plot
    plt.plot(dimensionality['E'], dimensionality['rho'])
    plt.title('Dimensionality of ' + type_)
    plt.xlabel('Dimension (lag multiples of -1)')
    plt.ylabel('Pearson Correlation')
    plt.xticks(ticks=dimensionality.index, labels=dimensionality.index, rotation=45)
    figure = plt.gcf()
    figure.set_size_inches(16, 12)
    plt.savefig('./' + type_ + '/knn_dimensionality_' + type_ + '.png', dpi=200)
    plt.close()
