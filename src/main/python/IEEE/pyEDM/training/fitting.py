# There's a few things to keep in mind here when estimating parameters
#   static vs multi-step forecasting
#   smap vs simplex
#   type of day
#   lag optimization

# We want to optimize the dimension for each day first. Regardless of what lags we are using, if the original system
# has n dimensions, two different lag reconstructions of n dimensions are diffeomorphic to the original system, and
# thus, to one another. Therefore it might be best to first optimize the dimensionality. But do we optimize via simplex
# or smap? I think simplex is the best choice, it's parsimonious. After this we can optimize the lags, and then we can
# optimize theta for smap and consider it against simplex projection. In all cases we could consider optimization as
# via the error on a static or multi-step test set. Since the ultimate goal is a multi-step forecast, it is probably
# best to consider optimization on multi-step test sets.

# Other things that might be insightful to look at:
#   prediction decay
#   lyapunov estimates
#   ...

# First, for the exploration, simplex will be used to estimate the dimensionality over sets of multi-step and static
# forecasts, for each day type

import numpy as np
import pickle
import pyEDM

from copy import copy
from datetime import timedelta
from tqdm import tqdm

from edynamics.modelling_tools.blocks.Block import Block
from edynamics.modelling_tools.data_types.Lag import Lag
from edynamics.modelling_tools.models.Model import Model

from Code.src.main.python.utils.IEEE_helpers import *
from Code.src.main.python.IEEE.data_loaders.CompetitionLoader import CompetitionLoader
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

# Load params
parameters = pd.read_csv("./parameters.csv", header=0, index_col="parameter_name")

data_folder = parameters.loc["data_folder"].parameter_value
n_splits = parameters.loc["n_splits"].parameter_value
split_length = parameters.loc["split_length"].parameter_value
maxE = parameters.loc["maxE"].parameter_value
training_start = parameters.loc["training_start"].parameter_value
frequency = parameters.loc["frequency"].parameter_value
day_types = eval(parameters.loc["day_types"].parameter_value)
target = parameters.loc["target"].parameter_value
library_start = pd.Timestamp(parameters.loc["library_start"].parameter_value)
library_end = pd.Timestamp(parameters.loc["library_end"].parameter_value)
prediction_start = pd.Timestamp(parameters.loc["prediction_start"].parameter_value)
prediction_end = pd.Timestamp(parameters.loc["prediction_end"].parameter_value)
theta = float(parameters.loc["theta"].parameter_value)
minimum_lag = int(parameters.loc["minimum_lag"].parameter_value)
maximum_lag = int(parameters.loc["maximum_lag"].parameter_value)

# Load data
loader = CompetitionLoader(data_directory=data_folder)
data = loader.load_data(variable="Load_(kW)", frequency="H").to_frame()

# Define the masks
types_ = {
    weekdays.__name__: weekdays,
    saturdays.__name__: saturdays,
    sundays.__name__: sundays,
}

for type_ in types_.keys():
    # Initialize embedding and model
    block = Block(
        library_start=library_start,
        library_end=prediction_end,
        series=data,
        frequency=timedelta(hours=1),
        lags=[Lag(variable_name=target, tau=0)],
    )

    # Output containers
    output = dict()
    lag_pool = [Lag(target, l) for l in reversed(range(minimum_lag, maximum_lag + 1))]
    n_lags = len(lag_pool)

    # K-Best lag selection
    progress = tqdm(range(n_lags))
    for i in progress:
        progress.set_description("Iteration %i" % i)
        # Add a slot for a new potential lag
        block.lags = block.lags + [None]
        # Set minimum for current round
        minimum = (None, np.inf)

        # Try adding every other lag not already selected
        for lag in tqdm(lag_pool, leave=False):
            block.lags[-1] = lag
            block.compile(mask_function=types_[type_])
            # Starting points of each prediction period
            times = (
                pd.date_range(start=prediction_start, end=prediction_end, freq="D")
                .to_series()
                .apply(lambda x: x if types_[type_](x) else None)
                .dropna()
                .index
            )
            starting_times = block.get_points(times).index

            model = Model(block=block, target=target, theta=1.0)

            projections_index = model._build_prediction_index(
                index=starting_times, steps=24, step_size=1
            )
            projections = pd.DataFrame(
                index=projections_index, columns=block.frame.columns
            )

            # Run the multi-step predictions
            for start_time in starting_times:
                frame = block.frame.copy()[:start_time]
                frame.insert(0, "Time", frame.index)
                lib_start = 1
                for prediction_time in projections.xs(key=start_time).index:
                    frame.loc[prediction_time] = [prediction_time] + [
                        0 for i in range(frame.shape[1] - 1)
                    ]
                    lib_end = len(frame) - 2
                    pred_start = lib_end + 1
                    pred_end = pred_start + 1

                    projection = pyEDM.Simplex(
                        dataFrame=frame,
                        lib=str(lib_start) + " " + str(lib_end),
                        pred=str(pred_start) + " " + str(pred_end),
                        target=target,
                        columns=list(frame.columns[1:]),
                        embedded=True,
                    )
                    projections.loc[(start_time, prediction_time)][target] = projection[
                        "Predictions"
                    ][1]
                    projections = model._update_lagged_values(
                        projections=projections,
                        current_time=start_time,
                        prediction_time=prediction_time,
                    )
                    frame.loc[prediction_time] = [prediction_time] + list(
                        projections.loc[(start_time, prediction_time)]
                    )

            projections = projections.astype("float64")
            # Compute errors
            times = projections.droplevel(level=0)[target].index

            points = block.get_points(times=times)

            error = projections.droplevel(level=0)[target] - points[target]
            mae = error.abs().mean()
            pmae = (error.abs() / points[target]).mean() * 100
            rmse = (error * error).sum() ** 0.5 / len(points)
            rho = projections[target].droplevel("Current_Time").corr(points[target])

            output[tuple([l.tau for l in block.lags])] = (mae, pmae, rmse, rho)

            # If better than the current best, update the minimum
            if mae < minimum[1]:
                minimum = (block.lags[-1], mae)

        # Remove the best performing lag from the lag pool and add to the model lags
        lag_pool.remove(minimum[0])
        block.lags[-1] = minimum[0]

    with open("simplex_" + type_ + ".pickle", "wb") as file:
        pickle.dump([parameters, output], file, protocol=pickle.HIGHEST_PROTOCOL)
