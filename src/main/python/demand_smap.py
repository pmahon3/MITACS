from Surrogate import *
import pandas as pd
import pyEDM
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import scipy.signal as signal

# parameters
in_file = '../resources/data/processed_data_sets/temp/forecast_demand.csv'
n_surrogates = 100
lib_length = 365 * 24 * 3

# data read_in
data = pd.read_csv(in_file, index_col=0)

# library/prediction split
max_forecast_nan = max([index for index, row in data['Forecast'].iteritems() if pd.isna(row)])
lib_start = max_forecast_nan - lib_length
pred_end = len(data)
lib_end = lib_start + lib_length


data_library = data.iloc[lib_start:lib_end, :]
data_prediction = data.iloc[(lib_end + 1):pred_end, :]

lib = str(lib_start) + " " + str(lib_end)
pred = str(lib_end + 1) + " " + str(pred_end)

# phase randomization surrogates
surrogate_data = Surrogate(data['Demand'], n_surrogates)
surrogate_data.generate_surrogates()

for surrogate in surrogate_data.surrogates:
    model = ARIMA(surrogate, order=(6, 0, 0))
    fitted = model.fit()
    forecast, standard_error, confidence_interval = fitted.forecast(steps=pred_end - lib_end)
    rmse = np.mean((forecast - data_prediction) ^ 2)
    print(rmse)
