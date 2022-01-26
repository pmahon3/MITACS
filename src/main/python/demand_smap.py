from Surrogate import *
import numpy as np
import pandas as pd

in_file = '../resources/data/processed_data_sets/temp/forecast_demand.csv'
data = pd.read_csv(in_file, index_col=0)

surrogate_data = Surrogate(data, 100)
surrogate_data.generate_surrogates()
