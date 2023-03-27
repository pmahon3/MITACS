import pickle
import os

from edynamics.modelling_tools.observers import lag
from edynamics.modelling_tools.embeddings import Embedding
from edynamics.modelling_tools.estimators import nonlinearity, dimensionality
from edynamics.modelling_tools.optimizers import gnn_lag_optimizer
from edynamics.modelling_tools.projectors import smap
from edynamics.modelling_tools.norms import minkowski
from edynamics.modelling_tools.weighers import exponential

from Code.src.main.python.utils.IEEE_helpers import *
from Code.src.main.python.IEEE.data_loaders.CompetitionLoader import CompetitionLoader
from datetime import timedelta
from warnings import simplefilter

import matplotlib.pyplot as plt

simplefilter(action='ignore', category=FutureWarning)

# manual params
steps = 24
step_size = 1
target = 'Load_(kW)'
method = 'smap'
improvement_threshold = 0.0

type_set = ['weekdays', 'saturdays', 'sundays']

data_folder = r'C:\Users\Patrick\OneDrive - The University of Western ' + \
              r'Ontario\Documents\Research\MITACS\Code\src\main\resources\data\IEEE'

# Load data
loader = CompetitionLoader(data_directory=data_folder)
data = loader.load_data(variable='Load_(kW)', frequency='H').to_frame()

splits = {
    'library': {data.loc[pd.Timestamp('2020-09-01'):pd.Timestamp('2021-01-18 7:00:00')].index},
    'prediction': {data.loc[pd.Timestamp('2021-01-18 07:00:00'):pd.Timestamp('2021-02-15 07:00:00')].index}
}

# Define the masks

# Initialize model and embedding
embedding = Embedding(
    library_times=None,
    data=data,
    observers=[lag(variable_name=target, tau=0)]
)