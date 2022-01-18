# Used for generating plots of embedding dimension skill, via EmbedDimension, for power Demand
import pandas as pd
import pyEDM
import matplotlib.pyplot as plt

inFile = "../resources/data/processed_data_sets/ontario_averaged_first_diff_weekly_shuffle_2020.csv"
outFile = '../resources/plots/2020/ccm_plots/demand_'
E = 6
Tp = 1
tau = -1
libSizes = "10 300 10"
sample = 100
column = ['Demand']
targets = [
    'MEAN_TEMPERATURE',
#    'SPEED_MAX_GUST',
    'MAX_REL_HUMIDITY',
    'ALLSKY_SFC_SW_DWN'
]
ts = pd.DataFrame(pd.read_csv(inFile))
ts = ts.loc[:, ['Time'] + column + targets]

for target in targets:
    pyEDM.CCM(dataFrame = ts,
              E = E,
              Tp = Tp,
              tau = tau,
              columns = column,
              target = target,
              libSizes = libSizes,
              sample = sample,
              showPlot = True)