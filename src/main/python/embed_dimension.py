# Used for generating plots of embedding dimension skill, via EmbedDimension, for power Demand
import pandas as pd
import pyEDM
import matplotlib.pyplot as plt

# inputs and outputs
inFile = "../resources/data/processed_data_sets/ontario_averaged_first_diff_2020.csv"
plotOutFile = '../resources/output/2020/embed_dimensions/delta_demand/plots/delta_demand_'
dataOutFile = '../resources/output/2020/embed_dimensions/delta_demand/delta_demand.csv'
plot_title = ""

# Variables
maxTp = 10
maxTau = 10
maxE = 10

# embedding is performed on variables[1]
variables = [
    'Time',
    'Demand'
]

# load in time series
ts = pd.DataFrame(pd.read_csv(inFile))
ts = ts.loc[:, variables]
ts.dropna(inplace=True)

# set library and prediction to whole time series
lib = "1 " + str(len(ts))
pred = lib

# initialize output containers
# raw output of py.EDM.EmbedDimension
embedDimensions = [[0 for i in range(maxTau)] for j in range(maxTp)]
# averaged output for rho vs. tp for a given E
tpCurves = [[0 for i in range(maxTp)] for j in range(maxE)]
# averaged output if rho vs tp
avg = [0 for j in range(maxTp)]

# perform embedding for across tp at various tau
for i in range(maxTp):
    for j in range(maxTau):
        embedDimensions[i][j] = pd.DataFrame(
            pyEDM.EmbedDimension(dataFrame=ts,
                                 lib=lib,
                                 pred=pred,
                                 columns="Demand",
                                 target="Demand",
                                 maxE=maxE,
                                 Tp=i+1,
                                 tau=-(j+1),
                                 showPlot=False)
        )

# plot rho vs. E for embeddings (varied over tau) for a given tp
for i in range(maxTp):
    avg_ = embedDimensions[i][1]
    for j in range(maxTau):
        if j == 1:
            continue
        else:
            avg_['rho'] = avg_['rho'] + embedDimensions[i][j]['rho']
        plt.plot(embedDimensions[i][j]['E'], embedDimensions[i][j]['rho'], color='blue', alpha=(maxTau-j)/maxTau)
        plt.xlabel('E')
        plt.ylabel('rho')
    avg_['rho'] = avg_['rho'] / 10
    avg[i] = avg_
    plt.plot(avg_['E'], avg_['rho'], color='red', alpha=1)
    plt.title("Prediction Skill for various embeddings (tau) with Tp = " + str(i + 1))
    plt.savefig(plotOutFile + "tp_" + str(i+1) + "_2020.png")
    plt.cla()

# plot rho vs. E for means of embeddings for a given tp
for i in range(maxTp):
    plt.plot(avg[i]['E'], avg[i]['rho'], color='red', alpha=(maxTp-i)/maxTp)
plt.title("Mean Prediction Skill for Varying Tp")
plt.xlabel('E')
plt.ylabel(r"$\rho$")
plt.savefig(plotOutFile + "average_E_skill" + "_2020.png")
plt.cla()

# compute rho vs. E across various embeddings (tau)
for i in range(maxE):
    avg_E = 0
    for j in range(maxTp):
        tpCurves[i][j] = avg[j].iloc[i]['rho']
    plt.plot(avg[1]['E'], tpCurves[i], color='blue', alpha=(maxE-i)/maxE)
plt.xlabel('Tp')
plt.ylabel(r'$\rho$')
plt.title('Time to Prediction Skill for Various Embeddings (E)')
plt.savefig(plotOutFile + "average_tp_skill.png")

# save output data
pd.DataFrame(embedDimensions).to_csv(dataOutFile)
