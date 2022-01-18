import pandas as pd
import pyEDM
import matplotlib.pyplot as plt

inFile = "../resources/data/processed_data_sets/ontario_averaged_first_diff_weekly_shuffle_2020.csv"
plotOutFile = '../resources/output/2020/nonlinearity/delta_demand_weekly_shuffle/plots/delta_demand_weekly_shuffle_'
dataOutFile = '../resources/output/2020/nonlinearity/delta_demand_weekly_shuffle/delta_demand_weekly_shuffle.csv'
plot_title = ""
maxE = 10
maxTp = 10
tau = -1
plot_title = "\n Demand"
variables = [
    'Time',
    'Demand'
]

ts = pd.DataFrame(pd.read_csv(inFile, index_col=0))
ts = ts.loc[:, variables]
ts.dropna(inplace=True)

lib = "1 " + str(len(ts))
pred = lib
nonlinearity = [[0 for i in range(maxTp)] for j in range(maxE)]
avg = [0 for i in range(maxE)]

for i in range(maxE):
    print("E: " + str(i+1))
    for j in range(maxTp):
        nonlinearity[i - 1][j - 1] = pd.DataFrame(
            pyEDM.PredictNonlinear(
                             dataFrame=ts,
                             lib=lib,
                             pred=pred,
                             E=i+1,
                             Tp=j+1,
                             tau=-1,
                             columns="Demand",
                             target="Demand",
                             showPlot=False)
        )

for i in range(maxE):
    avg_ = nonlinearity[i][1]
    for j in range(maxTp):
        if j == 1:
            continue
        else:
            avg_['rho'] = avg_['rho'] + nonlinearity[i][j]['rho']
        plt.plot(nonlinearity[i][j]['Theta'], nonlinearity[i][j]['rho'], color='blue', alpha=(maxTp-j)/maxTp)
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'\rho')
    avg_['rho'] = avg_['rho'] / 10
    avg[i] = avg_
    plt.plot(avg_['Theta'], avg_['rho'], color='red', alpha=1.0)
    plt.title("E = " + str(i + 1) + plot_title)
    plt.savefig(plotOutFile + "E_" + str(i+1) + "_2020.png")
    plt.cla()

for i in range(maxE):
    plt.plot(avg[i]['Theta'], avg[i]['rho'], color='red', alpha=(maxE-i)/maxE)
plt.title("Mean Prediction Skill for Varying E")
plt.xlabel(r'$\theta$')
plt.ylabel(r"$\rho$")
plt.savefig(plotOutFile + "average" + "_2020.png")
plt.cla()

pd.DataFrame(nonlinearity).to_csv(dataOutFile)