import numpy as np
import pandas as pd
import pyEDM
import matplotlib.pyplot as plt

inFile = "./main/resources/ontario_averaged_2020.csv"
outFile = './main/resources/plots/2020/nonlinearity_plots/demand_nonlinearity_'
maxE = 10
maxTp = 10
tau = -1
plot_title = "\n $\Delta$ Demand"

ts = pd.DataFrame(pd.read_csv(inFile)['Demand'])
ts.insert(0, 'Time', np.linspace(1, len(ts), len(ts)))
ts.dropna(inplace=True)

lib = "1 " + str(len(ts))
pred = lib
nonlinearity = [[0 for i in range(maxTp)] for j in range(maxE)]

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
    avg = nonlinearity[i][1]['rho']
    for j in range(maxTp):
        if j == 1:
            continue
        else:
            avg = avg + nonlinearity[i][j]['rho']
        plt.plot(nonlinearity[i][j]['Theta'],nonlinearity[i][j]['rho'], color='blue', alpha=(10-j)/10)
        plt.xlabel('$\Theta')
        plt.ylabel('rho')
    avg = avg / 10
    plt.plot(nonlinearity[i][j]['Theta'], avg, color='red', alpha=1.0)
    plt.title("E = " + str(i + 1) + plot_title)
    plt.savefig(outFile + "E_" + str(i) + "_2020.png")
    plt.cla()