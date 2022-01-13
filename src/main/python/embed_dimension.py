# Used for generating plots of embedding dimension skill, via EmbedDimension, for power Demand
import numpy as np
import pandas as pd
import pyEDM
import matplotlib.pyplot as plt

inFile = "./main/resources/ontario_averaged_2020.csv"
outFile = './main/resources/plots/2020/embed_dimensions_plots/demand_embed_dim_'
maxTp = 10
maxTau = 10
plot_title = "\n Demand"

ts = pd.DataFrame(pd.read_csv(inFile)['Demand'])
ts.insert(0, 'Time', np.linspace(1, len(ts), len(ts)))
ts.dropna(inplace=True)

lib = "1 " + str(len(ts))
pred = lib
embedDimensions = [[0 for i in range(maxTau)] for j in range(maxTp)]

for i in range(maxTp):
    for j in range(maxTau):
        embedDimensions[i - 1][j - 1] = pd.DataFrame(
            pyEDM.EmbedDimension(dataFrame=ts,
                                 lib=lib,
                                 pred=pred,
                                 columns="Demand",
                                 target="Demand",
                                 maxE=10,
                                 Tp=i+1,
                                 tau=-(j+1),
                                 showPlot=False)
        )

for i in range(maxTp):
    avg = embedDimensions[i][1]
    for j in range(maxTau):
        if j == 1:
            continue
        else:
            avg['rho'] = avg['rho'] + embedDimensions[i][j]['rho']
        plt.plot(embedDimensions[i][j]['rho'], color='blue', alpha=(10-j)/10)
        plt.xlabel('E')
        plt.ylabel('rho')
    avg['rho'] = avg['rho'] / 10
    plt.plot(avg['rho'], color='red', alpha=0.7)
    plt.title("Tp = " + str(i + 1) + plot_title)
    plt.savefig(outFile + "tp_" + str(i+1) + "_2020.png")
    plt.cla()
